from typing import List

import os
import pathlib
import csv
import numpy as np
from copy import deepcopy

from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from net.moce_ir import MoCEIR

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11, IRBenchmarks
from utils.loss_utils import FFTLoss



class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.balance_loss_weight = opt.balance_loss_weight
        # Optional CSV path for logging epoch-level metrics
        self.metrics_csv_path = getattr(opt, "metrics_csv", None)
        # LPIPS module for validation (Lightning will move it to the correct device)
        self.val_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True, reduction="mean"
        )

        self.net = MoCEIR(
            dim=opt.dim, 
            num_blocks=opt.num_blocks, 
            num_dec_blocks=opt.num_dec_blocks, 
            levels=len(opt.num_blocks),
            heads=opt.heads, 
            num_refinement_blocks=opt.num_refinement_blocks, 
            topk=opt.topk, 
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity, 
            depth_type=opt.depth_type, 
            stage_depth=opt.stage_depth, 
            rank_type=opt.rank_type, 
            complexity_scale=opt.complexity_scale,)
        
             
        if opt.loss_type == "fft":
            self.loss_fn = nn.L1Loss()
            self.aux_fn = FFTLoss(loss_weight=self.opt.fft_loss_weight)
        else:
            self.loss_fn = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch, de_id)
        balance_loss = self.net.total_loss

        if self.opt.loss_type == "fft":
            loss = self.loss_fn(restored,clean_patch)
            aux_loss = self.aux_fn(restored,clean_patch)
            loss += aux_loss
        else:
            loss = self.loss_fn(restored,clean_patch)
            
        loss += self.balance_loss_weight * balance_loss
        # NOTE: 在多卡下，step 级别的 sync_dist 会触发频繁的跨卡同步(all-reduce)，显著拖慢吞吐
        # 这里保留 step 曲线但不做跨卡同步；epoch 汇总再做同步，兼顾速度与可比性
        self.log("Train_Loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        self.log("Balance", balance_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        self.log("Train_Loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("Balance_epoch", balance_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR", lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Run validation on full-resolution images and compute PSNR/SSIM/LPIPS.

        Uses IRBenchmarks with generic test/lr & test/gt structure (e.g. open_dataset_8_1_1).
        """
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        # Forward pass
        restored = self.net(degrad_patch, de_id)
        restored = torch.clamp(restored, 0.0, 1.0)

        # LPIPS (batch-wise)
        lpips_val = self.val_lpips(clean_patch, restored)

        # Convert to numpy for PSNR / SSIM
        restored_np = (
            restored.detach().cpu().permute(0, 2, 3, 1).numpy()
        )  # (B, H, W, C), [0,1]
        clean_np = (
            clean_patch.detach().cpu().permute(0, 2, 3, 1).numpy()
        )  # (B, H, W, C), [0,1]

        psnr_vals = []
        ssim_vals = []
        for i in range(restored_np.shape[0]):
            psnr_vals.append(
                peak_signal_noise_ratio(clean_np[i], restored_np[i], data_range=1.0)
            )
            ssim_vals.append(
                structural_similarity(
                    clean_np[i],
                    restored_np[i],
                    data_range=1.0,
                    channel_axis=2,
                    gaussian_weights=True,
                )
            )

        psnr = float(np.mean(psnr_vals))
        ssim = float(np.mean(ssim_vals))

        psnr_t = clean_patch.new_tensor(psnr)
        ssim_t = clean_patch.new_tensor(ssim)

        # Log epoch-level metrics (Lightning aggregates over batches)
        self.log("val_psnr", psnr_t, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", ssim_t, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_lpips", lpips_val, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)
        
        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=1,max_epochs=self.opt.epochs)      
        return [optimizer],[scheduler]
    
    def _write_metrics_row(self, phase: str, metrics: dict):
        """Internal helper: append a row to metrics CSV if configured.

        phase: "val" or "test".
        """

        if not self.metrics_csv_path:
            return

        trainer = self.trainer
        if trainer is None or not trainer.is_global_zero:
            # Only write from global rank 0 in DDP
            return

        # Helper to safely convert tensors/values to float
        def _to_float(val):
            if val is None:
                return None
            try:
                return float(val)
            except Exception:
                return None

        row = {
            "phase": phase,
            "epoch": int(self.current_epoch),
            "Train_Loss_epoch": _to_float(metrics.get("Train_Loss_epoch")),
            "Balance_epoch": _to_float(metrics.get("Balance_epoch")),
            "val_psnr": _to_float(metrics.get("val_psnr")),
            "val_ssim": _to_float(metrics.get("val_ssim")),
            "val_lpips": _to_float(metrics.get("val_lpips")),
            "test_psnr": _to_float(metrics.get("test_psnr")),
            "test_ssim": _to_float(metrics.get("test_ssim")),
            "test_lpips": _to_float(metrics.get("test_lpips")),
        }

        # Current learning rate (from first optimizer)
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0].get("lr", None)
            row["lr"] = _to_float(lr)
        else:
            row["lr"] = None

        file_exists = os.path.exists(self.metrics_csv_path)
        fieldnames = list(row.keys())

        os.makedirs(os.path.dirname(self.metrics_csv_path), exist_ok=True)
        with open(self.metrics_csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def on_validation_epoch_end(self):
        """After each validation epoch, write aggregated val metrics to CSV."""
        if not self.metrics_csv_path:
            return

        metrics = self.trainer.callback_metrics if self.trainer is not None else {}
        self._write_metrics_row("val", metrics)

    def on_train_epoch_end(self):
        if not self.metrics_csv_path:
            return

        metrics = self.trainer.callback_metrics if self.trainer is not None else {}
        self._write_metrics_row("train", metrics)

    def test_step(self, batch, batch_idx):
        """Test step: same as validation but logs as test_* metrics."""
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        restored = self.net(degrad_patch, de_id)
        restored = torch.clamp(restored, 0.0, 1.0)

        lpips_val = self.val_lpips(clean_patch, restored)

        restored_np = (
            restored.detach().cpu().permute(0, 2, 3, 1).numpy()
        )
        clean_np = (
            clean_patch.detach().cpu().permute(0, 2, 3, 1).numpy()
        )

        psnr_vals = []
        ssim_vals = []
        for i in range(restored_np.shape[0]):
            psnr_vals.append(
                peak_signal_noise_ratio(clean_np[i], restored_np[i], data_range=1.0)
            )
            ssim_vals.append(
                structural_similarity(
                    clean_np[i],
                    restored_np[i],
                    data_range=1.0,
                    channel_axis=2,
                    gaussian_weights=True,
                )
            )

        psnr = float(np.mean(psnr_vals))
        ssim = float(np.mean(ssim_vals))

        psnr_t = clean_patch.new_tensor(psnr)
        ssim_t = clean_patch.new_tensor(ssim)

        self.log("test_psnr", psnr_t, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_ssim", ssim_t, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_lpips", lpips_val, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_test_epoch_end(self):
        """After the final test epoch, append test metrics to CSV."""
        if not self.metrics_csv_path:
            return

        metrics = self.trainer.callback_metrics if self.trainer is not None else {}
        self._write_metrics_row("test", metrics)
                        


def main(opt):
    print("Options")
    print(opt)
    # Use a shared run ID across all DDP processes so each launch
    # only creates a single experiment/<timestamp> directory.
    run_id = os.environ.get("MOCEIRV2_RUN_ID")
    if run_id is None:
        run_id = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        os.environ["MOCEIRV2_RUN_ID"] = run_id

    time_stamp = run_id

    # Anchor experiment directory to this file's directory to avoid CWD differences
    project_dir = pathlib.Path(__file__).resolve().parent
    base_exp_dir = project_dir / "experiment"
    base_exp_dir.mkdir(parents=True, exist_ok=True)

    # Per-run root directory under experiment/<timestamp>
    log_dir = base_exp_dir / time_stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # CSV metrics file path inside this run directory
    metrics_csv_path = log_dir / "metrics.csv"
    setattr(opt, "metrics_csv", metrics_csv_path)
    # Initialize metrics CSV so it exists even before first val/test
    if not os.path.exists(metrics_csv_path):
        fieldnames = [
            "phase",
            "epoch",
            "Train_Loss_epoch",
            "Balance_epoch",
            "val_psnr",
            "val_ssim",
            "val_lpips",
            "test_psnr",
            "test_ssim",
            "test_lpips",
            "lr",
        ]
        with open(metrics_csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    if opt.wblogger:
        name = opt.model + "_" + time_stamp
        logger  = WandbLogger(name=name, save_dir=log_dir, config=opt) 
        
    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    # Create model
    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt)
    else:
        model = PLTrainModel(opt)

    print(model)
    # Store checkpoints under experiment/<timestamp>/checkpoints
    checkpoint_path = log_dir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # Save only the best model (by highest val_psnr) and the last checkpoint
    # Validation is run every 50 epochs, so checkpoint saving follows the same interval.
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        every_n_epochs=5,
        save_top_k=1,
        save_last=True,
        monitor="val_psnr",
        mode="max",
    )
    
    # Create training dataset and dataloader
    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        trainset = AIOTrainDataset(opt)
        
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=opt.num_workers)
    # 更快的数据喂入：worker 常驻 + 预取（仅当 num_workers>0 时生效）
    if opt.num_workers and opt.num_workers > 0:
        trainloader = DataLoader(
            trainset,
            batch_size=opt.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=opt.num_workers,
            persistent_workers=getattr(opt, "persistent_workers", True),
            prefetch_factor=getattr(opt, "prefetch_factor", 2),
        )
    # Create validation dataset / dataloader (every 5 epochs)
    val_loader = None
    if "CDD11" not in opt.trainset:
        # For standard deblurring, use IRBenchmarks with generic test/lr & test/gt
        val_opt = deepcopy(opt)
        # For deblurring, IRBenchmarks expects a benchmark name; "gopro" is used
        # here but directory resolution is generic test/lr & test/gt under data_file_dir.
        val_opt.benchmarks = ["gopro"]
        valset = IRBenchmarks(val_opt)
        val_loader = DataLoader(
            valset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=opt.num_workers,
        )
    
    # Detect GPU availability
    if torch.cuda.is_available() and opt.num_gpus > 0:
        # 允许 TF32（Ampere+ 上通常能显著提速，精度影响很小；可在 config 里关掉）
        if getattr(opt, "tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        accelerator = "gpu"
        devices = opt.num_gpus
        strategy = "ddp_find_unused_parameters_true" if opt.num_gpus > 1 else "auto"
        print(f"Using GPU: {devices} GPU(s) available")
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
        print("GPU not available, using CPU")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=opt.accum_grad,
        deterministic=getattr(opt, "deterministic", False),
        benchmark=getattr(opt, "benchmark", True),
        precision=getattr(opt, "precision", "16-mixed"),
        log_every_n_steps=getattr(opt, "log_every_n_steps", 50),
        # Run validation every 50 epochs
        check_val_every_n_epoch=50,
    )
    
    # Optionally resume from a checkpoint stored under experiment/<run_id>/checkpoints
    if opt.resume_from:
        resume_ckpt = base_exp_dir / opt.resume_from / "checkpoints" / "last.ckpt"
        checkpoint_path = str(resume_ckpt)
    else:
        checkpoint_path = None

    # Train model (with optional validation loader)
    fit_kwargs = dict(
        model=model,
        train_dataloaders=trainloader,
        ckpt_path=checkpoint_path,
    )
    if val_loader is not None:
        fit_kwargs["val_dataloaders"] = val_loader

    trainer.fit(**fit_kwargs)

    # After training, run a final test pass on the same benchmark (e.g., test split)
    if "CDD11" not in opt.trainset:
        test_opt = deepcopy(opt)
        test_opt.benchmarks = ["gopro"]
        testset = IRBenchmarks(test_opt)
        test_loader = DataLoader(
            testset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=opt.num_workers,
        )
        trainer.test(model=model, dataloaders=test_loader)
    


if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)



