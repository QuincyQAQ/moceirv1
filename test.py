import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pathlib
import argparse
import importlib
import importlib.util
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm
from typing import List
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image

from utils.test_utils import save_img
from utils.image_utils import crop_img
from data.dataset_utils import IRBenchmarks, CDD11



####################################################################################################
## HELPERS
def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)

def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True, data_range = 1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def calc_psnr(img1, img2, data_range=1.0):
    err = np.sum((img1 - img2) ** 2, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / (err / img1.size))

def calc_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2, gaussian_weights=True, data_range = 1.0, full=False)



####################################################################################################
## HELPERS
def _forward_model(net: nn.Module, x: torch.Tensor, de_id: torch.Tensor) -> torch.Tensor:
    try:
        return net(x, de_id)
    except TypeError:
        return net(x)


def _load_module_from_file(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_torch_load(path: str):
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`.*",
    )

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def _extract_net_state_dict(state_dict: dict) -> dict:
    if any(str(k).startswith("net.") for k in state_dict.keys()):
        return {k[len("net."):]: v for k, v in state_dict.items() if str(k).startswith("net.")}
    return state_dict


def _resolve_ckpt_path(opt) -> pathlib.Path:
    ckpt_path = getattr(opt, "ckpt_path", None)
    if ckpt_path:
        return pathlib.Path(str(ckpt_path)).expanduser()

    ckpt_dir = getattr(opt, "ckpt_dir", None)
    checkpoint_id = getattr(opt, "checkpoint_id", None)
    if not ckpt_dir or not checkpoint_id:
        raise ValueError("Either opt.ckpt_path or (opt.ckpt_dir and opt.checkpoint_id) must be set")

    if str(checkpoint_id).lower().endswith(".ckpt"):
        return (pathlib.Path(str(ckpt_dir)) / str(checkpoint_id)).expanduser()
    return (pathlib.Path(str(ckpt_dir)) / str(checkpoint_id) / "last.ckpt").expanduser()


def _load_net_module(opt, ckpt_path: pathlib.Path):
    ckpt_path = ckpt_path.resolve()
    if ckpt_path.parent.name == "checkpoints":
        run_dir = ckpt_path.parent.parent
    else:
        run_dir = ckpt_path.parent
    net_snapshot_dir = run_dir / "net_snapshot"

    if net_snapshot_dir.exists() and net_snapshot_dir.is_dir():
        py_files = sorted(net_snapshot_dir.glob("*.py"))
        if len(py_files) == 0:
            pass
        elif len(py_files) == 1:
            net_file = py_files[0]
            module = _load_module_from_file(net_file, f"net_snapshot_{net_file.stem}")
            return module
        else:
            model_name = getattr(opt, "model", None)
            if model_name:
                cand = net_snapshot_dir / f"{model_name}.py"
                if cand.exists():
                    module = _load_module_from_file(cand, f"net_snapshot_{model_name}")
                    return module
            raise RuntimeError(f"Multiple .py files found in net_snapshot: {net_snapshot_dir}")

    project_dir = pathlib.Path(__file__).resolve().parent
    model_name = getattr(opt, "model", None)
    if not model_name:
        raise RuntimeError("opt.model is required when net_snapshot is not available")
    return importlib.import_module(f"net.{model_name}")



####################################################################################################
## PL Test Model
class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        module = _load_net_module(opt, _resolve_ckpt_path(opt))
        build_fn = getattr(module, "build_model", None)
        if build_fn is None:
            raise AttributeError("Network module must define build_model(opt)")
        self.net = build_fn(opt)
    
    def forward(self,x):
        return self.net(x)


####################################################################################################
## DRMI Test Dataset (test/meta + test/ground_truth)
class DRMITestDataset(Dataset):
    """Simple test dataset for DRMI_dataset.

    Expected structure under data_file_dir:
        test/meta/*.png          (degraded)
        test/ground_truth/*.png  (ground truth)
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.toTensor = ToTensor()
        self.de_type = self.args.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        # use 'deblur' id if present, otherwise 0
        self.de_id = self.de_dict.get("deblur", 0)
        self.patch_size = getattr(self.args, "patch_size", 128)
        self.full_res_eval = bool(getattr(self.args, "full_res_eval", False))

        data_dir = self.args.data_file_dir
        lr_dir = os.path.join(data_dir, "test", "meta")
        hr_dir = os.path.join(data_dir, "test", "ground_truth")

        self.lr = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
        self.hr = sorted(glob.glob(os.path.join(hr_dir, "*.png")))

        if len(self.lr) == 0 or len(self.hr) == 0:
            raise ValueError(f"No DRMI test images found under {lr_dir} and {hr_dir}")
        if len(self.lr) != len(self.hr):
            raise ValueError(f"LR/HR count mismatch in DRMI test set: {len(self.lr)} vs {len(self.hr)}")

    def __len__(self):
        return len(self.lr)

    def _center_crop_patch(self, img: np.ndarray) -> np.ndarray:
        """Center crop to patch_size x patch_size (if larger)."""
        h, w = img.shape[:2]
        p = int(self.patch_size)
        if h <= p or w <= p:
            return img

        top = (h - p) // 2
        left = (w - p) // 2
        return img[top:top + p, left:left + p, :]

    def __getitem__(self, idx):
        lr_path = self.lr[idx]
        hr_path = self.hr[idx]

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_np = np.array(lr_img)
        hr_np = np.array(hr_img)

        # Ensure size is multiple of 16 (same as training/validation) then take center patch
        lr_np = crop_img(lr_np, base=16)
        hr_np = crop_img(hr_np, base=16)

        if not self.full_res_eval:
            lr_np = self._center_crop_patch(lr_np)
            hr_np = self._center_crop_patch(hr_np)

        lr = self.toTensor(lr_np)
        hr = self.toTensor(hr_np)

        return [lr_path, self.de_id], lr, hr


####################################################################################################
def run_test(opts, net, dataset, factor=8):
    batch_size = getattr(opts, "batch_size", 1)
    testloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}")).mkdir(parents=True, exist_ok=True)
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()
    psnr, ssim, lpips = [], [], []
    with torch.no_grad():

        for ([clean_name, de_id], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
                        
            # Forward pass
            restored = _forward_model(net, degrad_patch, de_id)
            if isinstance(restored, List) and len(restored) == 2:
                restored , _ = restored
            
            # Unpad images to original dimensions
            assert restored.shape == clean_patch.shape, "Restored and clean patch shape mismatch."
            
            # save output images
            restored = torch.clamp(restored,0,1)
            lpips.append(calc_lpips(clean_patch, restored).cpu().numpy())
            
            restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            degrad_patch = degrad_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            clean = clean_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            ssim.append(calc_ssim(clean, restored))
            psnr_temp = peak_signal_noise_ratio(clean, restored, data_range=1)
            psnr.append(psnr_temp)
            
            if opts.save_results:
                save_name = os.path.splitext(os.path.split(clean_name[0])[-1])[0] + '_' + str(round(psnr_temp, 2)) +'.png'
                save_img(
                (os.path.join(os.getcwd(), 
                            f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}", 
                            save_name)), 
                img_as_ubyte(restored))

    print('PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n'.format(np.mean(psnr), np.mean(ssim), np.mean(lpips)))

            
## test LolV1
def run_lolv1(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)
    
## test GoPro
def run_gopro(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)
        
## test Derain
def run_derain(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)
        
## test Dehaze
def run_dehaze(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)
    
## test synthetic denoising
def run_denoise_15(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)
    
def run_denoise_25(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)
    
def run_denoise_50(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

# test CDD11
def run_cdd11(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

# test DRMI_dataset
def run_drmi(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


####################################################################################################
## main
def main(opt):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Load model
    ckpt_path = _resolve_ckpt_path(opt)
    if getattr(opt, "checkpoint_id", None) is None:
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
        opt.checkpoint_id = f"{run_dir.name}-{ckpt_path.stem}"
    print(f"[Test] Loading checkpoint from: {ckpt_path}")

    module = _load_net_module(opt, ckpt_path)
    build_fn = getattr(module, "build_model", None)
    if build_fn is None:
        raise AttributeError("Network module must define build_model(opt)")

    net = build_fn(opt)
    ckpt = _safe_torch_load(str(ckpt_path))
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _extract_net_state_dict(state_dict)
    net.load_state_dict(state_dict, strict=False)

    # Use all available GPUs via DataParallel to reduce per-GPU memory usage.
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            net = torch.nn.DataParallel(net)
        net = net.cuda()
    net.eval()
    
    for de in opt.benchmarks:
        ind_opt = opt
        ind_opt.benchmarks = [de]
        
        if de == "drmi":
            dataset = DRMITestDataset(ind_opt)
        elif "CDD11" in opt.trainset:
            _, subset = opt.trainset.split("_", maxsplit=1)
            dataset = CDD11(opt, split="test", subset=subset)
        else:
            dataset = IRBenchmarks(ind_opt)
        
        print("--------> Testing on", de, "testset.")
        print("\n")
        globals()[f"run_{de}"](opt, net, dataset, factor=8)
    

def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
if __name__ == '__main__':
    train_opt = argparse.Namespace(
        ckpt_path="/media/wsqlab/more/lqj/moceirv1-20260123_multi_network/experiment/MoCE_IR_S-2026_01_29_23_33_45/checkpoints/best_psnr_ssim-epoch=89-psnr=25.193-ssim=0.7150.ckpt",
        model=None,
        data_file_dir="/media/wsqlab/more/lqj/data/open_dataset_8_1_1",
        trainset="standard",
        benchmarks=["gopro"],
        de_type=["deblur"],
        patch_size=256,
        batch_size=32,
        save_results=True,
        full_res_eval=False,
    )
    main(train_opt)