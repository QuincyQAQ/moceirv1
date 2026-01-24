import os
import pathlib
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

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

from net.moce_ir import MoCEIR as MoCEIR_Base
from net.moce_w import MoCEIR as MoCEIR_W
from options import train_options
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
## PL Test Model
class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        moce_backbone = getattr(opt, "moce_backbone", "moce_ir")
        MoCEIRImpl = MoCEIR_W if moce_backbone == "moce_w" else MoCEIR_Base

        self.net = MoCEIRImpl(
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

        lr_np = self._center_crop_patch(lr_np)
        hr_np = self._center_crop_patch(hr_np)

        lr = self.toTensor(lr_np)
        hr = self.toTensor(hr_np)

        return [lr_path, self.de_id], lr, hr


####################################################################################################
def run_test(opts, net, dataset, factor=8):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)
    
    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}")).mkdir(parents=True, exist_ok=True)
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()
    psnr, ssim, lpips = [], [], []
    with torch.no_grad():

        for ([clean_name, de_id], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
                        
            # Forward pass
            restored = net(degrad_patch)
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
    # 支持两种用法：
    # 1) CHECKPOINT_ID = "2026_01_22_20_18_57/checkpoints"         -> 自动拼成 .../checkpoints/last.ckpt
    # 2) CHECKPOINT_ID = "2026_01_22_20_18_57/checkpoints/best*.ckpt" -> 直接使用该 ckpt 文件
    if opt.checkpoint_id and str(opt.checkpoint_id).lower().endswith('.ckpt'):
        ckpt_path = os.path.join(opt.ckpt_dir, opt.checkpoint_id)
    else:
        ckpt_path = os.path.join(opt.ckpt_dir, opt.checkpoint_id, "last.ckpt")

    print(f"[Test] Loading checkpoint from: {ckpt_path}")

    # 优先按 Lightning 标准 checkpoint 加载；
    # 若缺少 'pytorch-lightning_version' 等元数据（我们自己保存的 weights-only ckpt），
    # 则退回到手动加载 state_dict 的路径，与 train.py 中 fine_tune_from 的逻辑一致。
    try:
        # 训练时的 PLTrainModel state_dict 里包含 val_lpips.* 等额外模块，
        # PLTestModel 只有 MoCEIR 网络本身，这里用 strict=False 忽略多余键。
        net = PLTestModel.load_from_checkpoint(ckpt_path, opt=opt, strict=False).cuda()
        print(f"[Test] Loaded Lightning checkpoint from: {ckpt_path}")
    except KeyError:
        # 处理我们自己保存的 joint-best 权重 ckpt: {"state_dict": ...}
        print(f"[Test] Detected weights-only checkpoint, loading state_dict from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        model = PLTestModel(opt)
        model.load_state_dict(state_dict, strict=False)
        net = model.cuda()
    except Exception as e:
        raise RuntimeError(f"Failed to load test checkpoint: {ckpt_path}\n{e}")

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
    train_opt = train_options()
    main(train_opt)