import argparse
import os

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor, ToPILImage

from options import train_options
from train import PLTrainModel
from utils.image_utils import crop_img


def load_model(ckpt_path: str, device: torch.device):
    """Load PLTrainModel and its MoCE-IR network from a Lightning checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    opt = train_options()

    # 1) 优先按 Lightning 标准 checkpoint 加载；
    # 2) 如果缺少 'pytorch-lightning_version' 等元数据（我们自己保存的 weights-only ckpt），
    #    则退回到手动加载 state_dict 的路径。
    try:
        model = PLTrainModel.load_from_checkpoint(ckpt_path, opt=opt, map_location=device)
        print(f"[Infer] Loaded Lightning checkpoint from: {ckpt_path}")
    except KeyError:
        print(f"[Infer] Detected weights-only checkpoint, loading state_dict from: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            # 兼容旧版 torch 没有 weights_only 参数的情况
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model = PLTrainModel(opt)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)
    return model, opt


def prepare_input(image_path: str, device: torch.device):
    """Load a degraded RGB image, crop to multiple of 16, and convert to tensor [1,3,H,W]."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Ensure spatial size is divisible by 16, consistent with training/validation.
    img_np = crop_img(img_np, base=16)

    to_tensor = ToTensor()
    tensor = to_tensor(img_np).unsqueeze(0)  # [1, 3, H, W], in [0,1]
    return tensor.to(device)


def restore_image(model: PLTrainModel, opt, degraded: torch.Tensor, device: torch.device):
    """Run the MoCE-IR network on a single degraded image tensor and return restored tensor."""
    # Determine degradation id for deblur. For current config, DE_TYPE is ["deblur"].
    de_type_list = getattr(opt, "de_type", ["deblur"])
    try:
        de_idx = de_type_list.index("deblur")
    except ValueError:
        # Fallback: use the first degradation type
        de_idx = 0

    de_id = torch.full((degraded.size(0),), de_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        restored = model.net(degraded, de_id)
        restored = torch.clamp(restored, 0.0, 1.0)

    return restored


def save_output(restored: torch.Tensor, output_path: str):
    """Save restored tensor [1,3,H,W] or [3,H,W] to an image file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if restored.dim() == 4:
        restored = restored.squeeze(0)

    to_pil = ToPILImage()
    img = to_pil(restored.cpu())
    img.save(output_path)



def main():
    parser = argparse.ArgumentParser(description="Restore a single image using a trained MoCE-IR model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/media/wsqlab/more/lqj/moceir/moceir/experiment/2026_01_22_20_18_57/checkpoints/best_psnr_ssim-epoch=89-psnr=25.193-ssim=0.7150.ckpt",
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/media/wsqlab/more/lqj/data/DMDiff_ICCV2025/lr_test/lr/div_000001.png",
        help="Path to degraded input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="1.png",
        help="Path to save restored image (default: 1.png)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model, opt = load_model(args.ckpt, device)

    # Prepare input image
    degraded = prepare_input(args.input, device)

    # Run restoration
    restored = restore_image(model, opt, degraded, device)

    # Save output
    save_output(restored, args.output)

    print(f"Restored image saved to: {args.output}")


if __name__ == "__main__":
    main()
