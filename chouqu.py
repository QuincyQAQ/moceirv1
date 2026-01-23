import argparse
import os
import random
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="源图片文件夹（target）")
    parser.add_argument("--dst", type=str, required=True, help="目标文件夹（抽样后的图片会复制到这里）")
    parser.add_argument("--n", type=int, default=20000, help="要抽取的图片数量，默认 20000")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    n = args.n

    if not src_dir.is_dir():
        raise ValueError(f"源文件夹不存在: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_imgs = [p for p in src_dir.rglob("*") if p.suffix.lower() in exts]

    if not all_imgs:
        raise ValueError(f"在 {src_dir} 下没有找到图片文件")

    print(f"源文件夹共有 {len(all_imgs)} 张图片")
    if len(all_imgs) <= n:
        selected = all_imgs
        print(f"图片总数少于等于 {n}，将全部复制过去")
    else:
        selected = random.sample(all_imgs, n)
        print(f"随机抽取 {n} 张图片")

    for i, src_path in enumerate(selected, 1):
        # 保持文件名不变；如果你希望保留子目录结构，可以自行改逻辑
        dst_path = dst_dir / src_path.name
        # 如果重名，给文件名加上编号
        if dst_path.exists():
            stem = src_path.stem
            suffix = src_path.suffix
            k = 1
            while True:
                new_name = f"{stem}_{k}{suffix}"
                dst_path = dst_dir / new_name
                if not dst_path.exists():
                    break
                k += 1
        shutil.copy2(src_path, dst_path)
        if i % 1000 == 0 or i == len(selected):
            print(f"已复制 {i}/{len(selected)}")

    print("完成。")

if __name__ == "__main__":
    main()