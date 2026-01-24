import os
import pathlib

# ============================================================================
# 基础训练设置
# ============================================================================
# MODEL 可选: "MoCE_IR", "MoCE_IR_S", "ACFormer"
MODEL = "MoCE_IR_S"  # "MoCE_IR" 或 "MoCE_IR_S" 或 "ACFormer"
MOCE_BACKBONE = "moce_w"  # "moce_ir" 或 "moce_w"（仅对 MoCE_IR/MoCE_IR_S 生效）
EPOCHS = 300
BATCH_SIZE = 16  # 每个GPU的“常规” batch size（warmup 结束后使用）
LR = 2e-4

# 是否启用 batch size warmup：前若干个 epoch 用小 batch，之后切换到 BATCH_SIZE
DYNAMIC_BATCH = False
WARMUP_BATCH_SIZE = 2  # warmup 阶段使用的小 batch（仅在 DYNAMIC_BATCH=True 时生效）
WARMUP_EPOCHS = 0      # 使用 WARMUP_BATCH_SIZE 的 epoch 数（0 表示不开启 warmup）

DE_TYPE = ["deblur"]  # 可选: "denoise_15/25/50", "dehaze", "derain", "deblur", "synllie"
TRAINSET = "standard"  # "standard" 或 "CDD11_*"
LOSS_TYPE = "L1"  # "L1" 或 "fft"
PATCH_SIZE = 128
BALANCE_LOSS_WEIGHT = 0.01
FFT_LOSS_WEIGHT = 1.0
ACCUM_GRAD = 1
PRINT_MODEL = False

RESUME_FROM = None  # 从checkpoint恢复训练
FINE_TUNE_FROM = None # 微调checkpoint
CHECKPOINT_ID = None
BENCHMARKS = ["gopro"]
SAVE_RESULTS = True

# ============================================================================
# 性能相关
# ============================================================================
DETERMINISTIC = False
BENCHMARK = True
PRECISION = "16-mixed"  # "16-mixed" 或 "bf16-mixed" (A100/H100)
TF32 = True
LOG_EVERY_N_STEPS = 10
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True
VAL_EVERY_N_EPOCH = 15  # 每多少个epoch做一次验证

# ============================================================================
# 路径设置
# ============================================================================ open_dataset_8_1_1_mini

OUTPUT_PATH = "output/"
WBLOGGER = False
NUM_GPUS = 2
NUM_WORKERS = 12


# DATA_FILE_DIR = "/media/wsqlab/more/lqj/data/open_dataset_8_1_1"
# CKPT_DIR = "checkpoints"



# 路径设置
DATA_FILE_DIR = "/media/wsqlab/more/lqj/data/open_dataset_8_1_1"
# ckpt 根目录：包含各个 experiment 子目录
CKPT_DIR = "/media/wsqlab/more/lqj/moceir/moceir/experiment"
# 具体要测的那一次实验的子目录（到 checkpoints 这一层）
CHECKPOINT_ID = "2026_01_22_20_18_57/checkpoints"  # 例子，换成你自己的


# DATA_FILE_DIR = "/media/wsqlab/more/lqj/data/DMDiff_ICCV2025_8_1_1"
# CKPT_DIR = "/media/wsqlab/more/lqj/moceir/moceir/experiment"
# CHECKPOINT_ID = "2026_01_22_20_18_57/checkpoints/best_psnr_ssim-epoch=89-psnr=25.193-ssim=0.7150.ckpt"


# ============================================================================
# 模型参数
# ============================================================================
MoCE_IR_S_CONFIG = {
    "dim": 32,
    "num_blocks": [4, 6, 6, 8],
    "num_dec_blocks": [2, 4, 4],
    "latent_dim": 2,
    "num_exp_blocks": 4,
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "stage_depth": [1, 1, 1],
    "with_complexity": False,
    "complexity_scale": "max",
    "rank_type": "spread",
    "depth_type": "constant",
    "topk": 1,
}

ACFORMER_CONFIG = {
    # 与 Metalens-Transformer 中的 ACFormer 默认配置保持一致，
    # 可按需调整。
    "dim": 48,
    "num_blocks": [2, 4, 4, 4],
    "num_refinement_blocks": 4,
    "channel_heads": [1, 2, 4, 8],
    "spatial_heads": [1, 2, 4, 8],
    "overlap_ratio": [0.5, 0.5, 0.5, 0.5],
    "window_size": 8,
    "spatial_dim_head": 16,
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "WithBias",
    "M": 13,
    "ca_heads": 2,
    "ca_dim": 32,
    "window_size_ca": 8,
    "query_ksize": [15, 11, 7, 3, 3],
    # 在 MoCEIR 框架里默认只用单张输入图像，因此默认关闭 CA 模块；
    # 如果以后你想按 Metalens-Transformer 的多通道输入方式使用，可改为 True。
    "use_ca": False,
}

MoCE_IR_CONFIG = {
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "num_dec_blocks": [2, 4, 4],
    "latent_dim": 2,
    "num_exp_blocks": 4,
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "stage_depth": [1, 1, 1],
    "with_complexity": False,
    "complexity_scale": "max",
    "rank_type": "spread",
    "depth_type": "constant",
    "topk": 1,
}
