import os
import pathlib

# ============================================================================
# 基础训练设置
# ============================================================================
MODEL = "MoCE_IR"  # "MoCE_IR" 或 "MoCE_IR_S"
EPOCHS = 100
BATCH_SIZE = 16  # 每个GPU的batch size
LR = 2e-4

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
BENCHMARKS = None
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
VAL_EVERY_N_EPOCH = 10  # 每多少个epoch做一次验证

# ============================================================================
# 路径设置
# ============================================================================ open_dataset_8_1_1_mini

DATA_FILE_DIR = "/home/cxhlab/lqj/data/open_dataset_8_1_1"
OUTPUT_PATH = "output/"
WBLOGGER = False
CKPT_DIR = "checkpoints"
NUM_GPUS = 2
NUM_WORKERS = 9

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
