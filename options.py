"""
简化的配置加载模块 - 直接从 config.py 读取配置
不再需要命令行参数，所有设置都在 config.py 中
"""
import argparse
import os
from pathlib import Path
import config


class ConfigNamespace:
    """简单的配置命名空间类，模拟 argparse.Namespace"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({', '.join(items)})"


def train_options():
    """
    从 config.py 加载所有配置并返回配置对象
    完全移除命令行参数解析，所有配置都在 config.py 中
    """
    # 根据 MODEL 选择对应的模型配置
    if config.MODEL == "MoCE_IR_S":
        model_cfg = config.MoCE_IR_S_CONFIG
        model_type = "MoCE"
    elif config.MODEL == "MoCE_IR":
        model_cfg = config.MoCE_IR_CONFIG
        model_type = "MoCE"
    elif config.MODEL == "ACFormer":
        model_cfg = config.ACFORMER_CONFIG
        model_type = "ACFormer"
    else:
        raise NotImplementedError(
            f"Model '{config.MODEL}' not found. Use 'MoCE_IR', 'MoCE_IR_S' or 'ACFormer'."
        )
    
    # 合并所有配置到一个命名空间对象
    data_file_dir = str(Path(config.DATA_FILE_DIR).expanduser())
    # 归一化路径：避免相对路径/波浪号导致的歧义
    data_file_dir = os.path.abspath(data_file_dir)

    # 如果路径不存在，尝试做一次“大小写 Data/data”纠正（常见于 /media 下手工输入）
    if not os.path.exists(data_file_dir):
        alt = None
        if "/media/" in data_file_dir:
            if "/data/" in data_file_dir:
                alt = data_file_dir.replace("/data/", "/Data/")
            elif "/Data/" in data_file_dir:
                alt = data_file_dir.replace("/Data/", "/data/")
        if alt and os.path.exists(alt):
            print(f"[Warn] data_file_dir 不存在：{data_file_dir}，自动改用：{alt}")
            data_file_dir = alt

    base_kwargs = dict(
        # 基础训练设置
        model=config.MODEL,
        moce_backbone=getattr(config, "MOCE_BACKBONE", "moce_ir"),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LR,
        de_type=config.DE_TYPE,
        trainset=config.TRAINSET,
        loss_type=config.LOSS_TYPE,
        patch_size=config.PATCH_SIZE,
        balance_loss_weight=config.BALANCE_LOSS_WEIGHT,
        fft_loss_weight=config.FFT_LOSS_WEIGHT,
        num_workers=config.NUM_WORKERS,
        accum_grad=config.ACCUM_GRAD,
        print_model=getattr(config, "PRINT_MODEL", False),
        check_val_every_n_epoch=getattr(config, "VAL_EVERY_N_EPOCH", 5),
        resume_from=config.RESUME_FROM,
        fine_tune_from=config.FINE_TUNE_FROM,
        checkpoint_id=config.CHECKPOINT_ID,
        benchmarks=config.BENCHMARKS,
        save_results=config.SAVE_RESULTS,

        # 性能相关
        deterministic=getattr(config, "DETERMINISTIC", False),
        benchmark=getattr(config, "BENCHMARK", True),
        precision=getattr(config, "PRECISION", "16-mixed"),
        tf32=getattr(config, "TF32", True),
        log_every_n_steps=getattr(config, "LOG_EVERY_N_STEPS", 50),
        prefetch_factor=getattr(config, "PREFETCH_FACTOR", 2),
        persistent_workers=getattr(config, "PERSISTENT_WORKERS", True),
        
        # 路径设置
        data_file_dir=data_file_dir,
        output_path=config.OUTPUT_PATH,
        wblogger=config.WBLOGGER,
        ckpt_dir=config.CKPT_DIR,
        num_gpus=config.NUM_GPUS,
    )

    if model_type == "MoCE":
        model_kwargs = dict(
            dim=model_cfg["dim"],
            num_blocks=model_cfg["num_blocks"],
            num_dec_blocks=model_cfg["num_dec_blocks"],
            latent_dim=model_cfg["latent_dim"],
            num_exp_blocks=model_cfg["num_exp_blocks"],
            num_refinement_blocks=model_cfg["num_refinement_blocks"],
            heads=model_cfg["heads"],
            stage_depth=model_cfg["stage_depth"],
            with_complexity=model_cfg["with_complexity"],
            complexity_scale=model_cfg["complexity_scale"],
            rank_type=model_cfg["rank_type"],
            depth_type=model_cfg["depth_type"],
            topk=model_cfg["topk"],
        )
    else:  # ACFormer
        model_kwargs = dict(
            ac_dim=model_cfg["dim"],
            ac_num_blocks=model_cfg["num_blocks"],
            ac_num_refinement_blocks=model_cfg["num_refinement_blocks"],
            ac_channel_heads=model_cfg["channel_heads"],
            ac_spatial_heads=model_cfg["spatial_heads"],
            ac_overlap_ratio=model_cfg["overlap_ratio"],
            ac_window_size=model_cfg["window_size"],
            ac_spatial_dim_head=model_cfg["spatial_dim_head"],
            ac_ffn_expansion_factor=model_cfg["ffn_expansion_factor"],
            ac_bias=model_cfg["bias"],
            ac_layernorm_type=model_cfg["LayerNorm_type"],
            ac_M=model_cfg["M"],
            ac_ca_heads=model_cfg["ca_heads"],
            ac_ca_dim=model_cfg["ca_dim"],
            ac_window_size_ca=model_cfg["window_size_ca"],
            ac_query_ksize=model_cfg["query_ksize"],
            ac_use_ca=model_cfg["use_ca"],
        )

    options = ConfigNamespace(**base_kwargs, **model_kwargs)
    
    # Adjust batch size if gradient accumulation is used
    if options.accum_grad > 1:
        options.batch_size = options.batch_size // options.accum_grad
    
    return options