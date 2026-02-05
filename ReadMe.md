

# MoCE-IR 多网络版本项目说明 (MoCE-IR Multi-Network Version)

本项目是基于 **MoCE-IR** 改进的通用图像恢复（IR）框架。它引入了多网络动态加载机制、分布式训练（DDP）支持以及完善的实验快照管理，旨在提升实验效率与结果的可复现性。

---

## 🛠 环境安装

请严格按照以下命令顺序配置开发环境：

```bash
# 1. 克隆特定修复分支
git clone -b 20260131_newframework_fixbug [https://github.com/QuincyQAQ/moceirv1.git](https://github.com/QuincyQAQ/moceirv1.git)
cd moceirv1

# 2. 创建并激活 Conda 环境
conda create -n moceir python=3.10.2 -y
conda activate moceir

# 3. 安装 PyTorch 核心组件 (推荐使用清华镜像源)
pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 torchmetrics==1.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装项目依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

```

---

## 📂 项目结构

建议将数据集与代码按照以下层级存放，以便脚本正确读取路径：

```text
root/
├── data/
│   └── open_dataset_8_1_1/          # 标准数据集根目录
│       ├── train/                   # 训练集 (包含 gt/lr 子文件夹)
│       ├── val/                     # 验证集 (包含 gt/lr 子文件夹)
│       └── test/                    # 测试集 (包含 gt/lr 子文件夹)
└── moceir/
    └── moceir/                      # 代码根目录
        ├── config.py                # 训练全局配置文件
        ├── train.py                 # 分布式训练入口
        ├── test.py                  # 独立测试脚本
        ├── infer_image.py           # 单图推理工具
        ├── net/                     # 存放不同模型的定义 (如 MoCE_IR_S.py, ACFormer.py)
        ├── experiment/              # 实验产出目录 (自动生成)
        ├── train.sh                 # 训练启动脚本
        ├── test.sh                  # 测试启动脚本
        └── ...                      

```

---

## 🚀 推荐使用流程

1. **全局配置**：编辑 `config.py`，设定 `MODEL` 名称（如 `MoCE_IR`）、数据集路径及训练超参数。
2. **启动训练**：执行 `bash train.sh`。系统将自动在 `experiment/` 下创建带时间戳的实验文件夹，训练完成后会自动进行基本测试。
3. **模型评估**：
* **标准评估**：执行 `bash test.sh`。默认开启多卡并行及全分辨率模式（`batch=1`）。
* **单卡评估**：如需手动调试，可运行 `python test.py`。


4. **单图推理**：
```bash
python infer_image.py --ckpt <权重路径> --input <输入图像> --output <保存路径>

```



---

## 📄 核心脚本功能详解

### 1. `train.py` (训练核心)

支持基于 Hugging Face `accelerate` 的 DDP 多卡并行训练。

* **动态加载**：根据 `config.py` 中的 `MODEL` 字段，自动从 `net/` 目录加载对应的模型结构。
* **自动归档**：在 `experiment/<模型名-时间戳>/` 下实时保存：
* `checkpoints/`：存放 `last.ckpt` 及性能最优的权重。
* `net_snapshot/`：**关键功能**。自动备份本次训练的网络入口文件，确保即使项目代码修改，该实验也可独立运行。
* `metrics.csv`：记录各 Epoch 的 PSNR/SSIM/LPIPS 指标。



### 2. `config.py` (配置中心)

统一管理训练与推理的静态参数，包括模型选型（如 `"MoCE_IR"`、`"ACFormer"`）、数据根目录、Worker 数量、混合精度及验证频率。

### 3. `test.py` (独立测试)

**完全脱离 `config.py` 运行**。用户需在文件末尾的 `argparse.Namespace` 中配置：

* **权重路径**：支持 Lightning Checkpoint 的绝对路径。
* **加载逻辑**：优先从权重同级的 `net_snapshot/` 加载模型代码，若不存在则使用项目 `net/` 目录。
* **评估模式**：`full_res_eval` 设置为 `True` 时进行全图评估；为 `False` 则使用中心 Patch 评估。

### 4. `test_bucketing.py` (加速测试)

针对全分辨率评估的优化版本。通过对不同分辨率图像进行按桶（Bucket）排序，实现 `batch_size > 1` 的多卡并行加速评估。

### 5. `infer_image.py` (推理工具)

用于快速验证单张图像。通过命令行参数指定权重、输入输出路径。内部自动加载 `config.py` 中的模型定义完成图像恢复。

---

## 📊 实验输出说明

每次调用 `train.py` 都会生成唯一的实验目录，例如：`experiment/MoCE_IR_S-2026_01_29_23_33_45/`。该目录结构保证了模型权重、代码快照和指标记录的完整对齐，极大地方便了后续论文对比与结果复现。

---


<!-- ## 项目说明（MoCE-IR 多网络版本）

环境安装：
```bash
git clone -b 20260131_newframework_fixbug https://github.com/QuincyQAQ/moceirv1.git
```

项目结构为 root/data/open_dataset_8_1_1/
代码结构为 root/moceir/moceir/

数据集和代码的具体结构：
├── data
│   └── open_dataset_8_1_1
│       ├── test
│       │   ├── gt
│       │   └── lr
│       ├── train
│       │   ├── gt
│       │   └── lr
│       └── val
│           ├── gt
│           └── lr
── moceir
│   └── moceir
│       ├── config.py
│       ├── data
│       │   ├── dataset_utils.py
│       │   ├── degradation_utils.py
│       │   ├── __init__.py
│       │   └── __pycache__
│       ├── experiment
│       ├── infer_image.py
│       ├── __init__.py
│       ├── install.sh
│       ├── net
│       ├── nohup.out
│       ├── options.py
│       ├── plot_metrics.py
│       ├── __pycache__
│       ├── ReadMe.md
│       ├── requirements.txt
│       ├── test
│       │   └── test.csv
│       ├── test_bucketing.py
│       ├── test_ddp.py
│       ├── test.log
│       ├── test.py
│       ├── test.sh
│       ├── train.py
│       ├── train.sh
│       └── utils

安装环境
conda create -n moceir python=3.10.2
conda activate moceir

pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 torchmetrics==1.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


推荐使用流程：

1. 修改 `config.py`，确定模型、数据路径和训练的超参。  
2. 运行 `train.sh` 开始训练，等待 `experiment/` 下生成对应实验目录，训练完所有的epoch会生成测试数据在`train`文件夹。  
3. 可以自己运行 `test.sh` 做评估（双卡全分辨率 `batch=1`），如果只是python test.py 的话只是单卡效果。  
4. 如需对单张图像做推理，使用 `python infer_image.py --ckpt <ckpt> --input <img> --output <out>`。


- **train.py**  
  训练脚本。从 `config.py` 读取所有训练配置（模型类型、batch size、学习率、数据路径等），并自动：
  - 动态加载 `net/<MODEL>.py` 中的网络（`MODEL` 在 `config.py` 里设置，如 `"MoCE_IR_S"` / `"MoCE_IR"` / `"ACFormer"`）。
  - 使用 Hugging Face `accelerate` 进行单机多卡训练（DDP）。
  - 在 `experiment/<模型名-时间戳>/` 下保存：
    - `checkpoints/`：`last.ckpt` 和最佳指标 ckpt。
    - `net_snapshot/`：本次训练使用的单个网络文件快照（独立可用）。
    - `metrics.csv`：按 epoch 记录的 PSNR/SSIM/LPIPS 等指标。

- **config.py**  
  训练配置文件，只对 `train.py` / `infer_image.py` 有效，用来统一管理：
  - 模型选择：`MODEL`（`"MoCE_IR"` / `"MoCE_IR_S"` / `"ACFormer"`）。
  - 训练与数据相关参数：数据根目录、训练超参、workers、精度、验证频率等（按需修改）。

- **test.py**  
  测试脚本，**完全独立于 `config.py`**，只需在文件末尾的 `argparse.Namespace` 里填好：
  - `ckpt_path`：要测试的 ckpt 绝对路径（支持 Lightning checkpoint）。
  - `benchmarks`：如 `["gopro"]`、`["drmi"]` 等，这里不用动。
  - `patch_size`：patch 测试时的 patch 大小（默认 128 或 256），不用动。
  - `full_res_eval`：
    - `True`：全图像评估。
    - `False`：和原始 MoCE-IR 一样，使用中心 patch 评估。
  测试时会优先从 ckpt 同级的 `../net_snapshot/` 加载网络文件；如不存在，则回退到项目内 `net/<MODEL>.py`。

- **test.sh** 
  测试脚本，写好了所有配置了，直接运行即可。

- **test_bucketing.py**
  测试脚本（Accelerate 多卡可用），用于全分辨率 `batch_size > 1` 的加速评估：
  - 通过 padding collate 解决不同分辨率无法组成 batch 的问题。
  - 通过按分辨率分桶/排序减少 padding 浪费。
  - 但是一般情况下优先用 `test.py`（更接近原始评测方式）；需要更快全分辨率评测时再用这个。

- **infer_image.py**  
  单张图像推理脚本：
  - 通过命令行参数指定 `--ckpt`、`--input`、`--output`、`--device`。
  - 内部使用 `train_options()` 加载 `config.py`，动态加载 `net/<MODEL>.py` 并读取 ckpt 权重，然后对单张图像做恢复。

- **plot_metrics.py**  
  这个后面再完善
  
- **experiment/ 目录**  
  每次调用 `train.py`，都会在 `experiment/` 下自动创建一个子目录，例如：
  - `experiment/MoCE_IR_S-2026_01_29_23_33_45/`
  - 该目录包含：
    - `checkpoints/`：训练过程中的 ckpt（包括最佳 ckpt）。
    - `net_snapshot/`：本次实验使用的网络入口文件拷贝，供独立测试使用。
    - `metrics.csv`：训练/验证/测试的指标记录。


 -->


