# REVE：面向 EEG 的基础模型

<p align="center">
  <a href="README.en.md">English</a> · <a href="README.md">简体中文</a>
</p>

> REVE 是一个面向脑电（EEG）的基础模型项目，目标是通过大规模预训练，让模型适配不同设备、不同电极布局和不同下游脑机接口任务。

[![论文](https://img.shields.io/badge/arXiv-2510.21585-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.21585)
[![项目主页](https://img.shields.io/badge/Website-brain--bzh.github.io-0A66C2?logo=github&logoColor=white)](https://brain-bzh.github.io/reve/)
[![Colab 教程](https://img.shields.io/badge/Google%20Colab-Tutorial-orange?logo=google-colab&logoColor=orange)](https://colab.research.google.com/drive/13wMytamrucmU9c2OIc-TcoEIj6xSqPGo?usp=sharing)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-REVE%20Models-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/collections/brain-bzh/reve)

## 项目简介

REVE 是一个 EEG 基础模型代码库，核心思路是先利用大规模 EEG 数据进行自监督预训练，再将学到的通用脑电表征迁移到具体任务中。

仓库主要包含：

- `src/train.py`：使用 MAE 风格方法进行基础模型预训练；
- `src/dt.py`：下游任务的线性探测（linear probing）和全量微调（fine-tuning）；
- `src/dt_opt.py`：使用 Optuna 搜索下游任务超参数；
- `src/eval_dt.py`：评估保存的下游模型；
- `preprocessing/`：不同 EEG 数据集的转换和预处理脚本；
- `hf/`：Hugging Face 模型导出和封装工具；
- `.scripts/`：训练和下游实验的辅助启动脚本。

## 这个项目解决什么问题？

不同 EEG 设备可能有不同的电极数量、排列方式、采样设置和数据分布。传统模型往往需要针对每个数据集重新训练，导致迁移成本较高。

REVE 尝试学习更通用的 EEG 表征，使同一个预训练编码器能够迁移到不同设备和不同任务，例如脑机接口分类、睡眠分析或其他神经信号任务。

这不是一个“直接读取 EEG 就得到脑机接口结果”的黑盒工具，而是一个研究基础模型预训练、迁移学习和跨设备泛化的实验平台。

## 仓库结构

- `src/`：训练、下游任务、模型和工具函数；
- `src/configs/`：Hydra 配置文件；
- `src/downstream_tasks/`：下游任务定义；
- `preprocessing/`：数据集专用预处理脚本；
- `hf/`：Hugging Face 导出说明和封装；
- `.scripts/`：批量运行脚本。

详细说明：

- [配置指南](src/configs/README.md)
- [下游任务指南](src/downstream_tasks/README.md)
- [预处理指南](preprocessing/README.md)
- [Hugging Face 导出说明](hf/README.md)

## 环境安装

Python 版本要求来自 `pyproject.toml`：

```text
Python >= 3.11
```

使用 `uv`：

```bash
uv sync
```

或者使用 `pip`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

之后可以使用 `python`、`accelerate` 或 `torchrun` 运行实验。

## 配置方式

训练和评估脚本使用 Hydra 配置：

- 预训练主配置：`src/configs/config_train.yaml`；
- 下游任务主配置：`src/configs/config_dt.yaml`。

运行前建议先阅读：[完整配置说明](src/configs/README.md)。

## 运行预训练

主入口：

```bash
accelerate launch src/train.py --config-name config_train.yaml
```

常见参数覆盖：

```bash
accelerate launch src/train.py --config-name config_train.yaml \
  data.path=/path/to/preprocessed_foundation_data \
  checkpointing.path=/path/to/checkpoints \
  trainer.batch_size=256 \
  trainer.lr=3e-4
```

切换编码器规模：

```bash
accelerate launch src/train.py --config-name config_train.yaml encoder=small
```

也可以使用辅助脚本：

```bash
bash .scripts/train.sh
```

## 运行下游任务

线性探测（LP）：

```bash
torchrun --nproc_per_node=gpu src/dt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  pretrained_path=hf:brain-bzh/reve-base \
  training_mode=lp
```

连续执行线性探测和微调：

```bash
torchrun --nproc_per_node=gpu src/dt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  pretrained_path=/path/to/encoder.pth \
  training_mode=lp+ft
```

修改阶段超参数：

```bash
torchrun --nproc_per_node=gpu src/dt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  training_mode=lp \
  task.linear_probing.n_epochs=50 \
  task.linear_probing.optimizer.lr=1e-3
```

## Optuna 超参数搜索

```bash
torchrun --nproc_per_node=gpu src/dt_opt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  pretrained_path=hf:brain-bzh/reve-base \
  n_runs=50
```

## 评估保存的模型

```bash
python src/eval_dt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  checkpoint_path=/path/to/model_best.pth
```

## 数据预处理

预处理脚本位于 `preprocessing/`，并且与数据集相关。例如：

```bash
python preprocessing/preprocessing_bciciv2a.py \
  --raw /path/raw \
  --processed /path/processed

python preprocessing/preprocessing_physio.py \
  --root /path/raw \
  --processed /path/processed \
  --type reve

python preprocessing/ISRUC/prepare_ISRUC.py \
  --raw /path/raw \
  --processed /path/out/{}
```

完整说明见：[预处理指南](preprocessing/README.md)。

## 数据资源

部分允许再分发的数据已经发布在 Hugging Face：

[brain-bzh/reve-dataset](https://huggingface.co/datasets/brain-bzh/reve-dataset)

预训练模型集合见：[Hugging Face REVE Models](https://huggingface.co/collections/brain-bzh/reve)。

## 使用注意

- Hydra 会自动设置运行目录，具体参考 `hydra.run.dir`；
- 如果配置使用 `${env:SCRATCH}` 等环境变量，需要提前设置；
- 下游任务和数据集元信息位于 `src/configs/task/*.yaml`；
- 不同 EEG 设备、受试者和数据集之间的泛化能力需要独立验证，不能只根据单一数据集结果判断；
- 真实 BCI 应用还需要考虑数据质量、实时延迟、个体校准、安全性和伦理要求。

## 贡献者

感谢所有参与 REVE 项目的贡献者：

- Timotée Ly-Manson
- Alix Lamouroux

## 引用

如果使用本仓库或 REVE 预训练模型，请引用：

```bibtex
@article{elouahidi2025reve,
    title   = {{REVE}: A Foundation Model for {EEG}: Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects},
    author  = {El Ouahidi, Yassine and Lys, Jonathan and Thölke, Philipp and Farrugia, Nicolas and Pasdeloup, Bastien and Gripon, Vincent and Jerbi, Karim and Lioi, Giulia},
    journal = {Advances in Neural Information Processing Systems},
    year    = {2025},
    url     = {https://brain-bzh.github.io/reve/}
}
```

机器可读引用信息见 `CITATION.cff`。
