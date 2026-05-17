# AVI2026-VisionXL

本仓库包含 AVI Challenge 2026 的两个任务实现：

- `AVI_track1`：Self-Report Personality Regression（q3/q4/q5/q6）
- `AVI_track2`：Personality Assessment（多类别人格评估）

## 仓库结构

```text
AVI2026-visionxl/
├── AVI_track1/
│   ├── train_track1.sh
│   ├── test_task1.sh
│   └── README.md
├── AVI_track2/
│   ├── main.sh
│   ├── test.sh
│   └── README.md
└── requirements.txt
```

## 环境准备

建议使用 Python 3.10+ 与 conda：

```bash
conda create -n avi2026 python=3.10 -y
conda activate avi2026
```

可选安装方式：

- 安装仓库根目录依赖（通用）：

```bash
pip install -r requirements.txt
```

- 或进入对应赛道目录安装该赛道依赖：

```bash
# Track1
cd AVI_track1
pip install -r requirement.txt

# Track2
cd ../AVI_track2
pip install -r requirements.txt
```

## Track1 快速开始

```bash
cd AVI_track1

# 训练
bash train_track1.sh

# 测试（默认读取 checkpoints/q3/q3.json）
bash test_task1.sh
```

说明：

- `test_task1.sh` 会从 `Training_Args_JSON_FILE` 指向的 json 中读取模型与特征参数。
- 若需测试其他维度（q4/q5/q6），修改脚本中的 json 路径。
- `trans.py` 可将 q3/q4/q5/q6 结果按 id 合并为最终 `submission.csv`。

## Track2 快速开始

```bash
cd AVI_track2

# 训练
bash main.sh

# 测试
bash test.sh
```

说明：

- 训练与测试均由 `main.py` 驱动。
- 主要输入为 `features/audio|video|text` 与 `data/metadatacode/*.csv`。
- 默认输出目录为 `output/AVI2026_track2/`。

## 详细说明

每个赛道的完整方法说明、参数解释与结果说明见：

- `AVI_track1/README.md`
- `AVI_track2/README.md`

## 声明

本仓库代码仅用于学术研究，请遵守 AVI Challenge 官方规则与数据使用协议
