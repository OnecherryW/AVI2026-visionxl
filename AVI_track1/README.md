# 🏆 AVI Challenge 2026 - Track1: Self-Report Personality Regression

## 项目简介

本项目面向 AVI 2026 Track1 任务，基于音频、视频、文本三模态特征进行人格维度回归预测。  
当前实现以多模态融合模型为核心，训练后可输出指定维度（q3/q4/q5/q6）的测试集预测结果。

---

## 1️⃣ 方法流程简介

1. **特征准备**：使用预提取好的音频/视频/文本特征（`.npy`），并保证与元数据中的样本 id 对齐。  
2. **多模态融合建模**：通过 `FusionModel` 进行三模态编码与融合，输出回归分数。  
3. **模型训练**：
   - 加载 `train_data.csv` 与 `val_data.csv`。
   - 以 `MSELoss` 进行优化并在验证集上选择最优模型。
   - 自动保存模型与参数配置（json）。
4. **模型测试与导出**：
   - 根据训练参数 json 进行测试推理。
   - 输出对应人格维度的预测 CSV 文件。

---

## 2️⃣ 目录结构

```
AVI_track1/
├── README.md
├── requirement.txt
├── train_task1.py                # 训练入口
├── test_task1.py                 # 测试入口
├── train_track1.sh               # 训练脚本
├── test_task1.sh                 # 测试脚本（读取 checkpoints 下 json 参数）
├── train_data.csv
├── val_data.csv
├── test_data.csv
├── all_data.csv
├── dataset/
│   ├── baseline_dataset.py       # Track1 数据集与 collate
│   └── baseline_dataset2.py
├── model/
│   ├── baseline_model.py
│   ├── fusion_attention.py
│   └── new_model/                # 当前 Track1 融合模型实现
├── checkpoints/
│   ├── q3/
│   ├── q4/
│   ├── q5/
│   └── q6/                       # 模型与超参数 json
├── test_feature/
└── utils/
    └── npy_check.py              # 特征 shape 检查
```

---

## 3️⃣ 环境与依赖

- Python ≥ 3.10（推荐使用 conda 环境）
- 主要依赖见 `requirement.txt`

### 安装步骤参考

```bash
git clone https://github.com/OnecherryW/AVI2026-visionxl.git
cd AVI2026-visionxl/AVI_track1

# Step 1: 创建环境
conda create -n avi2026 python=3.10 -y
conda activate avi2026

# Step 2: 安装依赖
pip install -r requirement.txt
```

---

## 4️⃣ 数据与特征准备

1. 准备元数据文件：`train_data.csv`、`val_data.csv`、`test_data.csv`。  
2. 准备三模态特征目录，并在脚本或 json 中配置：
   - `audio_dir`
   - `video_dir`
   - `text_dir`
3. 建议先用 `utils/npy_check.py` 检查特征维度，确保与 `audio_dim/video_dim/text_dim` 一致。

---

## 5️⃣ 训练与测试流程

### 训练

可直接运行：

```bash
bash train_track1.sh
```

脚本会自动设置超参数、启动训练，并保存最优模型到输出目录。

### 测试

可直接运行：

```bash
bash test_task1.sh
```

默认会读取 `checkpoints/q3/q3.json`。若需测试其他维度，请修改脚本中的 `Training_Args_JSON_FILE` 路径（如 q4/q5/q6）。

---

## 6️⃣ 输出结果说明

- **模型文件**：由 `train_track1.sh` 中 `output_model` 指定（默认位于 `save_ckpt/track1/.../best_model.pth`）
- **测试结果**：由 `test_task1.sh` 根据 json 自动生成，输出为对应维度的 CSV

---

## 7️⃣ 参考/致谢

- 感谢 AVI Challenge 官方提供任务与数据协议
- 感谢开源社区在多模态特征建模方面的基础工作支持

---

## 8️⃣ 联系方式

如有问题，欢迎在仓库 Issues 区交流。

---

**声明**：本项目仅用于学术研究，请遵守比赛规则与数据使用协议，勿用于商业用途。
