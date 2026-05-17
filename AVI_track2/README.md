# 🏆 AVI Challenge 2026 - Track2: Personality Assessment

## 项目简介

本项目旨在通过多模态特征融合深度学习方法实现面试表现的人格评估，对音频、视频和文本三类特征进行有效建模与集成，预测候选人的以下人格维度评分：
- **诚信（Honesty-Humility）**  
- **外向性（Extraversion）**  
- **宜人性（Agreeableness）**  
- **责任心（Conscientiousness）**

我们在 AVI 2026 Track2 任务中基于音视频+文本多模态融合方法取得了优异表现，以下为代码结构与运行说明。

---

## 1️⃣ 方法流程简介

1. **特征提取**：提前利用预训练模型提取音频、视频、文本三类特征，分别存放于 features 目录下（audio, video, text）。
2. **特征融合**：模型设计实现多模态特征的拼接或加权融合，提升对多样性人格特质信息的捕捉能力。
3. **模型训练**：
   - 加载训练、验证集的多模态特征及标签（`g_level` 等）。
   - 使用定制的 loss（如`MSELoss`）进行优化。
   - 自动保存表现最优模型至 `output/AVI2026_track2`。
4. **模型测试与评估**：
   - 仅需文本特征（可多模态对比）。
   - 输出测试集预测分数至 `output/AVI2026_track2/submission.csv`。
   - 融合主/副指标，支持平衡多类别评估。

---

## 2️⃣ 目录结构

```
AVI_track2/
├── main.py                   # 训练/测试入口
├── test.sh                   # 测试脚本
├── model/
│   ├── vote_model/
│   │   └── M_model.py        # 多模态集成模型
│   └── baseline_model.py     # 单模态/基线模型
├── dataset/
│   └── baseline_dataset2_vote.py # 数据处理
├── features/
│   ├── audio/                # 音频特征
│   ├── video/                # 视频特征
│   └── text/                 # 文本特征
├── output/
│   └── AVI2026_track2/
│       ├── best_model_2026_track2.pth
│       ├── submission.csv
├── data/
│   └── metadatacode/
│       ├── train_data_meta.csv
│       ├── val_data_meta.csv
│       └── test_data_meta.csv
├── log/
├── requirements.txt
└── README.md
```

---

## 3️⃣ 环境与依赖

- Python ≥ 3.10（推荐使用 conda 环境）
- 主要依赖参见 `requirements.txt`

### 安装步骤参考

```bash
git clone https://github.com/OnecherryW/AVI2026-visionxl.git
cd AVI2026-visionxl/AVI_track2

# Step 1: 创建新的 conda 环境
conda create -n avi2026 python=3.10 -y
conda activate avi2026

# Step 2: 安装依赖
pip install -r requirements.txt
```

---

## 4️⃣ 数据准备

请从官方或组织者指定渠道下载音频、视频、文本特征文件，并放置于对应目录。

```bash
# 目录示例
features/
  ├── audio/
  ├── video/
  └── text/
data/metadatacode/
  ├── train_data_meta.csv
  ├── val_data_meta.csv
  └── test_data_meta.csv
```

---

## 5️⃣ 训练与测试流程

### 训练
可直接使用一键脚本：
```bash
bash train.sh
```

### 测试

可直接使用一键脚本：

```bash
bash test.sh
```

## 6️⃣ 结果说明

- **最佳模型**：`output/AVI2026_track2/best_model_2026_track2.pth`
- **最终测试提交**：`output/AVI2026_track2/submission.csv`
- **损失函数**：均方误差（MSE）
- 日志与训练过程保存在 `log/`

---

## 7️⃣ 参考/致谢

- 多模态特征处理、集成算法
- 感谢 AVI 2026 官方与 [MERtools](https://github.com/zeroQiaoba/MERTools) 对特征提取作出的支持
- 强烈建议对原始特征、比赛数据严格遵守官方协议及学术规范

---

## 8️⃣ 联系方式

如有问题欢迎联系 HFUT-VisionXL 团队维护者，或在 Issues 区留言。

---

**声明**：本项目代码仅限学术研究用途，数据及代码请勿用于商业目的，并请遵守大赛及数据协议。

- 🏆 Thanks to the AVI Challenge 2025 organizers
- 🤗 Thanks to the developers of [MERtools](https://github.com/zeroQiaoba/MERTools) for their excellent open-source tools that supported our data preprocessing.
