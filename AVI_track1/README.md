# AVI Track1

本目录用于 AVI Track1 的训练与测试，包含数据读取、模型定义、已保存权重与推理脚本。

## 目录结构

```text
AVI_track1/
├── dataset/
│   ├── baseline_dataset.py
│   └── baseline_dataset2.py
├── model/
│   ├── baseline_model.py
│   ├── fusion_attention.py
│   └── new_model/
├── checkpoints/
│   ├── q3/
│   │   ├── best_model.pth
│   │   └── q3.json
│   ├── q4/
│   │   ├── best_model.pth
│   │   └── q4.json
│   ├── q5/
│   │   ├── best_model.pth
│   │   └── q5.json
│   └── q6/
│       ├── best_model.pth
│       └── q6.json
├── utils/
│   └── npy_check.py
├── train_task1.py
├── test_task1.py
├── train_track1.sh
├── test_task1.sh
├── train_data.csv
├── val_data.csv
├── test_data.csv
├── all_data.csv
├── trans.py
├── requirement.txt
└── README.md
```

## 环境准备

```bash
conda activate <your_env>
pip install -r requirement.txt
```

## 数据与特征准备

1. 准备 `train_data.csv / val_data.csv / test_data.csv`。
2. 确认音频、视频、文本特征目录路径（在脚本参数中配置）。
3. 使用 `utils/npy_check.py` 检查特征 shape，确保与 `*_DIM` 参数匹配。

## 训练

默认训练脚本：

```bash
bash train_track1.sh
```

可在 `train_track1.sh` 中按需调整：
- `QUESTION`（q3/q4/q5/q6）
- `LABEL_COL`
- `AUDIO_DIR / VIDEO_DIR / TEXT_DIR`
- 训练超参数与模型结构参数

训练输出默认写入：

```text
save_ckpt/track1/<audio_name>_<video_name>_<text_name>/<question>/
```

## 测试 / 推理

```bash
bash test_task1.sh
```

`test_task1.sh` 会读取 `Training_Args_JSON_FILE`（如 `checkpoints/q3/q3.json`） 中的参数来恢复模型并生成结果。  
如需切换维度任务，请修改 `Training_Args_JSON_FILE` 指向对应的 `q4/q5/q6` JSON 文件。
