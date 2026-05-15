#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# ========== 定义参数 ==========
# for dataset
TRAIN_CSV="./train_data.csv"
VAL_CSV="./val_data.csv"
TEST_CSV="./val_data.csv"
QUESTION="q6"  # 可选: q3, q4, q5, q6
LABEL_COL="C_self"  # H_self，E_self，A_self，C_self
AUDIO_DIR="AVI_features/text/SFR-Embedding-Mistral-2048/"
VIDEO_DIR="AVI_features/text/SFR-Embedding-Mistral-2048/"
TEXT_DIR="AVI_features/text/SFR-Embedding-Mistral-2048/"
AUDIO_DIM=4096
VIDEO_DIM=4096
TEXT_DIM=4096

# for training
BATCH_SIZE=4
LEARNING_RATE=0.0001
NUM_EPOCHS=60
NUM_WORKERS=4
PIN_MEMORY="True"
OPTIM="adamw" # 可选: adamw, sgd, adam

## for model
# for projector
HCPdropout_audio=0
HCPdropout_video=0
HCPdropout_text=0
HCPdropout_pure_text=0
USE_PROMPT="True"
UNIFIED_DIM=1024

# for AT_VT connector
HEADS_NUM=8
ATCdropout=0
VTCdropout=0
HIDDEN_DIM=512

# for text feature enhancer
ENHANCER_DIM=512
TFEdropout=0

# for regression head
RHdropout=0
Target_dim=1
NUM_MODALITIES=3
MODALITIES="audio,text,video"
AUDIO_NAME=$(basename "$AUDIO_DIR")
VIDEO_NAME=$(basename "$VIDEO_DIR")
TEXT_NAME=$(basename "$TEXT_DIR")
RUN_TIME=$(date +"%Y%m%d_%H%M%S")
echo "$RUN_TIME"
COMBINED_NAME="${AUDIO_NAME}_${VIDEO_NAME}_${TEXT_NAME}"
OUTPUT_DIR="save_ckpt/track1/${COMBINED_NAME}/${QUESTION}/"
OUTPUT_MODEL="${OUTPUT_DIR}/best_model.pth"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ========== 执行训练脚本 ==========
python3 train_task1.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --question "$QUESTION" \
  --label_col "$LABEL_COL" \
  --audio_dir "$AUDIO_DIR" \
  --video_dir "$VIDEO_DIR" \
  --text_dir "$TEXT_DIR" \
  --audio_dim "$AUDIO_DIM" \
  --video_dim "$VIDEO_DIM" \
  --text_dim "$TEXT_DIM" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs "$NUM_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --pin_memory "$PIN_MEMORY" \
  --optim "$OPTIM" \
  --HCPdropout_audio "$HCPdropout_audio" \
  --HCPdropout_video "$HCPdropout_video" \
  --HCPdropout_text "$HCPdropout_text" \
  --HCPdropout_pure_text "$HCPdropout_pure_text" \
  --use_prompt "$USE_PROMPT" \
  --unified_dim "$UNIFIED_DIM" \
  --heads_num "$HEADS_NUM" \
  --ATCdropout "$ATCdropout" \
  --VTCdropout "$VTCdropout" \
  --hidden_dim "$HIDDEN_DIM" \
  --enhancer_dim "$ENHANCER_DIM" \
  --TFEdropout "$TFEdropout" \
  --RHdropout "$RHdropout" \
  --target_dim "$Target_dim" \
  --num_modalities "$NUM_MODALITIES" \
  --modalities "$MODALITIES" \
  --output_model "$OUTPUT_MODEL" \
  --log_dir "$OUTPUT_DIR" \
  --training_time "$RUN_TIME" \


# ========== 结果提示 ==========
if [ $? -eq 0 ]; then
  echo "训练任务成功完成!"
  echo "模型已保存到: $OUTPUT_MODEL"
else
  echo "训练任务失败，请检查错误信息!"
fi