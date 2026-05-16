#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# 定义参数
TEST_CSV="test_data.csv"               # 测试集CSV文件路径
Training_Args_JSON_FILE="checkpoints/q3/q3.json"   # 训练参数JSON文件路径以初始化模型


get_json_value() {
  python3 -c "import json; print(json.load(open('$Training_Args_JSON_FILE'))['$1'])"
}

# 读取JSON文件中的参数
QUESTION=$(get_json_value "question")    # [q3,q4,q5,q6]
LABEL_COL=$(get_json_value "label_col")  # [Honesty-Humility, Extraversion, Agreeableness, Conscientiousness]
AUDIO_DIR=$(get_json_value "audio_dir")  # 音频特征目录
VIDEO_DIR=$(get_json_value "video_dir")  # 视频特征目录
TEXT_DIR=$(get_json_value "text_dir")  # 文本特征目录
AUDIO_DIM=$(get_json_value "audio_dim")  # 音频特征维度
VIDEO_DIM=$(get_json_value "video_dim")  # 视频特征维度
TEXT_DIM=$(get_json_value "text_dim")  # 文本特征维度
MODEL_PATH=$(get_json_value "output_model")  # 模型路径
dir_path="$(dirname "$Training_Args_JSON_FILE")/"
OUTPUT_CSV="${dir_path}_${QUESTION}.csv"  # 输出CSV文件路径
BATCH_SIZE=$(get_json_value "batch_size")  # 批量大小
NUM_WORKERS=$(get_json_value "num_workers")  # 工作线程数
HCP_dropout_audio=$(get_json_value "HCPdropout_audio")  # 音频模态的dropout
HCP_dropout_video=$(get_json_value "HCPdropout_video")  # 视频模态的dropout
HCP_dropout_text=$(get_json_value "HCPdropout_text")  # 文本模态的dropout
HCP_dropout_pure_text=$(get_json_value "HCPdropout_pure_text")  # 纯文本模态的dropout
USE_PROMPT=$(get_json_value "use_prompt")  # 是否使用prompt
UNIFIED_DIM=$(get_json_value "unified_dim")  # 统一模态维度
HEADS_NUM=$(get_json_value "heads_num")  # 头数
ATC_dropout=$(get_json_value "ATCdropout")  # ATC的dropout
VTC_dropout=$(get_json_value "VTCdropout")  # VTC的dropout
HIDDEN_DIM=$(get_json_value "hidden_dim")  # 隐藏层维度
ENHANCER_DIM=$(get_json_value "enhancer_dim")  # 增强器维度
TFEdropout=$(get_json_value "TFEdropout")  # TFE的dropout
RHdropout=$(get_json_value "RHdropout")  # RH的dropout
Target_dim=$(get_json_value "target_dim")  # 目标维度
NUM_MODALITIES=$(get_json_value "num_modalities")  # 模态数量
MODALITIES=$(get_json_value "modalities")  # 模态列表（逗号分隔）
MODALITIES=$(echo "$MODALITIES" | sed -e "s/\[//" -e "s/\]//" -e "s/'//g" -e "s/, */,/g")
echo "模态为: $MODALITIES"



# 执行训练脚本
python3 test_task1.py \
  --test_csv "$TEST_CSV" \
  --question "$QUESTION" \
  --label_col "$LABEL_COL" \
  --audio_dir "$AUDIO_DIR" \
  --video_dir "$VIDEO_DIR" \
  --text_dir "$TEXT_DIR" \
  --audio_dim "$AUDIO_DIM" \
  --video_dim "$VIDEO_DIM" \
  --text_dim "$TEXT_DIM" \
  --model_path "$MODEL_PATH" \
  --output_csv "$OUTPUT_CSV" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --HCPdropout_audio "$HCP_dropout_audio" \
  --HCPdropout_video "$HCP_dropout_video" \
  --HCPdropout_text "$HCP_dropout_text" \
  --HCPdropout_pure_text "$HCP_dropout_pure_text" \
  --use_prompt "$USE_PROMPT" \
  --unified_dim "$UNIFIED_DIM" \
  --heads_num "$HEADS_NUM" \
  --ATCdropout "$ATC_dropout" \
  --VTCdropout "$VTC_dropout" \
  --hidden_dim "$HIDDEN_DIM" \
  --enhancer_dim "$ENHANCER_DIM" \
  --TFEdropout "$TFEdropout" \
  --RHdropout "$RHdropout" \
  --target_dim "$Target_dim" \
  --num_modalities "$NUM_MODALITIES" \
  --modalities "$MODALITIES" \

# 检查脚本执行状态
if [ $? -eq 0 ]; then
  echo "测试任务成功完成!"
  echo "csv结果已保存到: $OUTPUT_MODEL"
else
  echo "测试任务失败，请检查错误信息!"
fi
