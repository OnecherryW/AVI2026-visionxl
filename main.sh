mkdir -p ./output/AVI2026_track2 ./log

nohup python -u main.py \
    --output_model ./output/AVI2026_track2/best_model_2026_track2.pth \
    --test_output_csv ./output/AVI2026_track2/submission.csv \
    --training_time 2026-05-06 \
    --train_csv ./data/metadatacode/train_data_meta.csv \
    --val_csv ./data/metadatacode/val_data_meta.csv \
    --test_csv ./data/metadatacode/test_data_meta.csv \
    --question q1 q2 q3 q4 q5 q6 \
    --label_col g_level \
    --metadata_cols gender age education work_experience H_self_centered E_self_centered A_self_centered C_self_centered self_mean self_std self_max self_min self_range H_minus_E H_minus_A H_minus_C E_minus_A E_minus_C A_minus_C \
    --video_dim 1152 \
    --video_dir /home/gdp/AVI/data/face_embedding/siglip2_all_maxP_face \
    --audio_dim 768 \
    --audio_dir /home/gdp/AVI/data/audioFeatures/audioFeatures/emo2vec/max_pooling/emotion2vec_plus_seed \
    --text_dim 4096 \
    --text_dir /home/gdp/AVI/data/text_feature/SFR-Embedding-Mistral \
    --target_dim 3 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --log_dir ./log \
    > ./output/AVI2026_track2/2026_track2.log 2>&1 &
