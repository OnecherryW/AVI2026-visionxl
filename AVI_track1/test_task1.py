import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset.baseline_dataset import MultimodalDatasetForTestT1
from dataset.baseline_dataset import collate_fn_test
from model.new_model.builder import FusionModel
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_save(model, test_loader, output_csv, col):
    model.eval()
    predictions = []
    sample_ids = []

    with torch.no_grad():
        for features, mask, ids in tqdm(test_loader, desc="Predicting", ncols=100):
            print(features.keys())
            features = {k: v.to(device) for k, v in features.items()}
            audio_feat = features['audio']
            video_feat = features['video']
            text_feat = features['text']
            outputs = model(audio_feat, video_feat, text_feat)

            preds = (outputs.squeeze().cpu().numpy() * 4 + 1)
            predictions.extend(preds.tolist() if preds.ndim > 0 else [preds])
            sample_ids.extend(ids)

    results = pd.DataFrame({'id': sample_ids, col: predictions})
    results.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    #### for dataset
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--label_col', type=str, required=True)

    #### for input_features
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--text_dir', type=str, required=True)
    parser.add_argument('--audio_dim', type=int, default=384)
    parser.add_argument('--video_dim', type=int, default=512)
    parser.add_argument('--text_dim', type=int, default=768)

    #### for testing
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    #### for model
    # for projector
    parser.add_argument('--HCPdropout_audio', type=float, default=0.2)
    parser.add_argument('--HCPdropout_video', type=float, default=0.2)
    parser.add_argument('--HCPdropout_text', type=float, default=0.2)
    parser.add_argument('--HCPdropout_pure_text', type=float, default=0.1)
    parser.add_argument('--use_prompt', type=bool, default=False) # 可学习的prompt
    parser.add_argument('--unified_dim', type=int, default=512)   # projector对齐各个模态后的维度
    # for AT_VT connector
    parser.add_argument('--heads_num', type=int, default=4)
    parser.add_argument('--ATCdropout', type=float, default=0.3)  # AT跨模态交互模块的dropout
    parser.add_argument('--VTCdropout', type=float, default=0.3)  # VT跨模态交互模块的dropout
    parser.add_argument('--hidden_dim', type=int, default=256)    # 三模态进入text增强器前的维度，也是text增强器的输入维度
    # for text feature enhancer
    parser.add_argument('--enhancer_dim', type=int, default=512)  # text增强器的输出维度
    parser.add_argument('--TFEdropout', type=float, default=0.2)  # text增强器的dropout
    # for regression head
    parser.add_argument('--RHdropout', type=float, default=0.2)   # 回归头的dropout
    parser.add_argument('--target_dim', type=int, default=1)      # 最终回归的维度
    parser.add_argument('--num_modalities', type=int, default=3)
    parser.add_argument('--modalities', type=str, default="audio,video,text")

    args = parser.parse_args()
    args.modalities = [m.strip() for m in args.modalities.split(',')]

    test_dataset = MultimodalDatasetForTestT1(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_test, num_workers=args.num_workers, pin_memory=True)

    model = FusionModel(args).to(device)
    model = load_model(model, args.model_path, device)

    predict_and_save(model, test_loader, args.output_csv, args.label_col)

if __name__ == '__main__':
    main()
