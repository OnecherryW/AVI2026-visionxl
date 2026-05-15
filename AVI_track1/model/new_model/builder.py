import torch.nn as nn
import torch.nn.functional as F
from .ATconnector import ATconnector
from .VTconnector import VTconnector
from .TextFeatureEnhancer import TextFeatureEnhancer
from .Token_Refinement_Module import HybridChunkProjector

'''
我们自己的basecode
'''

'''
args:
    audio_dim: 音频特征维度
    video_dim: 视频特征维度
    text_dim: 文本特征维度
    unified_dim: 统一特征维度
    hidden_dim: 隐藏层维度
    enhancer_dim: 特征增强维度
    target_dim: 目标输出维度1
'''


class FusionModel(nn.Module):
    def __init__(self, args):
        super(FusionModel, self).__init__()
        self.args = args
        
        # 维度对齐模块
        self.audio_projector = HybridChunkProjector(
            input_dim=args.audio_dim,
            samples=1,
            output_dim=args.unified_dim,
            dropout=args.HCPdropout_audio,
            use_prompt=args.use_prompt
        )
        
        self.video_projector = HybridChunkProjector(
            input_dim=args.video_dim,
            samples=1,
            output_dim=args.unified_dim,
            dropout=args.HCPdropout_video,
            use_prompt=args.use_prompt
        )
        
        self.text_projector1 = HybridChunkProjector(
            input_dim=args.text_dim,
            samples=1,
            output_dim=args.unified_dim,
            dropout=args.HCPdropout_text,
            use_prompt=args.use_prompt
        )

        self.text_projector2 = HybridChunkProjector(
            input_dim=args.text_dim,
            samples=1,
            output_dim=args.hidden_dim,
            dropout=args.HCPdropout_pure_text,
            use_prompt=args.use_prompt
        )

        # 跨模态交互模块
        self.at_connector = ATconnector(
            n_heads= args.heads_num,
            dim_audio=args.unified_dim,
            dim_text=args.unified_dim,
            output_dim=args.hidden_dim,
            dropout = args.ATCdropout
        )
        
        self.vt_connector = VTconnector(
            n_heads= args.heads_num,
            dim_video=args.unified_dim,
            dim_text=args.unified_dim,
            output_dim=args.hidden_dim,
            dropout = args.VTCdropout
        )

        # 特征增强模块
        self.text_enhancer = TextFeatureEnhancer(
            feat_dim=args.hidden_dim,
            out_dim=args.enhancer_dim,
            hidden_dim=args.enhancer_dim,
            dropout= args.TFEdropout
        )

        # 回归预测头
        self.regression_head = nn.Sequential(
            nn.Linear(args.enhancer_dim, args.enhancer_dim//2),
            nn.ReLU(),
            nn.Dropout(args.RHdropout),
            nn.Linear(args.enhancer_dim//2, args.target_dim)
        )

    def forward(self, audio_feat, video_feat, text_feat):

        # Step 1: 维度对齐
        audio_proj = self.audio_projector(audio_feat)
        video_proj = self.video_projector(video_feat) 
        text_proj1 = self.text_projector1(text_feat)

        text_proj2 = self.text_projector2(text_feat)

        # Step 2: 跨模态交互
        at_fusion = self.at_connector(audio_proj, text_proj1)
        vt_fusion = self.vt_connector(video_proj, text_proj1)

        # Step 3: 特征增强
        enhanced_text = self.text_enhancer(
            t_feat = text_proj2,
            at_feat=at_fusion,
            vt_feat=vt_fusion
        )

        # Step 4: 回归预测
        prediction = self.regression_head(enhanced_text)
        
        return prediction

    def compute_loss(self, pred, target):
        return nn.MSELoss()(pred, target)