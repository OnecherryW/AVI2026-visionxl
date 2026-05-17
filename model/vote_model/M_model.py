import torch
import torch.nn.functional as F


class GPT2Shared(torch.nn.Module):
    def __init__(self, args):
        super(GPT2Shared, self).__init__()
        self.args = args
        self.base_dim = 768

        self.feature_projection = torch.nn.Linear(self.base_dim * 3, self.base_dim)  # 图像特征转换为GPT-2的嵌入维度

        self.video_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.video_dim, self.base_dim * 3),
            torch.nn.GELU()
        )

        self.audio_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.audio_dim, self.base_dim * 3),
            torch.nn.GELU()
        )

        self.text_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.text_dim, self.base_dim * 3),
            torch.nn.GELU()
        )

        # self.metadata_dim = getattr(args, "metadata_dim", 15)
        # self.metadata_hidden_dim = 64 if self.metadata_dim > 0 else 0
        # if self.metadata_dim > 0:
        #     self.metadata_adapter = torch.nn.Sequential(
        #         torch.nn.Linear(self.metadata_dim, self.metadata_hidden_dim),
        #         torch.nn.LayerNorm(self.metadata_hidden_dim),
        #         torch.nn.GELU()
        #     )
        # else:
        #     self.metadata_adapter = None

        self.metadata_dim = getattr(args, "metadata_dim", 0)
        self.metadata_hidden_dim = 64 if self.metadata_dim > 0 else 0

        if self.metadata_dim > 0:
            self.metadata_adapter = torch.nn.Sequential(
                torch.nn.Linear(self.metadata_dim, self.metadata_hidden_dim),
                torch.nn.LayerNorm(self.metadata_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(self.metadata_hidden_dim, self.metadata_hidden_dim),
                torch.nn.LayerNorm(self.metadata_hidden_dim),
                torch.nn.GELU()
            )
            self.video_meta_bias = torch.nn.Linear(self.metadata_hidden_dim, self.base_dim)
            self.audio_meta_bias = torch.nn.Linear(self.metadata_hidden_dim, self.base_dim)
            self.text_meta_bias = torch.nn.Linear(self.metadata_hidden_dim, self.base_dim)
        else:
            self.metadata_adapter = None
            self.video_meta_bias = None
            self.audio_meta_bias = None
            self.text_meta_bias = None

        classifier_input_dim = self.base_dim * 3
        
        self.ensemble = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(classifier_input_dim, self.base_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.base_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, args.target_dim)
            ) for _ in range(32)
        ])

    def soft_clamp(self, x, eps=1e-6):
        sp_pos = F.softplus(x);
        sp_neg = F.softplus(-x);
        return (sp_pos + eps) / (sp_pos + sp_neg + eps*2)

    def forward(self, audio_feat, video_feat, text_feat, metadata_feat=None, attention_mask=None):
        # Get batch size, sequence length and feature dimension
        B, T, _ = video_feat.shape
        video_feat = video_feat.reshape(B * T, -1)
        text_feat = text_feat.reshape(B * T, -1)
        audio_feat = audio_feat.reshape(B * T, -1)
        # Project features through adapters and feature projection
        video_feat = self.feature_projection(self.video_adapter(video_feat))  # [B*T, D]
        audio_feat = self.feature_projection(self.audio_adapter(audio_feat))  # [B*T, D]
        text_feat = self.feature_projection(self.text_adapter(text_feat))  # [B*T, D]
        
        if self.metadata_adapter is not None:
            if metadata_feat is None:
                metadata_feat = torch.zeros(B, self.metadata_dim, device=video_feat.device, dtype=video_feat.dtype)
            else:
                metadata_feat = metadata_feat.to(device=video_feat.device, dtype=video_feat.dtype)
            metadata_feat = self.metadata_adapter(metadata_feat)  # [B, H]
            video_bias = self.video_meta_bias(metadata_feat).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            audio_bias = self.audio_meta_bias(metadata_feat).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            text_bias = self.text_meta_bias(metadata_feat).unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            video_feat = video_feat + video_bias
            audio_feat = audio_feat + audio_bias
            text_feat = text_feat + text_bias

        multi_modal_chunk = torch.cat([video_feat, text_feat, audio_feat], dim=-1)  # [B*T, D*3]
        outputs = torch.stack([mlp(multi_modal_chunk) for mlp in self.ensemble], dim=0) # [32, B*T, target_dim]
        logits = outputs.mean(dim=0)  # [B*T, target_dim]
        logits = logits.view(B, T, -1)
        if attention_mask is not None:
            mask = attention_mask.to(device=logits.device, dtype=logits.dtype).unsqueeze(-1)
            logits = (logits * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            logits = logits.mean(dim=1)
        return logits
