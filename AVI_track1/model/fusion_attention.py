import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Note:gpt生成的一个模型
'''

class DynamicMLP(nn.Module):
    """动态MLP模块,可自定义层数和隐藏维度"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=None, activation=nn.ReLU, dropout=0.0):
        super(DynamicMLP, self).__init__()
        layers = []
        
        # 如果未指定输出维度，默认与隐藏维度相同
        if output_dim is None:
            output_dim = hidden_dim
            
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))  # FIXME: 这里的hidden_dim可以是任意值
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)

class ModalAttention(nn.Module):
    """可配置深度的模态注意力模块"""
    def __init__(self, num_modalities, hidden_dim, attention_layers=2):
        super(ModalAttention, self).__init__()
        self.attention = DynamicMLP(
            input_dim=num_modalities * hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=attention_layers,
            output_dim=num_modalities,
            activation=nn.ReLU,
            dropout=0.0
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, *modalities):
        # 连接所有模态
        concat = torch.cat([x for x in modalities], dim=1)
        # 计算注意力权重
        weights = self.softmax(self.attention(concat))
        # 加权融合
        # weighted_modalities = [w * m for w, m in zip(weights.t(), modalities)]

        weighted_modalities = []
        for j, m in enumerate(modalities):
            w = weights[:, j]  # 取出第j个模态的权重，形状是 (B,)
            while w.dim() < m.dim():
                w = w.unsqueeze(-1)  # 在末尾增加维度方便广播
            weighted_modalities.append(w * m)

        fused = torch.sum(torch.stack(weighted_modalities, dim=0), dim=0)
        return fused, weights

class AVTAttentionRegressionNetwork(nn.Module):
    """完全可配置的多模态注意力回归网络"""
    def __init__(
        self, 
        audio_dim, video_dim, text_dim, 
        hidden_dim=128, 
        output_dim=1,
        projection_layers=1,
        attention_layers=2,
        regressor_layers=2,
        dropout=0.5,
        args=None
    ):
        super(AVTAttentionRegressionNetwork, self).__init__()
        
        # 可配置的模态投影层
        self.audio_proj = DynamicMLP(
            input_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_layers=projection_layers,
            output_dim=hidden_dim,
            activation=nn.ReLU,
            dropout=dropout
        )
        
        self.video_proj = DynamicMLP(
            input_dim=video_dim,
            hidden_dim=hidden_dim,
            num_layers=projection_layers,
            output_dim=hidden_dim,
            activation=nn.ReLU,
            dropout=dropout
        )
        
        self.text_proj = DynamicMLP(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            num_layers=projection_layers,
            output_dim=hidden_dim,
            activation=nn.ReLU,
            dropout=dropout
        )

        self.num_modalities = args.num_modalities  # 这里假设有3个模态
        
        # 可配置的模态注意力模块
        self.modal_attention = ModalAttention(
            num_modalities=self.num_modalities,
            hidden_dim=hidden_dim,
            attention_layers=attention_layers
        )


        self.ffn = DynamicMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim*2,
            num_layers=1,
            output_dim=hidden_dim,
            activation=nn.ReLU,
            dropout=dropout
        )
        
        # 可配置的回归器
        self.regressor = DynamicMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=regressor_layers,
            output_dim=output_dim,
            activation=nn.ReLU,
            dropout=dropout
        )
        
    def forward(self, features_dict):
        """
        处理字典格式的输入,包含audio、video和text三个模态的特征
        """
        projected_features = []
    
        # 动态处理每个存在的模态
        for modality in features_dict:
            if modality == 'audio' and hasattr(self, 'audio_proj'):
                # print(f"audio shape: {features_dict[modality].shape}")  # 例如 (3919, 5, 384)
                audio_features = features_dict[modality]
                features_pooled = audio_features.mean(dim=2)
                audio_features1 = features_pooled.mean(dim=1)
                projected_features.append(self.audio_proj(audio_features1))  # whisper:(384)

            elif modality == 'video' and hasattr(self, 'video_proj'):
                projected_features.append(self.video_proj(features_dict[modality]))
            elif modality == 'text' and hasattr(self, 'text_proj'):
                projected_features.append(self.text_proj(features_dict[modality]))  #robeta:(768)
    
        # 模态注意力融合（动态解包）
        fused_features, _ = self.modal_attention(*projected_features)

        # 经过FFN
        fused_features = self.ffn(fused_features)
    
        # 回归预测
        predictions = self.regressor(fused_features)  # 调整regressor的层数可以对融合后的模型拟合能力调整，他本质上相当于MLP后的FFN
    
        return predictions