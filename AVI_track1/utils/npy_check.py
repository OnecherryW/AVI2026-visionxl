import numpy as np
import os

# 指定要检查的 .npy 文件路径
feature_path = "/data2/public_datasets/AVI/official_feature/whisper_feature/5484821efdf99b07b28f2300_q1.npy"  # 替换为你的实际路径



'''
audio shape:(3919, 5, 384)
3919个时间窗口, 每个时间窗口5个token, 每个token 384维
'''
# 判断文件是否存在
if not os.path.isfile(feature_path):
    print(f"文件不存在: {feature_path}")
else:
    # 加载特征并打印 shape
    feature = np.load(feature_path)
    print(f"特征 shape: {feature.shape}")
    print(f"特征类型: {type(feature)}")
    print(f"数据类型: {feature.dtype}")
