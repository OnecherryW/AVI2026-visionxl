### Directory Structure

AVI/
├─ args_log/                                           # 每次不同参数模型的各个参数配置都会在此保存  
├─ data/                                                 
│  ├─ all_data.csv                                     # 包含有训练和测试集的csv文件  
│  ├─ train_data.csv                                   # 训练集csv文件  
│  ├─ val_data.csv                                     # 验证集csv文件，不知道为什么数值列比表头列多一列  
│  └─ val_data_new.csv                                 # 删除多余列的csv文件  
├─ dataset/  
│  ├─ baseline_dataset.py                              # track1 dataset类  
│  └─ baseline_dataset2.py                             # track2 dataset类  
├─ model/                                                
│  ├─ new_model/                                       # 文件夹，我们当前自己的baseline  
│  ├─ baseline_model.py                                # 官方baseline  
│  └─ fusion_attention.py                              # gpt生成的一个baseline和官方效果应该差不多  
├─ save_ckpt/  
│  ├─ track1/                                   
│  └─ track2/  
├─ scripts/                                            # 运行文件  
│  ├─ test_task1.sh  
│  ├─ test_task2.sh  
│  ├─ train_task1.sh  
│  └─ train_task2.sh  
├─ utils/                                              # 检查输入的特征的shape以便于调参  
│  └─ npy_check.py  
├─ README.md  
├─ requirement.txt  
├─ test_task1.py                                   # track1 测试集代码  
├─ test_task2.py                                   # track2 测试集代码  
├─ train_task1.py                                  # track1 训练集代码  
└─ train_task2.py                                  # track2 测试集代码  


### Note
* 当前同模型track1效果好于官方baseline，**track2效果比较差，可能需要换个模型？**
* 4096维度3个模态，如果拼接在一块输出维度非常大影响训练，或许可以尝试按sequence维度拼接，那这样后面的模型就要改动了

### Usage
#### 1. Prepare features
* 确认feature文件夹路径，模型输入是我们提取好的features
* 并确认不同feature的shape，以便于调整模型参数
	* utils/npy_check.py可以帮助确认
#### 2. Install Requirements
```
conda activate [进入自己的环境]
```

```bash
pip install -r /data2/heyichao/AVI/requirement.txt
```
* 不一定全，运行不起来自己按照没安装的生成一下
#### 2. Run the Script
```bash
cd /data2/heyichao/AVI
```

```bash 
bash /data2/heyichao/AVI/scripts/train_task1.sh
```

#### 4. Output
- 模型会保存在save_ckpt/track1 or track文件夹
- 每次训练的args会以json保存在args_log文件夹以及**save_ckpt/track1 or track**文件夹
- 训练完成后的训练曲线也会同步保存在save_ckpt/track1 or track文件夹中
- 如果需要测试，使用**save_ckpt/track1 or track**文件夹中json路径的args传参到测试的bash文件中，不要使用**args_log文件夹**中的json路径，避免对不上
