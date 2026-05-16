### Directory Structure

AVI/
├─ dataset/  
│  ├─ baseline_dataset.py                              # track1 dataset类  
│  └─ baseline_dataset2.py                             # track2 dataset类  
├─ model/                                                
│  ├─ new_model/                                       # 文件夹，我们当前自己的baseline  
│  ├─ baseline_model.py                                # 官方baseline  
│  └─ fusion_attention.py                              # gpt生成的一个baseline  
├─ checkpoints/  
│  ├─ q3/                                              #存放模型和对应的超参数文件（JSON）                           
│  └─ q4/
│  └─ q5/
│  └─ q6/
├─ utils/                                              # 检查输入的特征的shape以便于调参  
│  └─ npy_check.py  
├─ README.md  
├─ requirement.txt  
├─ test_task1.py                                   # track1 测试集代码
├─ train_task1.py                                  # track1 训练集代码
├─ train_task1.sh 
├─ test_task1.sh 



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
#### 2. Run the Test Script
```bash 
bash test_task1.sh   # 注意每次只能运行一个维度，运行其他维度要修改脚本中Training_Args_JSON_FILE字段，路径就是checkpoints里面每个维度里面的JSON路径
```

