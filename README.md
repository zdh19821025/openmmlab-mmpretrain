题目：基于 ResNet50 的水果分类

背景：使用基于卷积的深度神经网络 ResNet50 对 30 种水果进行分类

任务

划分训练集和验证集
按照 MMPreTrain CustomDataset 格式组织训练集和验证集
使用 MMPreTrain 算法库，编写配置文件，正确加载预训练模型
在水果数据集上进行微调训练
使用 MMPreTrain 的 ImageClassificationInferencer 接口，对网络水果图像，或自己拍摄的水果图像，使用训练好的模型进行分类
需提交的验证集评估指标（不能低于 60%）
![image](https://github.com/zdh19821025/openmmlab-mmpretrain/assets/54253071/1eebaaf3-6460-4b0e-b99d-37949703aedd)

验证集评估指标见日志文件 20230608_082233.log

作业数据集下载：
链接：https://pan.baidu.com/s/1YgoU1M_v7ridtXB9xxbA1Q
提取码：52m9


推理代码：
from mmpretrain import ImageClassificationInferencer
image = r'E:\AI_mode\mmpretrain-1.0.0rc8\data\fruit30\val\菠萝\161.jpeg'
config = r'E:\AI_mode\mmpretrain-1.0.0rc8\configs\resnet50_8xb32-coslr_in1k.py'

checkpoint = r'E:\AI_mode\mmpretrain-1.0.0rc8\work_dirs\resnet50_8xb32-coslr_in1k\epoch_300.pth'
inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device='cuda')
result = inferencer(image)[0]
print(result['pred_class'])

推理结果：
Inference ---------------------------------------- 100% 0:00:00菠萝
