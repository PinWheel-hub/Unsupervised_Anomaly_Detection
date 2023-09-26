# 无监督异常检测方案


## 1. 无监督异常检测算法介绍

无监督异常检测(UAD)算法的技术方向可分为基于表示的算法和基于重构的算法。

基于表示的算法思想是希望通过某种映射将原输入图片映射到某个特征空间，在此特征空间可以更容易区分正常样本与异常样本，此类算法的特点是速度较快，但只能够实现patch级的分割；
基于重构的算法思想是仅使用正常样本训练的重构模型只能很好的重构正常样本，而不能准确地重构缺陷样本，从而可对比重构误差来检测缺陷样本，此类算法的特点是能够实现像素级的分割，但速度较慢；
在具体实现时，基于表示的算法通常采用在ImageNet上预训练的backbone提取图片的特征，在预测时通过比对正常样本与异常样本的特征的差异进行缺陷的分类与分割；
基于重构的算法通常采用自编码器作为重构模型，在预测时通过比对重构前后图片的差异进行缺陷的分类与分割。

目前UAD方案支持四种种基于表示的无监督异常检测算法，分别是[PaDiM](../../configs/uad/padim/README.md), [PatchCore](../../configs/uad/patchcore/README.md)以及[STFPM](../../configs/uad/stfpm/README.md)，以及本项目的local coreset，其中，PaDiM、PatchCore、local coreset无需训练网络。

## 2. 数据集准备

此工具以MVTec AD数据集为例, 首先下载数据集[下载MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/), 将数据集保存在`data`中，目录结构如下：

```
data/mvtec_anomaly_detection/
    |
    |--bottle                    # 某类产品
    |  |--ground_truth           # 标签图
    |     |--broken_large        # 某类缺陷标签图
    |        |--000_mask.png  
    |        |--001_mask.png
    |        |--...
    |  |  |--broken_small        # 某类缺陷标签图
    |        |--...
    |  |  |--contamination       # 某类缺陷标签图
    |        |--...
    |  |--test                   # 测试样本
    |     |--good                # 正常样本测试图
    |        |--000.png
    |        |--...
    |     |--broken_large        # 某类缺陷测试图
    |        |--000.png
    |        |--...
    |     |--broken_small        # 某类缺陷测试图
    |        |--...
    |     |--contamination       # 某类缺陷测试图
    |        |--...
    |  |--train                  # 训练样本
    |     |--good                # 正常样本训练图
    |        |--000.png
    |        |--...
    ...

```

MVTec AD数据包含结构和纹理类型的零件共计15类，其中训练集只包含OK图像，测试集包含NG和OK图像，示意图如下：

![](https://github.com/Sunting78/images/blob/master/mvtec.png)

另外，如果希望使用自己的数据集, 请组织成上述MVTec AD数据集的格式, 将自定义数据集作为MVTec AD中的一个category，即路径设置为`QualityInspector/data/mvtec_anomaly_detection/{category}/...`，标签文件为灰度图, 缺陷部分像素值为255;


## 2. 训练、评估、预测命令

训练、评估、预测脚本存放在`tools/uad/*/`目录下，使用script.py调用脚本，通过`config`参数传入对应模型YML配置文件:

* 训练:

```python script.py train configs/local_coreset.yml```

* 评估:

```python script.py val configs/local_coreset.yml```

* 预测:

```python script.py predict configs/local_coreset.yml```


## 3. 配置文件解读

无监督异常检测(uad)模型的参数可以通过YML配置文件和命令行参数两种方式指定, 如果YML文件与命令行同时指定一个参数, 命令行指定的优先级更高, 以PaDiM的YML文件为例, 主要包含以下参数:

```
# common arguments
device: cuda:0 # 硬件设备
seed: 3  # 指定numpy, torch的随机种子
model: &model patchcore # 算法
backbone: &backbone wide_resnet50_2  # Support resnet18, resnet50, wide_resnet50_2
method: &method local_coreset # projection method, one of [sample, ortho, svd_ortho, gaussian, coreset] # 采样方法
k: 10  # using feature channels
pretrained_model_path: null # checkpoints/resnet18.pth

# dataset arguments
batch_size: 1
num_workers: 0
category: &category transistor  # 数据集类别
resize: [256, 256] # 指定读取图像的resize尺寸
crop_size: [256, 256] # 指定resize图像的crop尺寸
data_path: /data/data_wbw/data/mvtec_anomaly_detection/transistor/train/good/ # 训练集路径
save_path: [output/, *method, _, *backbone, _, *category]  # 保存路径
save_name: &save_name transistor.pth # 保存名字

# train arguments
do_eval: True # 指定训练后是否进行评估

# val and predict arguments
test_path: /data/data_wbw/data/mvtec_anomaly_detection/
save_pic: True # Whether to save one output example picture in val and predict;
model_path: [output/, *method, _, *backbone, _, *category, /, *save_name]  # 指定加载模型参数的路径

# predict arguments
img_path: /data/data_wbw/data/mvtec_anomaly_detection/leather/test/hole  # 指定预测的图片路径
result_path: [output/, *model, _, *backbone, _, *category, '/hole']
threshold: 2.2 # 指定阈值
```