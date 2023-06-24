# Unsupervised-Learning-Defect-Inspection-on-Textured-Surfaces
基于卷积自编码器和图像金字塔的布料缺陷无监督学习与检测方法

图形图像处理大作业

在论文《An Unsupervised-Learning-Based Approach for Automated Defect Inspection on Textured Surfaces》的基础上进行了复现

其中自编码器使用pytorch实现。

结果图
![屏幕截图 2023-06-24 195051](https://github.com/Cerber2ol8/Unsupervised-Learning-Defect-Inspection-on-Textured-Surfaces/assets/41719978/d59f4e89-54e6-49bd-af49-7104b453088c)
![屏幕截图 2023-06-24 183316](https://github.com/Cerber2ol8/Unsupervised-Learning-Defect-Inspection-on-Textured-Surfaces/assets/41719978/e4c18fe0-9aee-457b-b16e-856200a97310)


## 主要功能介绍
本文采用基于无监督学习的方法，基于论文《An Unsupervised Learning Based Approach for Automated Defect Inspection on Textured Surfaces》进行复现实验验证，在自建无纺布匹数据集上进行训练和检测。

因为对缺陷进行标记或像素级分割很困难，缺陷的类型也十分复杂，所以基于学习的纹理缺陷检测大体思路是无监督的。即利用无监督学习算法学习正常纹理的数据分布特征，而不学习缺陷的数据分布特征。在待测图上以滑动区域为重构对象，与原图像做残差。由于正常纹理学习充分，重构残差应当很小，而缺陷区域的残差较大，故被凸现出来，随后再利用残差图做进一步处理。

在空域上进行无监督学习主要用卷积自编码器，其有两部分组成，编码器和解码器。编码器由卷积、激活、池化操作做成，对原始数据域分层做特征提取和降维，最后将数据映射到一个欠完备的隐层特征空间上。解码器由上采样、卷积、激活操作完成，将特征空间上的数据映射回空域。

本文主要功能实现：使用训练好的多尺度降噪自编码器对图像patch进行预测，将生成的图像patch与原始输入的patch进行对比，计算其重构残差，由于自编码器训练过程中使用的是无缺陷的图像，因而自编码器会对有缺陷的图像patch更加敏感，致使残差值偏高，将残差值与训练集的统计量进行比对，根据设置的阈值来判断当前像素位置是否为缺陷，从而实现像素级的缺陷检测。

## 模型及实现介绍

使用无瑕疵的训练集，在不同的图像金字塔尺度上(使用高斯金字塔，金字塔尺寸为512，256，128)训练一个自编码器(包含一个Decoder和Encoder)，将原始图像裁剪为8*8的图像patch块，然后对其添加噪声，使用自编码器对这些图像patch进行处理，计算自编码器对每个图像patch的重建输出x’与输入x的残差|x-x’|，然后使用在不同金字塔尺度下设定好的阈值，来对重构残差进行判断，当存在多个尺度下的残差都超过阈值就认为该像素区域为缺陷像素。

实现介绍：

### **1.数据收集和处理部分。**


**数据收集：**使用的数据为先前项目中使用的实际无纺布图像数据，原始图像为6800*8000，使用python脚本将其裁剪为多个512*512的图像。该步骤使用的脚本为cropImage.py。

**光照归一化：**由于数据图像本身光照足够，本文没有进行光照归一化，另一个原因是根据论文中给出的韦伯光照归一化公式，使用代码复现后发现效果不佳，无法还原至论文中的效果，代码为WLD.py。

**加入噪声：**使用椒盐噪声进行噪声腐蚀，噪声系数为0.01。相关代码位于utils.py

**图像金字塔和Patch提取：**使用opencv构建图像金字塔，然后将每个层级的图像裁剪为8*8的图像patch，然后保存为训练数据集。相关代码位于preprocess.py


### **2.模型搭建和训练**

使用pytorch搭建自编码器网络，代码位于CDAE.py，其结构为

| Encoder  Layers | Input channel | Output channel | Kernel size | Stride | Padding |
| --------------- | ------------- | -------------- | ----------- | ------ | ------- |
| Conv2d          | 3             | 64             | (3, 3)      | (1, 1) | (1, 1)  |
| ReLU            |               |                |             |        |         |
| MaxPool2d       |               | 2              | 2           | 2      | 0       |
| Conv2d          | 64            | 128            | (3, 3)      | (1, 1) | (1, 1)  |
| ReLU            |               |                |             |        |         |
| MaxPool2d       |               | 2              | 2           | 2      | 0       |
| Conv2d          | 128           | 256            | (3, 3)      | (1, 1) | (1, 1)  |
| ReLU            |               |                |             |        |         |
| MaxPool2d       |               | 2              | 2           | 2      | 0       |

 

 

| Decoder  Layers | Input channel | Output channel | Kernel size | Stride | Padding |
| --------------- | ------------- | -------------- | ----------- | ------ | ------- |
| ConvTranspose2d | 256           | 128            | (3, 3)      | (2, 2) | (1, 1)  |
| ReLU            |               |                |             |        |         |
| ConvTranspose2d | 128           | 64             | (3, 3)      | (2, 2) | (1, 1)  |
| ReLU            |               |                |             |        |         |
| ConvTranspose2d | 64            | 3              | (3, 3)      | (2, 2) | (1, 1)  |
| ReLU            |               |                |             |        |         |


训练过程使用重构均方误差作为损失函数，使用0.001的学习率，使用SGD作为优化器，weight_decay设置为0.01。对每个尺度的图像进行10个epoch的训练，由于时间问题本文没有做出可视化训练过程。相关代码为train.py。

### **3.残差计算和阈值确定**
将数据集图像按照图像金字塔的patch输入训练好的自编码器，将每个patch的残差作为一个残差集合的分布，计算所有patch的残差，并统计该残差集合的均值和方差。重构残差的分布情况如下：

size_128:

    mean: 0.9765125426365454,    std: 0.6384481523188913
    
size_256:

    mean: 0.990210571639617,    std: 0.0926547572016716
    
size_512:

    mean: 0.9766120079195384,    std: 0.3619536757469177
    
根据残差的分布（已知为长尾分布）情况，选用μ+σ作为残差的判别阈值。相关代码为residual.py。

### **4.测试和结果**

测试图像不需要进行噪声腐蚀，直接输入原始图像进行残差计算。每个像素都对应一个patch图作为残差信息来源，通过该值与阈值进行比对，来确定该像素是否符合正常分布，如果超过阈值则认为该像素邻域为缺陷分布，除此之外，还需要根据不同尺度下的残差阈值，综合判断该像素邻域是否为缺陷特征，实验中认为超过当达到两个尺度下都超过阈值，就认为该像素邻域为缺陷。

原论文并没有介绍如何将三个尺度的残差进行融合，因此本文采取的方案是使用多个尺度下训练的模型进行推理，并根据相应的结果使用RGB三个颜色通道来表示当前像素邻域的缺陷情况。
相关代码为test.py。

使用一张包含缺陷的图像进行检测结果如下：

使用单个尺度下（256）的缺陷图：
![屏幕截图 2023-06-24 195051](https://github.com/Cerber2ol8/Unsupervised-Learning-Defect-Inspection-on-Textured-Surfaces/assets/41719978/d59f4e89-54e6-49bd-af49-7104b453088c)
多个尺度混和下的缺陷图（R:512,G:256,B:128）：
![屏幕截图 2023-06-24 183316](https://github.com/Cerber2ol8/Unsupervised-Learning-Defect-Inspection-on-Textured-Surfaces/assets/41719978/e4c18fe0-9aee-457b-b16e-856200a97310)

多个尺度混和的情况下，结果图像对推理输出的矩阵做了resize操作以匹配不同尺度下的像素位置。

## 结果分析

**优势：**
实验中的瑕疵能够被正确的检测出来，平均来看，使用多尺度进行检测能够加准确地实现像素级检测，并且使用无监督方法不需要人工进行繁重的缺陷标注工作，与有监督学习的方式比较，该方法更便捷和可行。


**缺点：**
但是由于检测过程使用了三个尺度的模型进行推理，实测下来速度不佳，并且训练和推理过程需要进行patch提取操作，以及后续需要手动进行阈值设定，相对于各种端到端的模型，这些过程较为繁琐。除此之外，根据论文的描述来看，该方式可能对于布匹之外瑕疵检测效果可能表现不佳。



