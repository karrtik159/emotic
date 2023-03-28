

AI技术已经应用到了我们生活中的方方面面，而目标检测是其中应用最广泛的算法之一，疫情测温仪器、巡检机器人、甚至何同学的airdesk中都有目标检测算法的影子。下图就是airdesk，何同学通过目标检测算法定位手机位置，然后控制无线充电线圈移动到手机下方自动给手机充电。这看似简单的应用背后其实是复杂的理论和不断迭代的AI算法，今天笔者就教大家如何快速上手目标检测模型YOLOv5，并将其应用到情感识别中。

<img src="images/airdesk.gif">



# 一、背景

今天的内容来源于2019年发表在T-PAMI上的一篇文章[1]，在这之前已经有大量研究者通过AI算法识别人类情感，不过本文的作者认为，人们的情感不仅与面部表情和身体动作等有关，还和当前身处的环境息息相关，比如下图的男孩应该是一个惊讶的表情：

<img src="images/amaze_partial.png" width=30%>

不过加上周围环境后，刚刚我们认为的情感就与真实情感不符：

<img src="images/amaze_full.jpg" width=70%>

本文的主要思想就是将背景图片和目标检测模型检测出的人物信息结合起来识别情感。其中，作者将情感分为离散和连续两个维度。

| 连续情感      | 解释                                                         |
| ------------- | ------------------------------------------------------------ |
| Valence (V)   | measures how positive or pleasant an emotion is, ranging from negative to positive（高兴程度） |
| Arousal (A)   | measures the agitation level of the person, ranging from non-active / in calm to agitated / ready to act（激动程度） |
| Dominance (D) | measures the level of control a person feels of the situation, ranging from submissive / non-control to dominant / in-control（气场大小） |

| 离散情感        | 解释                                                         |
| --------------- | ------------------------------------------------------------ |
| Affection       | fond feelings; love; tenderness                              |
| Anger           | intense displeasure or rage; furious; resentful              |
| Annoyance       | bothered by something or someone; irritated; impatient; frustrated |
| Anticipation    | state of looking forward; hoping on or getting prepared for possible future events |
| Aversion        | feeling disgust, dislike, repulsion; feeling hate            |
| Confidence      | feeling of being certain; conviction that an outcome will be favorable; encouraged; proud |
| Disapproval     | feeling that something is wrong or reprehensible; contempt; hostile |
| Disconnection   | feeling not interested in the main event of the surrounding; indifferent; bored; distracted |
| Disquietment    | nervous; worried; upset; anxious; tense; pressured; alarmed  |
| Doubt/Confusion | difficulty to understand or decide; thinking about different options |
| Embarrassment   | feeling ashamed or guilty                                    |
| Engagement      | paying attention to something; absorbed into something; curious; interested |
| Esteem          | feelings of favourable opinion or judgement; respect; admiration; gratefulness |
| Excitement      | feeling enthusiasm; stimulated; energetic                    |
| Fatigue         | weariness; tiredness; sleepy                                 |
| Fear            | feeling suspicious or afraid of danger, threat, evil or pain; horror |
| Happiness       | feeling delighted; feeling enjoyment or amusement            |
| Pain            | physical suffering                                           |
| Peace           | well being and relaxed; no worry; having positive thoughts or sensations; satisfied |
| Pleasure        | feeling of delight in the senses                             |
| Sadness         | feeling unhappy, sorrow, disappointed, or discouraged        |
| Sensitivity     | feeling of being physically or emotionally wounded; feeling delicate or vulnerable |
| Suffering       | psychological or emotional pain; distressed; anguished       |
| Surprise        | sudden discovery of something unexpected                     |
| Sympathy        | state of sharing others emotions, goals or troubles; supportive; compassionate |
| Yearning        | strong desire to have something; jealous; envious; lust      |

# 二、准备工作与模型推理

## 2.1 快速入门

只需完成下面五步即可识别情感！

1. 通过克隆或者压缩包将项目下载到本地：git clone https://github.com/chenxindaaa/emotic.git

2. 将解压后的模型文件放到emotic/debug_exp/models中。（模型文件下载地址：链接：https://pan.baidu.com/s/1rBRYXpxyT_ooLCk4hmXRRA 提取码：x2rw ）

3. 新建虚拟环境（可选）：

```
conda create -n emotic python=3.7
conda activate emotic
```

4. 环境配置

```
python -m pip install -r requirement.txt
```

5. cd到emotic文件夹下，输入并执行:

```
python detect.py
```

运行完后结果会保存在emotic/runs/detect文件夹下。

## 2.2 基本原理

看到这里可能会有小伙伴问了：如果我想识别别的图片该怎么改？可以支持视频和摄像头吗？实际应用中应该怎么修改YOLOv5的代码呢？

对于前两个问题，YOLOv5已经帮我们解决，我们只需要修改detect.py中的第158行：



将'./testImages'改为想要识别的图像和视频的路径，也可以是文件夹的路径。对于调用摄像头，只需要将'./testImages'改为'0'，则会调用0号摄像头进行识别。

**修改YOLOv5：**

在detect.py中，最重要的代码就是下面几行：


其中det是YOLOv5识别出来的结果，例如tensor([[121.00000,   7.00000, 480.00000, 305.00000,   0.67680,   0.00000], [278.00000, 166.00000, 318.00000, 305.00000,   0.66222,  27.00000]])就是识别出了两个物体。

xyxy是物体检测框的坐标，对于上面的例子的第一个物体，xyxy = [121.00000,   7.00000, 480.00000, 305.00000]对应坐标(121, 7)和(480, 305)，两个点可以确定一个矩形也就是检测框。conf是该物体的置信度，第一个物体置信度为0.67680。cls则是该物体对应的类别，这里0对应的是“人”，因为我们只识别人的情感，所以cls不是0就可以跳过该过程。这里我用了YOLOv5官方给的推理模型，其中包含很多类别，大家也可以自己训练一个只有“人”这一类别的模型，详细过程可以参考:

[使用YOLOv5模型进行目标检测！]: https://mp.weixin.qq.com/s/JgoaLeYTAhDUnQ-ZLEvxow
[用YOLOv5模型识别出表情！]: https://mp.weixin.qq.com/s/LdCuXL49P2JhDoz9iY8wqA

在识别出物体坐标后输入emotic模型就可以得到对应的情感，即


这里我将原来的图片可视化做了些改变，将emotic的结果打印到图片上：


运行结果：

<img src="./images/happy.png" width="500">

完成了上面的步骤，我们就可以开始整活了。众所周知，特朗普以其独特的演讲魅力征服了许多选民，下面我们就看看AI眼中的特朗普是怎么演讲的：

<img src="images/trump.gif">

可以看出自信是让人信服的必备条件之一。

# 三、模型训练

## 3.1 数据预处理

首先通过格物钛进行数据预处理，在处理数据之前需要先找到自己的accessKey(开发者工具$\rightarrow$AccessKey$\rightarrow$新建AccessKey)：

<img src="./images/ak.jpg">

我们可以在不下载数据集的情况下，通过格物钛进行预处理，并将结果保存在本地（下面的代码不在项目中，需要自己创建一个py文件运行，记得填入AccessKey）：



等程序运行完成后可以看到多了一个文件夹emotic_pre，里面有一些npy文件则代表数据预处理成功。

## 3.2 模型训练

打开main.py文件，35行开始是模型的训练参数，运行该文件即可开始训练。

# 四、Emotic模型详解

## 4.1 模型结构

<img src='./images/pipeline.png'>

该模型的思想非常简单，流程图中的上下两个网络其实就是两个resnet18，上面的网络负责提取人体特征，输入为$128 \times 128$的彩色图片，输出是512个$1 \times 1$的特征图。下面的网络负责提取图像背景特征，预训练模型用的是场景分类模型places365，输入是$224\times 224$的彩色图片，输出同样是是512个$1\times 1$的特征图。然后将两个输出flatten后拼接成一个1024的向量，经过两层全连接层后输出一个26维的向量和一个3维的向量，26维向量处理26个离散感情的分类任务，3维向量则是3个连续情感的回归任务。



离散感情是一个多分类任务，即一个人可能同时存在多种感情，作者的处理方法是手动设定26个阈值对应26种情感，输出值大于阈值就认为该人有对应情感，阈值如下，可以看到engagement对应阈值为0，也就是说每个人每次识别都会包含这种情感：

```
>>> import numpy as np
>>> np.load('./debug_exp/results/val_thresholds.npy')
array([0.0509765 , 0.02937193, 0.03467856, 0.16765128, 0.0307672 ,
       0.13506265, 0.03581731, 0.06581657, 0.03092133, 0.04115443,
       0.02678059, 0.        , 0.04085711, 0.14374524, 0.03058549,
       0.02580678, 0.23389584, 0.13780132, 0.07401864, 0.08617007,
       0.03372583, 0.03105414, 0.029326  , 0.03418647, 0.03770866,
       0.03943525], dtype=float32)
```

## 4.2 损失函数：

对于**分类任务**，作者提供了两种损失函数，一种是普通的均方误差损失函数（即self.weight_type == 'mean'），另一种是加权平方误差损失函数（即self.weight_type == 'static‘）。其中，加权平方误差损失函数如下，26个类别对应的权重分别为[0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620, 0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907, 0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537]。
$$
L(\hat y) = \sum^{26}_{i=1}w_i(\hat y_i - y_i)^2
$$




对于**回归任务**，作者同样提供了两种损失函数，L2损失函数：
$$
L_2(\hat y) = \sum^3_{k=1}v_k(\hat y_k - y_k)^2
$$
其中，当$|\hat y_k - y_k|<margin$(默认是1)时，$v_k=0$，否则$v_{k} = 1$。

L1损失函数：
$$
L1(\hat y) = \sum_{k=1}^3v_k\left\{
\begin{aligned}
0.5x^2,   \qquad&|x_k| <margin\\
|x_k| - 0.5,  \qquad&otherwise
\end{aligned}
\right.
$$
其中$x_k = (\hat y_k - y_k)$。

