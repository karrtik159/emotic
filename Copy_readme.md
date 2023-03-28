

AI technology has been applied to all aspects of our lives, and target detection is one of the most widely used algorithms. There are target detection algorithms in epidemic temperature measuring instruments, inspection robots, and even classmate He’s airdesk. The picture below is airdesk. He uses the target detection algorithm to locate the location of the mobile phone, and then controls the wireless charging coil to move to the bottom of the mobile phone to automatically charge the mobile phone. Behind this seemingly simple application is actually a complex theory and an iterative AI algorithm. Today I will teach you how to quickly get started with the target detection model YOLOv5 and apply it to emotion recognition.

<img src="images/airdesk.gif">



# 1. Background

Today's content comes from an article [1] published on T-PAMI in 2019. Before that, a large number of researchers have used AI algorithms to identify human emotions, but the author of this article believes that people's emotions are not only related to facial expressions and It is related to body movements, etc., and is also closely related to the current environment. For example, the boy in the picture below should have a surprised expression:

<img src="images/amaze_partial.png" width=30%>

However, after adding the surrounding environment, the emotion we thought just now does not match the real emotion:

<img src="images/amaze_full.jpg" width=70%>

The main idea of ​​this article is to combine the background image and the character information detected by the target detection model to identify emotions. Among them, the author divides emotion into two dimensions: discrete and continuous.

| Continuous Emotion | Interpretation |
| ------------- | ----------------------------------- ------------------------- |
| Valence (V) | measures how positive or pleasant an emotion is, ranging from negative to positive (degree of happiness) |
| Arousal (A) | measures the agitation level of the person, ranging from non-active / in calm to agitated / ready to act (excitement level) |
| Dominance (D) | measures the level of control a person feels of the situation, ranging from submissive / non-control to dominant / in-control (aura size) |

| Discrete Emotion | Interpretation |
| --------------- | --------------------------------- ------------------------------ |
| Affection | fond feelings; love; tenderness |
| Anger | intense distress or rage; furious; resentful |
| Annoyance | bothered by something or someone; irritated; impatient; frustrated |
| Anticipation | state of looking forward; hoping on or getting prepared for possible future events |
| Aversion | feeling disgust, dislike, repulsion; feeling hate |
| Confidence | feeling of being certain; conviction that an outcome will be favorable; encouraged; proud |
| Disapproval | feeling that something is wrong or reprehensible; contempt; hostile |
| Disconnection | feeling not interested in the main event of the surrounding; indifferent; bored; distracted |
| Disquietment | nervous; worried; upset; anxious; tense;
| Doubt/Confusion | difficulty to understand or decide; thinking about different options |
| Embarrassment | feeling ashamed or guilty |
| Engagement | paying attention to something; absorbed into something; curious;
| Esteem | feelings of favorable opinion or judgment; respect; admiration;
| Excitement | feeling enthusiasm; stimulated; energetic |
| Fatigue | weariness; tiredness; sleepy |
| Fear | feeling suspicious or afraid of danger, threat, evil or pain; horror |
| Happiness | feeling delighted; feeling enjoyment or amusement |
| Pain | physical suffering |
| Peace | well being and relaxed; no worry; having positive thoughts or feelings; satisfied |
| Pleasure | feeling of delight in the senses |
| Sadness | feeling unhappy, sorrow, disappointed, or discouraged |
| Sensitivity | feeling of being physically or emotionally wounded; feeling delicate or vulnerable |
| Suffering | psychological or emotional pain;
| Surprise | sudden discovery of something unexpected |
| Sympathy | state of sharing others emotions, goals or troubles;
| Yearning | strong desire to have something; jealous;

# 2. Preparation and model reasoning

## 2.1 Quick Start

Just complete the five steps below to recognize emotions!

1. Download the project locally by cloning or compressing the package: git clone https://github.com/chenxindaaa/emotic.git

2. Put the decompressed model file into emotic/debug_exp/models. (Model file download address: link: https://pan.baidu.com/s/1rBRYXpxyT_ooLCk4hmXRRA extraction code: x2rw )

3. Create a new virtual environment (optional):

```
conda create -n emotic python=3.7
conda activate emoticon
```

4. Environment configuration

```
python -m pip install -r requirement.txt
```

5. cd to the emotic folder, enter and execute:

```
python detect.py
```

After running, the result will be saved in emotic/runs/detect folder.

## 2.2 Basic Principles

Seeing this, some friends may ask: If I want to recognize other pictures, how should I change it? Can video and camera be supported? How should the code of YOLOv5 be modified in practical application?

For the first two problems, YOLOv5 has solved it for us, we only need to modify line 158 in detect.py:



Change './testImages' to the path of the images and videos you want to recognize, or the path of the folder. For calling the camera, just change './testImages' to '0', and then camera 0 will be called for recognition.

**Modified YOLOv5:**

In detect.py, the most important code is the following lines:


The DET is the result recognized by YOLOV5, such as Tensor ([[121.00000, 7.00000, 480.00000, 305.00000, 0.67680, 0.00000], [278.00000, 166.00000, 318.00000, 305.00000, 0.66222, 27.00000]).

xyxy is the coordinates of the object detection frame. For the first object in the above example, xyxy = [121.00000, 7.00000, 480.00000, 305.00000] corresponds to the coordinates (121, 7) and (480, 305), two points can determine a rectangle That is the detection frame. conf is the confidence of the object, the confidence of the first object is 0.67680. cls is the category corresponding to the object, where 0 corresponds to "person", because we only recognize human emotions, so the process can be skipped if cls is not 0. Here I used the reasoning model officially given by YOLOv5, which contains many categories. You can also train a model that only has the category of "person". For the detailed process, please refer to:

[Target detection using YOLOv5 model! ]: https://mp.weixin.qq.com/s/JgoaLeYTAhDUnQ-ZLEvxow
[Use the YOLOv5 model to recognize expressions! ]: https://mp.weixin.qq.com/s/LdCuXL49P2JhDoz9iY8wqA

After identifying the coordinates of the object, input the emotic model to get the corresponding emotion, that is


Here I made some changes to the visualization of the original picture, and printed the result of emotic on the picture:


operation result:

<img src="./images/happy.png" width="500">

After completing the above steps, we can start the whole work. As we all know, Trump has conquered many voters with his unique speech charm. Let's take a look at how Trump speaks in the eyes of AI:

<img src="images/trump.gif">

It can be seen that self-confidence is one of the necessary conditions for convincing people.

# 3. Model training

## 3.1 Data preprocessing

First, data preprocessing is performed through Gewu Titanium. Before processing data, you need to find your own accessKey (Developer Tools $\rightarrow$AccessKey$\rightarrow$ Create AccessKey):

<img src="./images/ak.jpg">

We can preprocess through Gewuti without downloading the data set, and save the result locally (the following code is not in the project, you need to create a py file to run by yourself, remember to fill in the AccessKey):



After the program is finished running, you can see that there is an additional folder emotic_pre, and there are some npy files in it, which means that the data preprocessing is successful.

## 3.2 Model Training

Open the main.py file, starting from line 35 is the training parameters of the model, run this file to start training.

# Four, Emotic model details

## 4.1 Model structure

<img src='./images/pipeline.png'>

The idea of ​​this model is very simple. The upper and lower networks in the flowchart are actually two resnet18s. The upper network is responsible for extracting human body features. The input is a $128 \times 128$ color picture, and the output is 512 $1 \times 1$ feature map. The following network is responsible for extracting image background features. The pre-training model uses the scene classification model places365. The input is a color image of $224\times 224$, and the output is also 512 feature maps of $1\times 1$. Then the two outputs are flattened and spliced ​​into a 1024 vector. After two layers of fully connected layers, a 26-dimensional vector and a 3-dimensional vector are output. The 26-dimensional vector handles 26 discrete emotion classification tasks, and the 3-dimensional vector is is a regression task for 3 consecutive emotions.



Discrete emotion is a multi-classification task, that is, a person may have multiple emotions at the same time. The author's processing method is to manually set 26 thresholds corresponding to 26 emotions. If the output value is greater than the threshold, it is considered that the person has the corresponding emotion. The threshold is as follows, you can Seeing that the threshold corresponding to engagement is 0, that is to say, everyone will contain this emotion every time they recognize:

```
>>> import numpy as np
>>> np.load('./debug_exp/results/val_thresholds.npy')
array([0.0509765, 0.02937193, 0.03467856, 0.16765128, 0.0307672,
0.13506265, 0.03581731, 0.06581657, 0.03092133, 0.04115443,
0.02678059, 0. , 0.04085711, 0.14374524, 0.03058549,
0.02580678, 0.23389584, 0.13780132, 0.07401864, 0.08617007,
0.03372583, 0.03105414, 0.029326, 0.03418647, 0.03770866,
0.03943525], dtype=float32)
```

## 4.2 Loss function:

For **classification tasks**, the author provides two loss functions, one is the ordinary mean square error loss function (ie self.weight_type == 'mean'), and the other is the weighted square error loss function (ie self .weight_type == 'static'). Among them, the weighted square error loss function is as follows, the weights corresponding to the 26 categories are [0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620, 0.1540, 0.1987, 0.1057, 0.1419.2, 91 0.1158, 0.1907, 0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537].
$$
L(\hat y) = \sum^{26}_{i=1}w_i(\hat y_i - y_i)^2
$$




For **regression task**, the author also provides two loss functions, L2 loss function:
$$
L_2(\hat y) = \sum^3_{k=1}v_k(\hat y_k - y_k)^2
$$
Among them, when $|\hat y_k - y_k|<margin$ (the default is 1), $v_k=0$, otherwise $v_{k} = 1$.

L1 loss function:
$$
L1(\hat y) = \sum_{k=1}^3v_k\left\{
\begin{aligned}
0.5x^2, \qquad&|x_k| <margin\\
|x_k| - 0.5, \qquad&otherwise
\end{aligned}
\right.
$$
where $x_k = (\hat y_k - y_k)$.

