### [Modeling Rater Preference and Rater Calibration from Optic Disc and Cup Multi-rater Segmentation Annotations]
------
![image](image0.png) 
------
### Introduction 

##### The segmentation of the optic disc (OD) and optic cup (OC) from fundus images suffers from rater variation due to differences in raters’ expertise and image blurriness. In clinical, we can fuse the annotations of multiple raters to reduce rater-related biases through methods such as majority voting. However, these methods ignores the unique preferences of each rater. In this paper, we propose a novel neural network framework to jointly learn rater calibration and preference for multiple annotations of OD and OC segmentation, which consists of two main parts. In the first part, we model multi-annotation as a multi-class segmentation problem to learn calibration segmentation. Further, we propose a multi-annotation smoothing network (MSNet), which can effectively remove high-frequency components in calibration predictions. In the second part, we represent different raters with specific codes, which are used as parameters of the model and optimized during training. In this way, we can achieve modeling different rater preferences under a single network, and the learned rater codes can represent differences in preference patterns among different raters.
------
### Framework
------
![image](image1.png) 
------
### > Requirment
+ pytorch 1.0.0+
+ torchvision
+ PIL
+ numpy
+ tensorboard==1.7.0
+ tensorboardX==2.0

```
### Run MRNet Code

1. train

```python main.py```                              # set '--phase' as train cal
```python main_apm.py```                          # set '--phase' as train apm

2. test

```python test_DCNet.py```                        # set '--phase' as test


### Dataset
1. RIGA benchmark: you can access to this download [link](https://pan.baidu.com/s/1CJzY6WYJfyLwDEKGo_D2IQ), with fetch code (**1627**) or [Google Drive](https://drive.google.com/drive/folders/1Oe9qcuV0gJQSe7BKw1RX_O1dGwenIx9i?usp=sharing). 
```
### Contact Us
If you have any questions, please contact us ( 9469044@qq.com ).

