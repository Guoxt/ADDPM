### [Accelerating Denoising Diffusion Probabilistic Model via Truncated Inverse Processes for Medical Image Segmentation]
------
### Introduction 

##### Due to the impressive advancements achieved by Denoising Diffusion Probability Models (DDPMs) in image generation, researchers have explored the possibility of utilizing DDPMs in discriminative tasks to achieve superior performance. However, the inference process of DDPMs is highly inefficient since it requires thousands of iterative denoising steps. In this study, we propose an accelerated denoising diffusion probabilistic model via truncated inverse processes (ADDPM) that is specifically designed for medical image segmentation. The inverse process of ADDPM starts from a non-Gaussian distribution and terminates early once a prediction with relatively low noise is obtained after multiple iterations of denoising. We employ a separate powerful segmentation network to obtain pre-segmentation and construct the non-Gaussian distribution of the segmentation based on the forward diffusion rule. By further adopting a separate denoising network, the final segmentation can be obtained with just one denoising step from the predictions with low noise. ADDPM greatly reduces the number of denoising steps to approximately one-tenth of that in vanilla DDPMs. Our experiments on three segmentation tasks demonstrate that ADDPM outperforms both vanilla DDPMs and existing representative accelerating DDPMs methods. Moreover, ADDPM can be easily integrated with existing advanced segmentation models to improve segmentation performance and provide uncertainty estimation.
------
### Framework
------
![image](https://github.com/Guoxt/ADDPM/assets/46101051/f8479690-0771-49c5-b26b-d5efd27a6ccd)
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

