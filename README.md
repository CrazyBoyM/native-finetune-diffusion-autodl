# Native Finetune Diffusion 集成训练环境
一种更native的stable diffusion模型finetune方式 (改进于原始的dreambooth)

本环境是对日本大佬@Kohya S方法的在线运行封装
https://note.com/kohya_ss/n/nbf7ce8d80f29  

主要包含以下特性：
- 支持finetune unet
- 支持finetune text encoder
- 支持更长的clip token输入长度限制（75， 150， 225）
- 支持采用clip倒数第二层的输出（以更接近NovelAI的训练技巧）
- 支持使用非固定分辨率训练，以保留更多图像信息（无需先裁剪数据集为512x512）
- 支持图像的自动标注（prompt / tags）
- 支持训练Stable diffusion 2.0 

## 如何使用
可直接在autodl上使用环境在线运行训练（1块钱每小时）：[点击运行](https://www.codewithgpu.com/i/CrazyBoyM/native-finetune-diffusion-autodl/native-finetune-diffusion)  
也可以到codewithgpu界面把docker镜像下载到本地使用nvidia-docker运行。
