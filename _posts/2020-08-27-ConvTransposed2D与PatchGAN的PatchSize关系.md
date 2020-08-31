---
layout:     post
title:      ConvTransposed2D与PatchGAN的PatchSize关系
subtitle:   关于PatchGAN如何得到70*70大小的解惑
date:       2020-08-31
author:     flyshare
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - trick
---


ConvTransposed2d()其实是Conv2d()的逆过程，其参数是一样的

　　Conv2d():

　　　　output = (input+2*Padding-kernelSize) / stride + 1(暂时不考虑outputPadding 注意：outputPadding只是在一边Padding)

　　　　=>input  = (output-1) * stride - 2*Padding + kernelSize

　　　　例如输入图片尺寸为128，inputPadding为0，kernelSize为4，stride为2，outputPadding为1，那么输出图片尺寸为64

　　同理可得

　　ConvTransposed2d():

　　　　output = (input-1) * stride - 2*Padding + kernelSize

```python

PatchGAN的最后输出的每一个像素点可以代表一个70*70的感受野，所以从一个像素点一层一层的逆卷积过程，最后得到 70 * 70 的 Receptive field

def f(output_size, ksize, stride):
    return (output_size - 1) * stride + ksize

last_layer = f(output_size=1, ksize=4, stride=1)
# Receptive field: 4
fourth_layer = f(output_size=last_layer, ksize=4, stride=1)
# Receptive field: 7
third_layer = f(output_size=fourth_layer, ksize=4, stride=2)
# Receptive field: 16
second_layer = f(output_size=third_layer, ksize=4, stride=2)
# Receptive field: 34
first_layer = f(output_size=second_layer, ksize=4, stride=2)
# Receptive field: 70

print(first_layer)


```