---
layout:     post
title:      机器学习算法
subtitle:   LogisticRegression
date:       2020-08-20
author:     flyshare
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - Machine Learning Algorithm
---


# 逻辑斯蒂分布(Logistic distribution)

#### 分布函数

<!-- logisticRegressionDistributionFunc.png -->
<p align='center'>
      <img src="/img/logisticRegressionDistributionFunc.png">
</p>

#### 密度函数

<!--LRdensity.png-->

<p align='center'>
      <img src="/img/LRdensity.png">
</p>

#### μ为位置参数，γ大于0为形状参数，(μ,1/2)中心对称

<p align='center'>
      <img src="/img/LR3.png">
</p>

<p align='center'>
      <img src="/img/LR4.png">
</p>

---

# 二项逻辑斯蒂回归

由条件概率P(Y|X)表示的分类模型,形式化为logistic distribution,X取实数，Y取值1,0

<p align='center'>
      <img src="/img/LR5.png">
</p>


事件的几率odds：事件发生与事件不发生的概率之比为

<p align='center'>
      <img src="/img/LE6.png">
</p>

称为事件的发生比(the odds of experiencing an event), 
对数几率：

<p align='center'>
      <img src="/img/LR7.png">
</p>

对逻辑斯蒂回归：

<p align='center'>
      <img src="/img/LR8.png">
</p>

---

# 似然函数

logistic分类器是由一组权值系数组成的，最关键的问题就是如何获取这组权值，通过极大似然函数估计获得，并且Y~f(x;w)

似然函数是统计模型中参数的函数。给定输出x时，关于参数θ的似然函数L(θx)（在数值上）等于给定参数θ后变量X的概率：L(θ|x)=P(X=x|θ)

似然函数的重要性不是它的取值，而是当参数变化时概率密度函数到底是变大还是变小。

极大似然函数：似然函数取得最大值表示相应的参数能够使得统计模型最为合理


那么对于上述m个观测事件，设

<p align='center'>
      <img src="/img/LR9.png">
</p>

其联合概率密度函数，即似然函数为：

<p align='center'>
      <img src="/img/LR10.png">
</p>

目标：求出使这一似然函数的值最大的参数估，w1,w2,…,wn，使得L(w)取得 最大值。
对L(w)取对数
对数似然函数

<p align='center'>
      <img src="/img/LR11.png">
</p>

对L(w)求极大值，得到w的估计值。
通常采用梯度下降法及拟牛顿法，学到的模型：

<p align='center'>
      <img src="/img/LR12.png">
</p>


<p align='center'>
      <img src="/img/LR13.png">
</p>


多项logistic回归

设Y的取值集合为
<p align='center'>
      <img src="/img/LR14.png">
</p>


多项logistic回归模型


<p align='center'>
      <img src="/img/LR15.png">
</p>

<p align='center'>
      <img src="/img/LR16.png">
</p>

