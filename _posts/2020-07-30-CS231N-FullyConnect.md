---
layout:     post
title:      CS231N
subtitle:   Fully_Connect_Layer
date:       2020-07-30
author:     flyshare
header-img: img/post-bg-swift2.jpg
catalog: true
tags:
    - cs231n
---

# Fully Connect & ReLu Forward Backward

#### Affine

```python
  
def affine_forward(x, w, b): 
    #前向传播没有什么东西的
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    num_train = x.shape[0] #number of training examples
 
    out = np.dot(x.reshape(num_train, -1), w) + b  #将x打平和w做矩阵运算
   
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
  # dout 上一层传过来的梯度,用于链式求导
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    num_train = x.shape[0]
    dx, dw, db = None, None, None
    #利用链式法则求出本层的三个倒数
    db = np.sum(dout, axis = 0)
    dx = np.dot(dout, w.T).reshape(*x.shape) #to match dimensions of x
    dw = np.dot(x.reshape(num_train, -1).T, dout)

    return dx, dw, db

 

```

#### Relu()
```python
 def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = np.maximum(0, x) #ReLU函数实现起来就一句话

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
 
    dx = np.zeros(x.shape)
    dx[x>0] = 1  #小于0求导为0
    dx = dx * dout
  
    return dx


```
