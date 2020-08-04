---
layout:     post
title:      CS231N
subtitle:   Convolution
date:       2020-08-04
author:     flyshare
header-img: img/post-bg-github-cup.jpg
catalog: true
tags:
    - cs231n
---

##### Convolution的前向传播

```python

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    padded_x = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
    H1 = np.int(1 + ((H + 2 * pad - HH) / stride))
    W1 = np.int(1 + ((W + 2 * pad - WW) / stride))

    out = np.zeros([N, F, H1, W1])

    for nn in range(N):  #一共有N张图片要做卷积
        for ff in range(F):  # 有F个卷积核 在和F个卷积核做卷积运算后，得到的shape的channel是F
            for hh in range(H1):  #直接用最后输出的高和宽做循环遍历 H1*W1就是做卷积运算的次数
                for ww in range(W1):
                    out[nn,ff,hh,ww] = np.sum(padded_x[nn, :, hh*stride:hh*stride+HH, ww*stride:ww*stride+WW] * w[ff,...]) + b[ff]
					# [nn, :, hh*stride:hh*stride+HH, ww*stride:ww*stride+WW]                                 第ff个卷积核
					#   b  c           h                       w

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


```



#####  Convolution的反向传播

```python

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    stride = conv_param['stride']
    pad = conv_param['pad']
    x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')  #用0填充

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, Hh, Ww = dout.shape

    dx = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    #db
    for ff in range(F):
        db[ff] = np.sum(dout[:,ff,:,:])

    #dw and dx
    for nn in range(N):
        for ff in range(F):
            for hh in range(Hh):
                for ww in range(Ww):
                    dw[ff,...] += x_pad[nn,:,hh*stride:(hh*stride+HH),ww*stride:(ww*stride+WW)]*dout[nn,ff,hh,ww]
                    dx[nn,:,hh*stride:(hh*stride+HH),ww*stride:(ww*stride+WW)] += w[ff,...] * dout[nn,ff,hh,ww]
    dx = dx[:,:,pad:-pad,pad:-pad] #updating according to dimensions
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
	
	
```

