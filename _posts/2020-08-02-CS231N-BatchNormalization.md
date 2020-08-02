---
layout:     post
title:      CS231N
subtitle:   Fully_Connect_Layer
date:       2020-07-30
author:     flyshare
header-img: img/tag-bg.jpg
catalog: true
tags:
    - cs231n
---

#### BatchNormalization 前向传播

```python

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape #200，3
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
     
        sample_mean = np.mean(x, axis = 0) #3 均值
        #print('sample_mean.shape',sample_mean.shape)
        sample_var = np.var(x, axis = 0)  #3 方差
		#指数加权平均数得出均值
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #正则化进行处理
		norm = (x - sample_mean) / (np.sqrt(sample_var + eps))
        #print('gamma.shape',gamma.shape)
        out = (gamma * norm) + beta
        cache = (x, norm, sample_mean, sample_var, gamma, eps)

    elif mode == 'test':
      norm = (x - running_mean) / (np.sqrt(running_var + eps))
        out = (gamma * norm) + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

```


#### BatchNormalization 反向传播

```python
    dx, dgamma, dbeta = None, None, None

    (x, norm, sample_mean, sample_var, gamma, eps) = cache
    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(dout * norm, axis = 0)

    X, X_norm, mu, var, gamma, eps = cache
    N, D = X.shape
    X_mu = X - mu
    std_inv = 1. / np.sqrt(var + eps)
	#  out = (gamma * norm) + beta
    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
    dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)
    dx = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
```