# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:11:18 2017

@author: maohui
"""
import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_diff(x):
    return  sigmoid(x) * (1 - sigmoid(x));


def mean_square_error(y, y_):
    return np.sum((y - y_) * (y - y_)) / len(y)

def relu(x):
    idx = x < 0
    x[idx] = 0
    return x

def relu_diff(x):
    idx = x < 0
    x[idx] = 0
    x[~idx] = 1
    return x



# @image [num, channel, height, weidth]
# @kernel [out_channel, in_channel, kernel_h, kernel_w]
def conv(image, kernel, stride = 1):
    if(len(image.shape) == 3):
        image = image[None,]
    num = image.shape[0]
    height = image.shape[2]
    width = image.shape[3]
    out_channel, in_channel, kernel_h, kernel_w = kernel.shape
    assert((height-kernel_h) % stride == 0)
    assert((width-kernel_w) % stride == 0)
    H = (height - kernel_h) / stride + 1
    W = (width - kernel_w) / stride + 1
    out_put = np.zeros([num, out_channel, H, W])
    for n in range(num):
        for c in range(out_channel):
            for h in range(H):
                for w in range(W):
                    out_put[n, c, h, w] = np.sum(image[n, :, h*stride:h*stride+kernel_h,
                               w*stride:w*stride+kernel_w] * kernel[c])
                
    return out_put



























