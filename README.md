# PyNet
目标：一种构建深度卷积神经网络的框架方法
任务：1. 
     2.
     3.
     4.
     5.
基于Python的Numpy的简单的深度学习框架，梯度计算主要参考了Caffe的计算方式。
利用该框架定义一个具体的任意深度的卷积神经网络如下：

    net = Net() 
    
    net.add(conv_layer(16, 1, (2, 2)))
    
    net.add(activation('relu'))
    
    net.add(conv_layer(32, 1, (2, 2)))
    
    net.add(activation('relu'))
    
    net.add(reshape())
    
    net.add(inner_product(100))
    
    net.add(activation('relu'))
    
    net.add(inner_product(1))
    
    net.add(activation('sigmoid'))
    
    
