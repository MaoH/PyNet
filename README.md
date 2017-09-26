PyNet
====
基于Python的Numpy的简单的深度学习框架，梯度计算主要参考了Caffe的计算方式。
利用该框架定义一个卷积神经网络如下：


``` python
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
 
```
