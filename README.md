# PyNet
目标：一种构建深度卷积神经网络的方法
任务：1. 
     2.
     3.
     4.
     5.
基于Python的Numpy的简单的深度学习框架，梯度计算主要参考了Caffe的计算方式。
使用很简单，比如定义一个简单的卷积网络：

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
    
    
