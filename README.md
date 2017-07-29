# PyNet
基于Python的Numpy的简单的深度学习框架，梯度计算主要参考了Caffe的计算方式。
使用很简单，比如定义一个简答的卷积网络：
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
