# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:06:00 2017

@author: maohui
"""

import numpy as np

# math function
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_diff(x):
    return  sigmoid(x) * (1. - sigmoid(x));


def mean_square_error(y, y_):
    return np.sum((y - y_) * (y - y_)) / len(y)


def relu(x):
    idx = x < 0
    x[idx] = 0
    return x

def relu_diff(x):
    idx = x < 0
    x[idx] = 0
    x[~idx] = 1.
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
    H = (height - kernel_h) // stride + 1
    W = (width - kernel_w) // stride + 1
    out_put = np.zeros([num, out_channel, H, W])
    for n in range(num):
        for c in range(out_channel):
            for h in range(H):
                for w in range(W):
                    out_put[n, c, h, w] = np.sum(image[n, :, h*stride:h*stride+kernel_h,
                               w*stride:w*stride+kernel_w] * kernel[c])
                
    return out_put







#base layer
class layer:
    def __init__(self, name):
        self.trainable = True
        self.name = name
        
    def forward(self, bottom):
        pass
    
    def backward(self, top_diff):
        pass
    
    def update(self, lr = 0.001):
        pass
    
    
    
#full connected layer
class inner_product(layer):
    def __init__(self, output_shape, initial_method = 'constant', name ='inner_product'):
        layer.__init__(self, name)
        self.input_shape = 0
        self.output_shape = output_shape
        self.initial_method = initial_method
    
    def setup_layer(self):
        if(self.initial_method == 'constant'):
            self.weight = np.zeros([self.input_shape, self.output_shape])
            self.weight += 0.5
            self.bias = np.zeros([self.output_shape])
        else:
            print('NOT IMPLETEMENT')
            exit()
        
    
    def forward(self, bottom):
        if(self.input_shape == 0):
            if(len(bottom.shape) == 2):
                self.input_shape = bottom.shape[1]
            elif(len(bottom.shape) == 1):
                self.input_shape = bottom.shape[0]
            else:
                print('NOT MATCH')
                exit()
            self.setup_layer()
        self.bottom = bottom;
        self.top = np.dot(bottom, self.weight) + self.bias
        return self.top
        
    
    def backward(self, top_diff):
        self.top_diff = top_diff
        self.weight_diff = np.dot(self.bottom.T, top_diff) / len(top_diff)
        self.bias_diff = np.sum(top_diff, axis=0) / len(top_diff)
        self.bottom_diff = np.dot(top_diff, self.weight.T)
        self.update()
        return self.bottom_diff
        
    def update(self,lr = 0.001):
        if(self.trainable == True):
            self.weight -= lr * self.weight_diff
            self.bias -= lr * self.bias_diff
      
        
        
#activation layer
class activation(layer):
    def __init__(self, neural_type = 'relu', name = 'activation'):
        layer.__init__(self,name)
        self.trainable = False
        self.neural_type = neural_type
        if(neural_type == 'relu'):
            pass
        elif(neural_type == 'sigmoid'):
            pass
        else:
            print('NOT IMPLETEMENT')
            exit()
            
    def forward(self, bottom):
        if(self.neural_type == 'relu'):
            return relu(bottom)
        elif(self.neural_type == 'sigmoid'):
            return sigmoid(bottom)
    
    def backward(self, top_diff):
        if(self.neural_type == 'relu'):
            return relu_diff(top_diff)
        elif(self.neural_type == 'sigmoid'):
            return sigmoid_diff(top_diff)
    
            
        
class conv_layer(layer):
    def __init__(self, out_put, stride = 1, kernel_size = (3, 3), pad = 'same', name = 'conv_layer'):
        layer.__init__(self,name)
        self.out_channel = out_put
        self.in_channel = 0
        self.pad = pad
        self.stride = stride
        self.kernel_h, self.kernel_w = kernel_size
        
    def forward(self, bottom):
        if(self.in_channel == 0):
            self.in_channel = bottom.shape[1]
            self.kernel = np.random.uniform(size = [self.out_channel, self.in_channel, self.kernel_h, self.kernel_w])
      
        self.bottom = bottom
        self.padding_h = (bottom.shape[2] - self.kernel_h) % self.stride
        self.padding_w = (bottom.shape[3] - self.kernel_w) % self.stride
        if(self.pad == 'same'):
            if(self.padding_h == 0):
                self.padding_h += self.stride
            if(self.padding_w == 0):
                self.padding_w += self.stride
            bottom = np.zeros([self.bottom.shape[0], 
                                    self.bottom.shape[1], 
                                    self.bottom.shape[2]+self.padding_h,
                                    self.bottom.shape[3]+self.padding_w])
            bottom[:,:,self.padding_h:,self.padding_w:] = self.bottom
            
        self.top = conv(bottom, self.kernel, self.stride)
        return self.top
    
    
    def rat180(self, x):
        x = np.fliplr(x)
        x = np.flipud(x)
        return x
        
        
        
    def backward(self, top_diff):
        self.top_diff = top_diff
        if(self.pad == 'same'):
            self.top_diff = np.zeros([top_diff.shape[0], 
                                    top_diff.shape[1], 
                                    top_diff.shape[2]+self.padding_h,
                                    top_diff.shape[3]+self.padding_w])
            self.top_diff[:,:,self.padding_h:,self.padding_w:] = top_diff
            
        elif(self.pad == 'valid'):
            self.top_diff = np.zeros([top_diff.shape[0], 
                                    top_diff.shape[1], 
                                    top_diff.shape[2]+2*self.padding_h,
                                    top_diff.shape[3]+2*self.padding_w])
            self.top_diff[:,:,self.padding_h:,self.padding_w:] = top_diff
        else:
            print('NOT IMPLETEMENT')
            exit()
        rat180_kernel = self.rat180(self.kernel)
        rat180_kernel = np.swapaxes(rat180_kernel, 0, 1)
        self.bottom_diff = conv(self.top_diff, rat180_kernel, self.stride)
        rat180_bottom = self.rat180(self.bottom)
        self.kernel_diff = conv(top_diff.swapaxes(0, 1), rat180_bottom.swapaxes(0, 1),
                                    self.stride) / len(self.bottom)
#        print(top_diff.shape)
#        print(self.top_diff.shape)
#        print(self.bottom_diff.shape)
#        print(self.padding_h)
#        print(rat180_kernel.shape)
        self.update()
        return self.bottom_diff
        
    def update(self, lr = 0.01):
        if(self.trainable == True):
            self.kernel -= lr * self.kernel_diff
    
        
        
        
# reshape layer   conv-->reshape-->inner_layer 
class reshape(layer):
    def __init__(self, name = 'reshape'):
        layer.__init__(self,name)
        self.trainable = False
        
    def forward(self, bottom):
        self.bottom_shape = bottom.shape
        return bottom.reshape(bottom.shape[0], -1)
    
    def backward(self, top_diff):
        return top_diff.reshape(self.bottom_shape)
    
    def update(self, lr = 0.001):
        pass
        
# Net
class Net:
    def __init__(self):
        self.layers = []
        
    def size(self):
        return   len(self.layers)  
    
    def forward(self, train_batch):
        assert(len(self.layers) > 0)
        data = train_batch
        for layer in self.layers:
            data = layer.forward(data)
        self.output = data
        return self.output
                
    def backward(self, loss_diff):
        diff = loss_diff
        for layer in self.layers[::-1]:
            diff = layer.backward(diff)
        return diff
    
    def add(self, layer):
        self.layers.append(layer)
        
    def fit_one_batch(self, train_batch, labels_batch, lr = 0.0001):
        self.lr = lr
        self.input = train_batch
        output = self.forward(train_batch)
        loss = mean_square_error(output, labels_batch)
        print('loss:',loss)
        diff = output - labels_batch
        self.backward(diff)
    
    def fit(self, train_data, labels, epoch = 1, batch_size = 32, lr = 0.0001):
        assert(len(train_data) == len(labels))
        num_batch = len(train_data) // batch_size
        for e in range(epoch):
            for batch in range(num_batch):
                self.fit_one_batch(train_data[batch * batch_size:batch * batch_size + batch_size],
                                   labels[batch * batch_size:batch * batch_size + batch_size],
                                   lr)
    
    
    def predict(self, data):
        return self.forward(data)
    



    
if __name__=='__main__':
    net = Net()
   
    net.add(conv_layer(16, 1, (2, 2)))
    net.add(activation('relu'))
    net.add(conv_layer(32, 1, (2, 2)))
    net.add(activation('relu'))
    net.add(reshape())
    fc = inner_product(100)
    
    net.add(fc)
    net.add(activation('relu'))
    net.add(inner_product(1))
    net.add(activation('sigmoid'))
    
    data = np.zeros([1000, 3, 32, 32])
    data[0:500] = np.random.randn(500, 3, 32, 32)
    data[500:]  = np.random.uniform(size = [500, 3, 32, 32]) - 0.5
    labels = np.zeros([1000,1])
    labels[500:] = 1
    idx = np.random.permutation(1000)
    net.fit(data[idx], labels[idx], 2, 8, 0.01)
    

            
            
            
            
            
            
            
            
            
