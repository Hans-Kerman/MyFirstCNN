# coding=utf-8
import numpy as np
import struct
import os
import time

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        # 卷积层的初始化
        self.kernel_size = kernel_size  #卷积核大小k
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding  #边缘填充(的尺寸)
        self.stride = stride    #滑动步长
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):  # 参数初始化(生成卷积核)
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input # [N, C, H, W]
        """
        N:Batch_size样本数
        C:Channels通道数
        H:特征图高度
        W:特征图宽度
        """
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):         #idx n：第idxn张图片
            for idxc in range(self.channel_out):        #idx c：第idxc个通道
                for idxh in range(height_out):          #idx h：第idxh行像素
                    for idxw in range(width_out):       #idx w：第w列的像素
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置

                        # 1. 提取 input_patch (和之前一样)
                        start_h = idxh * self.stride
                        start_w = idxw * self.stride
                        input_patch = self.input_pad[idxn, :, start_h:start_h + self.kernel_size, start_w:start_w + self.kernel_size]

                        # 2. 提取 current_kernel (根据你的权重维度顺序)
                        # 假设你的权重维度是 (C_in, K, K, C_out)
                        current_kernel = self.weight[:, :, :, idxc]

                        self.output[idxn, idxc, idxh, idxw] = input_patch.flatten() @ current_kernel.flatten() + self.bias[idxc]
        return self.output
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):  # 最大池化层的初始化
        self.kernel_size = kernel_size
        self.stride = stride
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
			            # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        self.output[idxn, idxc, idxh, idxw] = _______________________
        return self.output

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):  # 扁平化层的初始化
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):  # 前向传播的计算
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out ')
        return self.output
