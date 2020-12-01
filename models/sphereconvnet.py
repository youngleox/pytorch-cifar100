"""SphereConvnet in pytorch (WIP)


orthogonal constraint is not implemented

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

'''
functions from original repo with tensorflow

def _get_filter_norm(self, filt):
    eps = 1e-4
    return tf.sqrt(tf.reduce_sum(filt*filt, [0, 1, 2], keep_dims=True)+eps)

def _get_input_norm(self, bottom, ksize, stride, pad):
    eps = 1e-4
    shape = [ksize, ksize, bottom.get_shape()[3], 1]
    filt = tf.ones(shape)
    input_norm = tf.sqrt(tf.nn.conv2d(bottom*bottom, filt, [1,stride,stride,1], padding=pad)+eps)
    return input_norm    

def _add_orthogonal_constraint(self, filt, n_filt):
        
    filt = tf.reshape(filt, [-1, n_filt])
    inner_pro = tf.matmul(tf.transpose(filt), filt)

    loss = 2e-4*tf.nn.l2_loss(inner_pro-tf.eye(n_filt))
    tf.add_to_collection('orth_constraint', loss)
'''
eps = 1e-4

def neuron_norm(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.norm(dim=1) + eps
    else:
        print("shouldn't happen")
        return x.abs()

class SphereConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=1, padding=0, bias=False, bn=False, 
                affine=True, momentum=0.1, relu=False, pad='SAME', 
                norm='cosine', reg=False, orth=False, w_norm=False):
        super(SphereConvLayer, self).__init__()
        print("in channel: {}, out channel: {}, norm: {}, bn: {}, relu: {}".format(in_channels,out_channels,norm,bn,relu))
        self.bn_flag = bn
        self.relu = relu
        self.norm = norm
        self.reg = reg
        self.orth = orth
        self.w_norm = w_norm

        if self.bn_flag:
            self.bn = nn.BatchNorm2d(out_channels, affine=affine,momentum=momentum)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        #create a all-ones convolution to compute input norm
        self.fake_conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.fake_conv.weight.requires_grad = False
        weights = torch.ones_like(self.fake_conv.weight)
        self.fake_conv.weight.data = weights

    def forward(self, x):
        # if len(x[torch.isnan(x)]) >0:
        #     print("nan x")
        #     print([torch.isnan(x)])
        conv = self.conv(x)        
        # if len(conv[torch.isnan(conv)]) >0:
        #     print("nan conv")
        #     print(conv[torch.isnan(conv)])
        with torch.no_grad():
            xnorm = torch.sqrt(self.fake_conv(x**2.0)+eps) + eps
            wnorm = neuron_norm(self.conv.weight.data+eps)
            wnorm = wnorm.view(1,wnorm.shape[0],1,1)
        #print("weight shape: {}, conv shape: {}, wnorm shape: {}, xnorm shape: {}".format(self.conv.weight.data.shape,conv.shape,wnorm.shape,xnorm.shape))
        if self.norm == 'linear':
            conv = conv/xnorm
            conv = conv/wnorm
            conv = -0.63662*torch.acos(conv)+1.0
        elif self.norm == 'cosine':
            conv = conv/xnorm
            # if len(conv[torch.isnan(conv)]) >0:
            #     print("nan conv xnorm")
            #     print(conv[torch.isnan(conv)])
            conv = conv/wnorm
            # if len(conv[torch.isnan(conv)]) >0:
            #     print("nan conv wnorm")
            #     print(conv[torch.isnan(conv)])
        elif self.norm == 'sigmoid':
            k_value = 0.3
            constant_coeff = (1 + numpy.exp(-numpy.pi/(2*k_value)))/(1 - numpy.exp(-numpy.pi/(2*k_value)))
            conv = conv/xnorm
            conv = conv/wnorm
            conv = constant_coeff*(1-torch.exp(torch.acos(conv)/k_value-numpy.pi/(2*k_value)))/(1+torch.exp(torch.acos(conv)/k_value-numpy.pi/(2*k_value)))
        # if len(conv[torch.isnan(conv)]) >0:
        #     print("nan")
        #     print(conv[torch.isnan(conv)])
        conv[torch.isnan(conv)] = 0
        if self.bn_flag:
            conv = self.bn(conv)
        if self.relu:
            conv = nn.ReLU(inplace=True)(conv)
        return conv

class BasicBlock(nn.Module):
    """Basic Block with sphere convolution layer for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, affine=True,
                    bn=True,bias=False,res=True,scale=1.0,momentum=0.1,
                    norm='linear'):
        super().__init__()
        self.scale = scale
        self.bn = bn
        self.res = res
        print('non linearity scale: {}'.format(scale))
        #residual function
        if self.bn:
            self.conv1 = nn.Sequential(
                SphereConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, 
                                bias=bias,bn=False,relu=False,norm=norm),
                nn.BatchNorm2d(out_channels, affine=affine,momentum=momentum)
            )
            self.conv2 = nn.Sequential(
                SphereConvLayer(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, 
                                bias=bias,bn=False,relu=False,norm=norm),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion, affine=affine,momentum=momentum)
            )

            #shortcut
            self.shortcut = nn.Sequential()

            #the shortcut output dimension is not the same with residual function
            #use 1*1 convolution to match the dimension
            if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
                self.shortcut = nn.Sequential(
                    SphereConvLayer(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, 
                                    bias=bias,bn=False,relu=False,norm='none'),
                    nn.BatchNorm2d(out_channels * BasicBlock.expansion, affine=affine,momentum=momentum)
                )
        else:
            print("batch norm disabled")
            self.conv1 = nn.Sequential(
                SphereConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, 
                                bias=bias,bn=False,relu=False,norm=norm)
            )
            self.conv2 = nn.Sequential(
                SphereConvLayer(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, 
                                bias=bias,bn=False,relu=False,norm=norm)
            )

            #shortcut
            self.shortcut = nn.Sequential()

            #the shortcut output dimension is not the same with residual function
            #use 1*1 convolution to match the dimension
            if (stride != 1 or in_channels != BasicBlock.expansion * out_channels) and self.res:
                self.shortcut = nn.Sequential(
                    SphereConvLayer(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, 
                                    bias=bias,bn=False,relu=False,norm='none')
                    #nn.BatchNorm2d(out_channels * BasicBlock.expansion, affine=affine)
                )

    def forward(self, x):
        if self.res:
            residual = self.shortcut(x)
            output = self.conv1(x)
            output = nn.ReLU(inplace=True)(output*self.scale)
            output = self.conv2(output)
            #print('output shape: {}, residual shape: {}'.format(output.shape, residual.shape))
            output = nn.ReLU(inplace=True)(output*self.scale + residual)
            return output 
        else:
            output = self.conv1(x)
            output = nn.ReLU(inplace=True)(output*self.scale)
            output = self.conv2(output)
            output = nn.ReLU(inplace=True)(output*self.scale)
            return output 

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, affine=True,bn=True,bias=False,res=True,scale=1.0,momentum=0.1):
        super().__init__()
        self.scale = scale
        self.bn = bn
        self.res = res
        if self.bn:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
                nn.BatchNorm2d(out_channels, affine=affine,momentum=momentum),
                #nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=bias),
                nn.BatchNorm2d(out_channels, affine=affine,momentum=momentum),
                #nn.ReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=bias),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion, affine=affine,momentum=momentum)
            )

            self.shortcut = nn.Sequential()

            if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=bias),
                    nn.BatchNorm2d(out_channels * BottleNeck.expansion, affine=affine,momentum=momentum)
                )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
                #nn.BatchNorm2d(out_channels, affine=affine),
                #nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=bias),
                #nn.BatchNorm2d(out_channels, affine=affine),
                #nn.ReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=bias),
                #nn.BatchNorm2d(out_channels * BottleNeck.expansion, affine=affine)
            )

            self.shortcut = nn.Sequential()

            if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=bias)
                    #nn.BatchNorm2d(out_channels * BottleNeck.expansion, affine=affine)
                )
    def forward(self, x):
        if self.res:
            residual = self.shortcut(x)
            output = self.conv1(x)
            output = nn.ReLU(inplace=True)(output*self.scale)
            output = self.conv2(output)
            output = nn.ReLU(inplace=True)(output*self.scale)
            output = self.conv3(output)
            output = nn.ReLU(inplace=True)(output*self.scale)
            return output + residual
        else:
            output = self.conv1(x)
            output = nn.ReLU(inplace=True)(output*self.scale)
            output = self.conv2(output)
            output = nn.ReLU(inplace=True)(output*self.scale)
            output = self.conv3(output)
            output = nn.ReLU(inplace=True)(output*self.scale)
            return output 

def SRELU(x):
    return F.ReLU(x*1.4142)

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100,affine=True,bn=True,bias=False,nl=nn.ReLU):
        #self.nl = SRELU
        super().__init__()
        self.out_clamp = False
        print("output clamping: {}".format(self.out_clamp))
        self.momentum = 0.1
        self.out_nl = nn.ReLU6(inplace=True)
        self.out_scale = 1.0
        self.out_bias = 0.0
        scale = 1.0 #1.0/1.414
        res = True #True #True #False
        affine = True # False #True
        bn = True #False
        bias = False
        self.out_bn_flag = False #False #True

        vanilla = False
        
        if vanilla:
            print("vanilla")
            self.momentum = 0.1
            self.out_scale = 1.0
            self.out_bias = 0.0
            scale = 1.0 #1.0/1.414
            res = True #True #True #False
            affine = True # False #True
            bn = True #False
            bias = False
            self.out_bn_flag = False #True
        if not res:
            print("no res!")
        if not bn:
            print("no bn!")
        if not bias:
            print("no bias!")
        if not affine:
            print("no affine!")
        width = 1
        self.in_channels = 64*width
        
        if bn:
            self.conv1 = nn.Sequential(
                SphereConvLayer(3, 64*width, kernel_size=3, padding=1, bias=bias,relu=False,bn=False,norm='linear'),
                nn.BatchNorm2d(64*width, affine=affine),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(
                SphereConvLayer(3, 64*width, kernel_size=3, padding=1, bias=bias,relu=False,bn=False,norm='linear'),
                #nn.BatchNorm2d(64, affine=affine),
                nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64*width, num_block[0], 1, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=self.momentum,norm='linear')
        self.conv3_x = self._make_layer(block, 128*width, num_block[1], 2, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=self.momentum,norm='linear')
        self.conv4_x = self._make_layer(block, 256*width, num_block[2], 2, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=self.momentum,norm='linear')
        self.conv5_x = self._make_layer(block, 512*width, num_block[3], 2, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=self.momentum,norm='linear')
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 *width * block.expansion, num_classes,bias=True) # Yang removed bias
        #self.out_bn = nn.BatchNorm1d(num_classes, affine=affine,momentum=self.momentum)
    def _make_layer(self, block, out_channels, num_blocks, stride, affine=True,bn=True,bias=True,res=True,scale=1.0,momentum=0.1,norm='linear'):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, affine=affine,bn=bn,bias=bias,res=res,scale=scale,momentum=momentum,norm=norm))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output) #*self.out_scale
        #Yang output scaling
        if self.out_bn_flag:
            output = self.out_bn(output)

        return output *self.out_scale + self.out_bias

class SphereConvNet(nn.Module):

    def __init__(self, block=None, num_block=3, num_classes=100,affine=True,bn=True,bias=False,nl=nn.ReLU):
        #self.nl = SRELU
        super().__init__()

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).cuda() # mean and std using cifar10
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
        relu = True
        norm = 'none' #True
        n_layer = 3

        # Conv1 layer
        conv1 = []
        channels = 128
        conv1.append(SphereConvLayer(3, channels, kernel_size=3, stride=1, padding=1,
                                    bn=bn,relu=relu,norm=norm))
        for i in range(n_layer-1):
            conv1.append(SphereConvLayer(channels, channels, kernel_size=3, stride=1, padding=1,
                                    bn=bn,relu=relu,norm=norm))
        self.conv1 = nn.Sequential(*conv1)
        self.maxpool1 = nn.MaxPool2d(2,stride=2)
        
        conv2 = []
        channels = 192
        conv2.append(SphereConvLayer(128, channels, kernel_size=3, stride=1, padding=1,
                                    bn=bn,relu=relu,norm=norm))
        for i in range(n_layer-1):
            conv2.append(SphereConvLayer(channels, channels, kernel_size=3, stride=1, padding=1,
                                    bn=bn,relu=relu,norm=norm))
        self.conv2 = nn.Sequential(*conv2)
        self.maxpool2 = nn.MaxPool2d(2,stride=2)

        conv3 = []
        channels = 256
        conv3.append(SphereConvLayer(192, channels, kernel_size=3, stride=1, padding=1,
                                    bn=bn,relu=relu,norm=norm))
        for i in range(n_layer-1):
            conv3.append(SphereConvLayer(channels, channels, kernel_size=3, stride=1, padding=1,
                                    bn=bn,relu=relu,norm=norm))
        self.conv3 = nn.Sequential(*conv3)
        self.maxpool3 = nn.MaxPool2d(2,stride=2)

        self.conv4 = SphereConvLayer(channels, channels, kernel_size=4, stride=1, padding=0,
                                    bn=False,relu=False,norm=norm)
        
        self.conv5 = SphereConvLayer(channels, num_classes, kernel_size=1, stride=1, padding=0,
                                    bn=False,relu=False,norm='none')
        
    def forward(self, x):
        #print(x.shape)
        #output = x
        #output = ((x * self.std.view(1,3,1,1) + self.mean.view(1,3,1,1)) * 255.0 - 127.5) / 128.0 # normalization found in the paper
        #output = ((x * 0.2 + 0.5) * 255.0 - 127.5) / 128.0
        output = self.conv1(x)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.maxpool3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = output.view(output.size(0), -1)
        return output

class SphereResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100,affine=True,bn=True,bias=False,nl=nn.ReLU):
        #self.nl = SRELU
        super().__init__()
        self.out_clamp = False
        print("output clamping: {}".format(self.out_clamp))
        self.momentum = 0.1
        self.out_nl = nn.ReLU6(inplace=True)
        self.out_scale = 1.0
        self.out_bias = 0.0
        scale = 1.0 #1.0/1.414
        res = True #True #True #False
        affine = True # False #True
        bn = True #False
        bias = False
        self.out_bn_flag = False #False #True

        vanilla = False
        
        if vanilla:
            print("vanilla")
            self.momentum = 0.1
            self.out_scale = 1.0
            self.out_bias = 0.0
            scale = 1.0 #1.0/1.414
            res = True #True #True #False
            affine = True # False #True
            bn = True #False
            bias = False
            self.out_bn_flag = False #True
        if not res:
            print("no res!")
        if not bn:
            print("no bn!")
        if not bias:
            print("no bias!")
        if not affine:
            print("no affine!")
        width = 1
        self.in_channels = 96
        
        if bn:
            self.conv1 = nn.Sequential(
                SphereConvLayer(3, 96, kernel_size=3, padding=1, bias=bias,
                                relu=False,bn=False,norm='linear'),
                nn.BatchNorm2d(96*width, affine=affine),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(
                SphereConvLayer(3, 96, kernel_size=3, padding=1, bias=bias,norm='linear'),
                #nn.BatchNorm2d(64, affine=affine),
                nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 96, num_block[0], 1, affine=affine,bn=bn,bias=bias,
                                        res=res,scale=scale,momentum=self.momentum,norm='linear')
        self.conv3_x = self._make_layer(block, 192, num_block[1], 2, affine=affine,bn=bn,bias=bias,
                                        res=res,scale=scale,momentum=self.momentum,norm='linear')
        self.conv4_x = self._make_layer(block, 384, num_block[2], 2, affine=affine,bn=bn,bias=bias,
                                        res=res,scale=scale,momentum=self.momentum,norm='linear')
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SphereConvLayer(384, num_classes, kernel_size=1, padding=0, 
                                    bias=bias,relu=False,bn=False,norm='none') 
        #self.out_bn = nn.BatchNorm1d(num_classes, affine=affine,momentum=self.momentum)
    def _make_layer(self, block, out_channels, num_blocks, stride, 
                    affine=True,bn=True,bias=True,res=True,scale=1.0,
                    momentum=0.1,norm='cosine'):

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, affine=affine,
                                bn=bn,bias=bias,res=res,scale=scale,momentum=momentum,
                                norm=norm))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)

        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        
        output = self.fc(output) 
        output = output.view(output.size(0), -1)

        return output

def sphereconvnet(num_classes=100):
    """ return a ResNet 18 object
    """
    return SphereConvNet(num_classes=num_classes)

def sphereresnet32(num_classes=100):
    """ return a ResNet 32 object
    """
    return SphereResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)

def resnet18(num_classes=100):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)

def resnet34(num_classes=100):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=num_classes)

def resnet50(num_classes=100):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3],num_classes=num_classes)

def resnet101(num_classes=100):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=num_classes)

def resnet152(num_classes=100):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])



