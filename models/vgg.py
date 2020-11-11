"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=100,embedding=False):
        super().__init__()
        self.features = features

        self.embedding = embedding
        if embedding:
            print("using embedding layer")
            self.embed = nn.Linear(num_classes,num_classes,bias=False)
            self.embed.weight.requires_grad = False
            weights = torch.zeros_like(self.embed.weight)
            for i in range(num_class):
                weights[i,i] = (-1.0) ** i
            print(weights)
            self.embed.weight.data = weights
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
            )

        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
            )    

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        if self.embedding:
            output = self.embed(output)
        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11(num_classes=100):
    return VGG(make_layers(cfg['A'], batch_norm=False),num_classes=num_classes)

def vgg13(num_classes=100):
    return VGG(make_layers(cfg['B'], batch_norm=False),num_classes=num_classes)

def vgg16(num_classes=100):
    return VGG(make_layers(cfg['D'], batch_norm=False),num_classes=num_classes)

def vgg19(num_classes=100):
    return VGG(make_layers(cfg['E'], batch_norm=False),num_classes=num_classes)


def vgg11_bn(num_classes=100):
    return VGG(make_layers(cfg['A'], batch_norm=True),num_classes=num_classes)

def vgg13_bn(num_classes=100):
    return VGG(make_layers(cfg['B'], batch_norm=True),num_classes=num_classes)

def vgg16_bn(num_classes=100):
    return VGG(make_layers(cfg['D'], batch_norm=True),num_classes=num_classes)

def vgg19_bn(num_classes=100):
    return VGG(make_layers(cfg['E'], batch_norm=True),num_classes=num_classes)


