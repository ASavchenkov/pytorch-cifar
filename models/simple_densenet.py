'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

#this block structure just makes more sense from a memory perspective
#it also does all of the concatenation too, so here's to hoping
#pytorch is memory and compute efficient when it comes to cat.
class Simple_Block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Simple_Block, self).__init__()
        self.selector = nn.Conv2d(in_planes, out_planes,1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv = nn.Conv2d(out_planes, out_planes,3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_planes)
    
    def forward(self, x):
        selected = F.relu(self.bn1(self.selector(x)))
        out = F.relu(self.bn2(self.conv(selected)))
        out = torch.cat([out,x],1)
        return out

#The Simple_DenseNet does not use any transition layers.
class Simple_DenseNet(nn.Module):
    def __init__(self, nblocks, growth_rate=12, num_classes=10):
        super(Simple_DenseNet, self).__init__()
        self.growth_rate = growth_rate
        block = Simple_Block

        #n_blocks is a list of numbers
        


            
        #this section really just makes our set well sized.
        num_planes = 2*growth_rate
        self.setup_layer = block(3,num_planes-3)
        
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate


        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)
    
    #this allows us to selectively l1 regularize selectors (except for biases)
    def get_selector_params(self):
        selectors = list()
        selectors.append(self.setup_layer.selector.weight)

        for block in [self.dense1,self.dense2, self.dense3,self.dense4]:
            for layer in block:
                selectors.append(layer.selector.weight)

        return selectors

    def forward(self, x):
        out = self.setup_layer(x)

        out = F.max_pool2d(self.dense1(out),2)
        out = F.max_pool2d(self.dense2(out),2)
        out = F.max_pool2d(self.dense3(out),2)
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test_densenet():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y)

# test_densenet()
