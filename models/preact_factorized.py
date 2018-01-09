'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def transpose_channels(x,groups):
    N, C, H, W = x.size()
    x = x.view(N, C//groups, groups, H, W)
    x.permute(0, 2, 1, 3, 4)
    x = x.view(N, C, H, W)
    return x

class PreActFactorized(nn.Module):
    #ratio should be the square root of the number of layers
    #(we generally only count the ones in this "sequence"
    # since there's only ever like 4 sequences)
    def __init__(self, planes, groups, external_layers, kernel_size = 3, padding = 1):
        super().__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.conv = nn.Conv2d(planes, planes, kernel_size, padding=padding, groups = groups, bias=False)
        self.ratio = math.sqrt(external_layers/groups)*2
        self.groups = groups

    def forward(self, x):
        x = x + self.conv(F.relu(self.bn(x)))*self.ratio
        if self.groups==1:
            return x
        else:
            return transpose_channels(x,self.groups)

'''
    a residual group of factorized 1x1 convs with 3x1 and 1x3 depthwise included
    a group size of 2 is assumed, but the depth between convolutions is not.
    
    ratio computation is rather involved. Basically, the scaling of the weights
    should be 1/sqrt(number of inputs per neuron * number of residual layers total)
    pytorch doesn't know this though, so it just takes 1/sqrt(number of total inputs)
    
    so we need to divide it's original computation out, multiply by 2
    
    It is important to use the same depth and grouping throughout the entire
    network, since layers need to know the total layers used.
'''
class FactorizedModule(nn.Module):

    def __init__(self, planes, depth, external_layers):
        super().__init__()
        total_layers = external_layers * depth
        self.conv1 = nn.Sequential(*[PreActFactorized(planes,planes//2,total_layers, 1, 0) for i in range(depth//2)])
        self.convH = PreActFactorized(planes,planes,total_layers,(3,1),(1,0))
        self.conv2 = nn.Sequential(*[PreActFactorized(planes,planes//2,total_layers, 1, 0) for i in range(depth//2)])
        self.convV = PreActFactorized(planes,planes,total_layers,(1,3),(0,1))
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.convH(x)
        x = self.conv2(x)
        x = self.convV(x)
        return x 

#one of these at the end of every group
#This is the closest we can get to a "Do nothing" layer
#that exclusively increases dimensionality and pools
#uses max 512*10 params per invocation.
class Transition(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.conv = nn.Conv2d(planes, planes, kernel_size = 3, stride=1, padding=1, groups=planes, bias=False)

    def forward(self, x):
        x = torch.cat((x,self.conv(F.relu(self.bn(x)))),dim=1) #double planes without breaking residuality
        return F.avg_pool2d(x,2)


class FactorizedResNet(nn.Module):
    def __init__(self, num_blocks, group_size, num_classes=10):
        super().__init__()
        self.group_size = group_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.group1 = self._make_sequence(64, num_blocks[0])
        self.group2 = self._make_sequence(128, num_blocks[1])
        self.group3 = self._make_sequence(256, num_blocks[2])
        self.group4 = self._make_sequence(512, num_blocks[3],False)
        self.linear = nn.Linear(512, num_classes)

    
    def _make_sequence(self, planes, num_blocks, transition=True):

        layers = []
        
        # for i in range(num_blocks):
            # layers.append(PreActFactorized(planes,planes//self.group_size,num_blocks))

        for i in range(num_blocks):
            layers.append(FactorizedModule(planes,4,num_blocks))

        if transition:
            layers.append(Transition(planes))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def PreActResNet18():
    # return PreActResNet(PreActBlock, [2,2,2,2])

# def test():
    # net = PreActResNet18()
    # y = net(Variable(torch.randn(1,3,32,32)))
    # print(y.size())

# test()
