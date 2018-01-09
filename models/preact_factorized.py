'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
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

    def __init__(self, planes, groups, ratio):
        super().__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.conv = nn.Conv2d(planes, planes, 3, padding=1, groups = groups, bias=False)
        self.ratio = ratio
        self.groups = groups

    def forward(self, x):
        x = x + self.conv(F.relu(self.bn(x)))/self.ratio
        if self.groups==1:
            return x
        else:
            return transpose_channels(x,self.groups)

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
    def __init__(self, num_blocks, groups, num_classes=10):
        super().__init__()
        self.groups = groups

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.group1 = self._make_sequence(64, num_blocks[0])
        self.group2 = self._make_sequence(128, num_blocks[1])
        self.group3 = self._make_sequence(256, num_blocks[2])
        self.group4 = self._make_sequence(512, num_blocks[3],False)
        self.linear = nn.Linear(512, num_classes)

    
    def _make_sequence(self, planes, num_blocks, transition=True):

        layers = []
        
        for i in range(num_blocks):
            layers.append(PreActFactorized(planes,self.groups,num_blocks))

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
