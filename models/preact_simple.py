'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActSimple(nn.Module):

    def __init__(self, planes,ratio):
        super().__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.conv = nn.Conv2d(planes, planes, kernel_size = 3, stride=1, padding=1, bias=False)
        self.ratio = ratio

    def forward(self, x):
        return x + self.conv(F.relu(self.bn(x)))/self.ratio
        # return x + F.relu(self.bn(self.conv(x)))

#one of these at the end of every group
class Transition(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.conv = nn.Conv2d(planes, planes, kernel_size = 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = torch.cat((x,self.conv(F.relu(self.bn(x)))),dim=1) #double planes without breaking residuality
        return F.avg_pool2d(x,2)


class SimpleResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.group1 = self._make_group(64, num_blocks[0])
        self.group2 = self._make_group(128, num_blocks[1])
        self.group3 = self._make_group(256, num_blocks[2])
        self.group4 = self._make_group(512, num_blocks[3],False)
        self.linear = nn.Linear(512, num_classes)

    #increases dimensionality and pools without using any parameters
    #used to avoid having to make weird group convs 
    def _make_group(self, planes, num_blocks, transition=True):

        layers = []
        
        for i in range(num_blocks):
            layers.append(PreActSimple(planes,num_blocks))

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

# def PreActResNet34():
    # return PreActResNet(PreActBlock, [3,4,6,3])

# def PreActResNet50():
    # return PreActResNet(PreActBottleneck, [3,4,6,3])

# def PreActResNet101():
    # return PreActResNet(PreActBottleneck, [3,4,23,3])

# def PreActResNet152():
    # return PreActResNet(PreActBottleneck, [3,8,36,3])


# def test():
    # net = PreActResNet18()
    # y = net(Variable(torch.randn(1,3,32,32)))
    # print(y.size())

# test()
