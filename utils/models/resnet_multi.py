from bayesian_torch.layers.variational_layers.conv_variational import Conv2dReparameterization_Multivariate
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.models.bayesian.resnet_variational import BasicBlock
from termcolor import colored
from torchvision import transforms
import torch

    
class BasicBlock_multi(BasicBlock):
 
    def __init__(self, in_planes, planes, stride=1, option='A', rank = 1):
        super(BasicBlock_multi, self).__init__(in_planes, planes, stride, option)
        self.conv1 = Conv2dReparameterization_Multivariate(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            rank = rank)
        self.conv2 = Conv2dReparameterization_Multivariate(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            rank = rank)
    
class ResNet_multivariate(nn.Module):
    
    def __init__(self, block, num_blocks, rank = 1, num_classes=10):
        super(ResNet_multivariate, self).__init__()
        
        self.rank = rank
        print(colored(f"Rank: {self.rank}", 'blue'))
        div = 1
        self.in_planes = 16//div # 16
        self.conv1 = Conv2dReparameterization_Multivariate(
            in_channels=3,
            out_channels=16//div, # 16
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            rank = 3*3*16*3 if not self.rank == 'full' else self.rank)
        self.bn1 = nn.BatchNorm2d(16//div) # 16
        self.layer1 = self._make_layer(block, 16//div, num_blocks[0], stride=1) # 16
        self.layer2 = self._make_layer(block, 32//div, num_blocks[1], stride=2) # 32
        self.layer3 = self._make_layer(block, 64//div, num_blocks[2], stride=2) # 64
        self.linear = nn.Linear(64//div, num_classes)
        print(colored(f"Linear layer is not variational", 'red'))
        
        print(f"The number of parameters: {(sum(p.numel() for p in self.parameters())):,}")        
        self.debug = False
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rank = self.rank))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        kl_sum = 0
        out, kl = self.conv1(x)
        kl_sum += kl
        out = self.bn1(out)
        out = F.relu(out)

        for l in self.layer1:
            out, kl = l(out)
            kl_sum += kl
        for l in self.layer2:
            out, kl = l(out)
            kl_sum += kl

        for l in self.layer3:
            out, kl = l(out)
            kl_sum += kl

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        # kl_sum += kl ## It must be enabled if the linear layer is variational
        
        return out, kl_sum
    

def resnet20_multi(rank = 1):
    return ResNet_multivariate(BasicBlock_multi, [3, 3, 3], rank = rank)


def resnet32_multi():
    return ResNet_multivariate(BasicBlock_multi, [5, 5, 5])


def resnet44_multi():
    return ResNet_multivariate(BasicBlock_multi, [7, 7, 7])


def resnet56_multi():
    return ResNet_multivariate(BasicBlock_multi, [9, 9, 9])


def resnet110_multi():
    return ResNet_multivariate(BasicBlock_multi, [18, 18, 18])
