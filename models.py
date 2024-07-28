from bayesian_torch.layers.variational_layers.conv_variational import Conv2dReparameterization_Multivariate, Conv2dReparameterization
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.models.bayesian.resnet_variational import BasicBlock
from termcolor import colored

# prior_mu = 0.0
# prior_sigma = 1.0
# posterior_mu_init = 0.0
# posterior_rho_init = -3.0

class LeNet(nn.Module):
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16*12*12, 10)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.pool(x)
        
        x = x.view(-1, 16*12*12)
        
        logit = self.fc1(x)
        
        return logit
    
# LeNet
class LeNet_BNN(nn.Module):
    
    def __init__(self):
        super(LeNet_BNN, self).__init__()
        self.conv1 = Conv2dReparameterization_Multivariate(1, 6, 3, 1)
        self.conv2 = Conv2dReparameterization_Multivariate(6, 16, 3, 1)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16*12*12, 10)
        
    def forward(self, x):
        kl_sum = 0
        
        x, kl = self.conv1(x)
        kl_sum += kl
        x = F.relu(x)
        
        x, kl = self.conv2(x)
        kl_sum += kl
        x = F.relu(x)
        
        x = self.pool(x)
        x = x.view(-1, 16*12*12)
        
        logit = self.fc1(x)
        
        return logit, kl_sum

class LeNet_BNN_uni(LeNet_BNN):
    
    def __init__(self):
        super(LeNet_BNN_uni, self).__init__()
        self.conv1 = Conv2dReparameterization(1, 6, 3, 1)
        self.conv2 = Conv2dReparameterization(6, 16, 3, 1)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16*12*12, 10)
        
        
class BasicBlock_multi(BasicBlock):
 
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_multi, self).__init__(in_planes, planes, stride, option)
        self.conv1 = Conv2dReparameterization_Multivariate(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.conv2 = Conv2dReparameterization_Multivariate(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)        
    
class ResNet_multivariate(nn.Module):
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_multivariate, self).__init__()
        
        div = 4
        print(colored(f"in planes is reduced by {div} times since the computation is heavy", 'red'))
        self.in_planes = 16//div # 16
        self.conv1 = Conv2dReparameterization_Multivariate(
            in_channels=3,
            out_channels=16//div, # 16
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(16//div) # 16
        self.layer1 = self._make_layer(block, 16//div, num_blocks[0], stride=1) # 16
        self.layer2 = self._make_layer(block, 32//div, num_blocks[1], stride=2) # 32
        self.layer3 = self._make_layer(block, 64//div, num_blocks[2], stride=2) # 64
        self.linear = nn.Linear(64//div, num_classes)
        print(colored(f"Linear layer is not variational", 'red'))
        
        # self.apply(_weights_init)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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

        kl_sum += kl
        return out, kl_sum
    

def resnet20_multi():
    return ResNet_multivariate(BasicBlock_multi, [3, 3, 3])


def resnet32_multi():
    return ResNet_multivariate(BasicBlock_multi, [5, 5, 5])


def resnet44_multi():
    return ResNet_multivariate(BasicBlock_multi, [7, 7, 7])


def resnet56_multi():
    return ResNet_multivariate(BasicBlock_multi, [9, 9, 9])


def resnet110_multi():
    return ResNet_multivariate(BasicBlock_multi, [18, 18, 18])



    
    
        
        
            