from bayesian_torch.layers.variational_layers.conv_variational import Conv2dReparameterization_Multivariate, Conv2dReparameterization
import torch.nn as nn
import torch.nn.functional as F

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
        
