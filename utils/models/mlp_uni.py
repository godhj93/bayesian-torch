import torch
import torch.nn as nn
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
class MLP_uni(nn.Module):
    
    def __init__(self, input_size=28*28, hidden_size=100, output_size=10):
        super(MLP_uni, self).__init__()
        self.fc1 = LinearReparameterization(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = LinearReparameterization(hidden_size, hidden_size)
        self.fc3 = LinearReparameterization(hidden_size, output_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        
        kl = 0
        x = x.view(x.size(0), -1)
        x, kld = self.fc1(x)
        kl += kld
        x = self.bn1(x)
        x = self.relu(x)
        
        x, kld = self.fc2(x)
        kl += kld
        x = self.bn2(x)
        x = self.relu(x)
        
        x, kld = self.fc3(x)
        kl += kld
        
        return x, kl

if __name__ == "__main__":

    model = MLP_uni(input_size=28*28, hidden_size=100, output_size=10)
    print("\nModel created using class structure:")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 28, 28)
        y_hat, kld = model(x)
        
    print("Model summary:")
    print(model)
    print("Output shape:", y_hat, kld)
