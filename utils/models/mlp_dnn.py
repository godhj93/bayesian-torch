import torch
import torch.nn as nn
import torchvision

class MLP_dnn(nn.Module):
    
    def __init__(self, input_size=28*28, hidden_size=100, output_size=10):
        super(MLP_dnn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x

if __name__ == "__main__":

    model = MLP_dnn(input_size=28*28, hidden_size=100, output_size=10)
    print("\nModel created :")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        
    print("Model summary:")
    print(model)
    print("Output shape:", output.shape)
