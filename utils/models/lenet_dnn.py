import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_dnn(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_dnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)   # 입력 채널 3 -> 출력 채널 6
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 계산:
        # 입력이 32x32일 경우,
        # conv1 → 28x28 → pool1 → 14x14
        # conv2 → 10x10 → pool2 → 5x5
        # 따라서 fc1 입력: 16 x 5 x 5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = LeNet5(num_classes=10)
    print(model)
    
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print("Output shape:", output.shape)
