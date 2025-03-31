import torchvision

class ResNet18_dnn(torchvision.models.ResNet):
    """
    ResNet18 DNN 모델.
    
    Args:
        num_classes (int): 분류할 클래스 수.
    """
    def __init__(self, num_classes=1000):
        super(ResNet18_dnn, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        
    def forward(self, x):
        return super(ResNet18_dnn, self).forward(x)
    
    