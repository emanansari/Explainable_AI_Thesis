import torch
import torch.nn as nn
import torchvision.models as models

class ResNetAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetAudio, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

print("ResNet model initialized")
