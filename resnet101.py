import torch
import torch.nn as nn
import torchvision.models as models

class ResNetAudio101(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetAudio101, self).__init__()
        self.resnet = models.resnet101(pretrained=True)  # Load the ResNet-101 model
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Modify the first convolutional layer to accept 1-channel input
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_classes)  # Modify the final fully connected layer to output the correct number of classes

    def forward(self, x):
        x = self.resnet(x)
        return x

print("ResNet-101 model initialized")
