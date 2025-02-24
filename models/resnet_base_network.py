import torchvision.models as models
import torch
from models.mlp_head import MLPHead
import torch.nn as nn


class ResNet18(torch.nn.Module):
    def __init__(self, channels=1, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
            resnet.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=2, padding=(3, 3), bias=False)
        elif kwargs['name'] == 'resnet34':
            resnet = models.resnet34(pretrained=False)
            resnet.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=1, padding=(3, 3), bias=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)
            resnet.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
