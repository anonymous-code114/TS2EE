import torchvision.models as models
import torch
from models.mlp_head import MLPHead
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class DenseNetModel(torch.nn.Module):
    def __init__(self, channels=1, *args, **kwargs):
        super(DenseNetModel, self).__init__()
        if kwargs['name'] == 'densenet121':
            densenet = models.densenet121(pretrained=False)
        elif kwargs['name'] == 'densenet169':
            densenet = models.densenet169(pretrained=False)
        elif kwargs['name'] == 'densenet201':
            densenet = models.densenet201(pretrained=False)

        # Replace the first convolution layer
        densenet.features.conv0 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=2, padding=(3, 3), bias=False)

        # Reinitialize the weights for the new convolution layer
        nn.init.kaiming_normal_(densenet.features.conv0.weight, mode='fan_out', nonlinearity='relu')

        # Remove the final classification layer
        self.features = torch.nn.Sequential(*list(densenet.features)[:-1], densenet.features.norm5,
                                            nn.AdaptiveAvgPool2d((1, 1)), Flatten())
        # self.norm5 = densenet.features.norm5
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add the projection head
        self.projetion = MLPHead(in_channels=densenet.classifier.in_features, **kwargs['projection_head'])
        # print('projection head:', self.projetion)

    def forward(self, x):
        features = self.features(x)
        # features = self.norm5(features)
        # features = self.adaptive_pool(features)
        # print(out.shape)
        # features = torch.flatten(features, 1)
        return self.projetion(features)
