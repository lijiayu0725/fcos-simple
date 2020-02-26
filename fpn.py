import torch.nn as nn
import torch.nn.functional as F

from resnet import ResNet


class FPN(nn.Module):
    'Feature Pyramid Network - https://arxiv.org/abs/1612.03144'

    def __init__(self, features):
        super().__init__()


        self.features = features

        channels = [512, 1024, 2048]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.apply(init_layer)

        self.features.initialize()
        self.features.bn1.eval()
        self.features.layer1.eval()
        for m in [self.features.conv1, self.features.bn1, self.features.layer1]:
            for param in m.parameters():
                param.requires_grad = False

        for m in self.parameters():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        c3, c4, c5 = self.features(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, scale_factor=2) + p3

        p6 = self.pyramid6(p5)
        p7 = self.pyramid7(F.relu(p6))

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return p3, p4, p5, p6, p7


def ResNet50FPN(state_dict_path='/Users/nick/.cache/torch/checkpoints/resnet50-19c8e357.pth'):
    return FPN(ResNet(layers=[3, 4, 6, 3], outputs=[3, 4, 5], state_dict_path=state_dict_path))


if __name__ == '__main__':
    net = ResNet50FPN()
    net.initialize()
    from data import DataIterator

    dataiter = DataIterator()
    net.initialize()
    i = 0
    for data, target in dataiter:
        i += 1
        if i == 5:
            break
        y = net(data)
        for item in y:
            print(item.shape, end=' ')
        print()
