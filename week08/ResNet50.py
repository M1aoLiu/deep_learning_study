import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        """

        Args:
            in_channel: 输入的通道
            out_channel: 第一次卷积后的输出通道
            stride:
            shortcut:
        """
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=1, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=out_channel * self.expansion)
        self.relu3 = nn.ReLU()
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, block, block_num, num_classes=1000):
        """

        Args:
            block: 堆叠的基本模块
            block_num: 每层block的个数
            num_classes: 分类个数
        """
        super(ResNet50, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = self.make_layer(block, 64, block_num[0], stride=1)
        self.layer3 = self.make_layer(block, 128, block_num[1], stride=2)
        self.layer4 = self.make_layer(block, 256, block_num[2], stride=2)
        self.layer5 = self.make_layer(block, 512, block_num[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

    def make_layer(self, block, channel, block_num, stride=1):
        """

        Args:
            block: 堆叠的基本模块
            channel: 每个stage中第一个conv的channel，为64，128，256，512
            block_num: 每层block的个数
            stride:

        Returns:

        """
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel,
                      stride=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.in_channel)
        )
        if stride != 1 or self.in_channel != channel * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion,
                          stride=stride, kernel_size=1, padding=0),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, stride, shortcut))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = ResNet50(block=BottleNeck, block_num=[3, 4, 6, 3], num_classes=1000)
    data = torch.randn(1, 3, 224, 224)
    output = model.forward(data)
    # print(output)
    print(model)
