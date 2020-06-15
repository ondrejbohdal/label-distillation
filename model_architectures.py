import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# layers for meta-learning with second-order gradients
# code inspired by https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py


class Linear_fw(nn.Linear):
    # forward input with fast weight link

    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            # weight.fast (fast weight) is the temporarily adapted weight
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels,
                                        kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None,
                               stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast,
                               stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        return out


class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features, device):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None
        self.device = device

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).to(device=self.device)
        running_var = torch.ones(x.data.size()[1]).to(device=self.device)
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast,
                               self.bias.fast, training=True, momentum=1)
        else:
            out = F.batch_norm(x, running_mean, running_var,
                               self.weight, self.bias, training=True, momentum=1)
        return out


# architectures for meta-learning with second-order gradients
# code inspired by https://github.com/SsnL/dataset-distillation/blob/master/networks/networks.py


class LeNetMeta(nn.Module):
    meta = False  # Default

    def __init__(self, args):
        super(LeNetMeta, self).__init__()
        self.args = args

        if self.args.target == "cifar10":
            self.in_channels = 3
            self.input_size = 32
            self.num_classes = 10
        elif self.args.target == "k49":
            self.in_channels = 1
            self.input_size = 28
            self.num_classes = 49
        else:
            self.in_channels = 1
            self.input_size = 28
            self.num_classes = 10

        if self.meta:
            self.conv1 = Conv2d_fw(
                self.in_channels, 6, 5, padding=2 if self.input_size == 28 else 0)
            self.conv2 = Conv2d_fw(6, 16, 5)
            self.fc1 = Linear_fw(16 * 5 * 5, 120)
            self.fc2 = Linear_fw(120, 84)
            self.fc3 = Linear_fw(84, self.num_classes)
        else:
            self.conv1 = nn.Conv2d(
                self.in_channels, 6, 5, padding=2 if self.input_size == 28 else 0)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexCifarNetMeta(nn.Module):
    meta = False  # Default

    def __init__(self, args):
        super(AlexCifarNetMeta, self).__init__()
        self.args = args

        if self.meta:
            self.features = nn.Sequential(
                Conv2d_fw(3, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
                Conv2d_fw(64, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.fc1 = Linear_fw(4096, 384)
            self.fc2 = Linear_fw(384, 192)
            if self.args.target == "cifar100":
                self.fc3 = Linear_fw(192, 100)
            else:
                self.fc3 = Linear_fw(192, 10)
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.fc1 = nn.Linear(4096, 384)
            self.fc2 = nn.Linear(384, 192)
            if self.args.target == "cifar100":
                self.fc3 = nn.Linear(192, 100)
            else:
                self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# code inspired by https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py


def conv3x3meta(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Conv2d_fw(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlockMeta(nn.Module):
    expansion = 1
    meta = False

    def __init__(self, inplanes, planes, stride=1, downsample=None, device=None):
        super(BasicBlockMeta, self).__init__()
        if self.meta:
            self.conv1 = conv3x3meta(inplanes, planes, stride)
            self.bn1 = BatchNorm2d_fw(planes, device)
            self.conv2 = conv3x3meta(planes, planes)
            self.bn2 = BatchNorm2d_fw(planes, device)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckMeta(nn.Module):
    expansion = 4
    meta = False

    def __init__(self, inplanes, planes, stride=1, downsample=None, device=None):
        super(BottleneckMeta, self).__init__()
        if self.meta:
            self.conv1 = Conv2d_fw(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = BatchNorm2d_fw(planes, device)
            self.conv2 = Conv2d_fw(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = BatchNorm2d_fw(planes, device)
            self.conv3 = Conv2d_fw(
                planes, planes * BottleneckMeta.expansion, kernel_size=1, bias=False)
            self.bn3 = BatchNorm2d_fw(
                planes * BottleneckMeta.expansion, device)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(
                planes, planes * BottleneckMeta.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * BottleneckMeta.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetMeta(nn.Module):
    meta = False

    def __init__(self, dataset, depth, num_classes, bottleneck=False, device=None):
        super(ResNetMeta, self).__init__()
        self.dataset = dataset
        self.device = device

        assert self.dataset.startswith(
            'cifar'), "ResNet currently only supports CIFAR"

        self.inplanes = 16
        print(bottleneck)
        if bottleneck is True:
            n = int((depth - 2) / 9)
            if self.meta:
                block = BottleneckMeta
            else:
                block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            if self.meta:
                block = BasicBlockMeta
            else:
                block = BasicBlock
        if self.meta:
            self.conv1 = Conv2d_fw(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = BatchNorm2d_fw(self.inplanes, self.device)
            self.fc = Linear_fw(64 * block.expansion, num_classes)
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.meta:
                downsample = nn.Sequential(
                    Conv2d_fw(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BatchNorm2d_fw(planes * block.expansion, self.device),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, self.device))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, device=self.device))

        return nn.Sequential(*layers)

    def forward(self, x, w=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# architectures for meta-learning with pseudo-gradient and ridge regression
# AdjustLayer inspired by https://github.com/bertinetto/r2d2/blob/master/fewshots/models/adjust.py


class AdjustLayer(nn.Module):
    def __init__(self, init_scale=1, device=None):
        super().__init__()
        self.scale = nn.Parameter(
            torch.FloatTensor([init_scale]).to(device=device))

    def forward(self):
        return self.scale


# architectures inspired by https://github.com/SsnL/dataset-distillation/blob/master/networks/networks.py


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        self.args = args
        if self.args.target == "cifar10":
            self.in_channels = 3
            self.input_size = 32
        else:
            self.in_channels = 1
            self.input_size = 28

        self.conv1 = nn.Conv2d(self.in_channels, 6, 5,
                               padding=2 if self.input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x, w=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # use ridge regression weights instead of the final layer
        if w is not None:
            x = x.matmul(w[:-1, :]) + w[-1, :]

        return x


class AlexCifarNet(nn.Module):
    def __init__(self, args):
        super(AlexCifarNet, self).__init__()
        self.args = args

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.fc1 = nn.Linear(4096, 384)
        self.fc2 = nn.Linear(384, 192)

    def forward(self, x, w=None):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # use ridge regression weights instead of the final layer
        if w is not None:
            x = x.matmul(w[:-1, :]) + w[-1, :]

        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, dataset, depth, num_classes, bottleneck=False):
        super(ResNet, self).__init__()
        self.dataset = dataset

        assert self.dataset.startswith(
            'cifar'), "ResNet currently only supports CIFAR"

        self.inplanes = 16
        print(bottleneck)
        if bottleneck is True:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, w=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # use ridge regression weights instead of the final layer
        if w is not None:
            x = x.matmul(w[:-1, :]) + w[-1, :]

        return x
