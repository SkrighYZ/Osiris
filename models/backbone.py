# Modified from https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm='gn', group_element=4, act='mish'):
        super(BasicBlock, self).__init__()
        norm_layer = nn.GroupNorm if norm == 'gn' else nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if norm == 'gn':
            gs = min([32, planes//group_element])
            self.norm1 = norm_layer(gs, planes)
        else:
            self.norm1 = norm_layer(planes)

        self.activation = nn.Mish(inplace=True) if act == 'mish' else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if norm == 'gn':
            gs = min([32, planes//group_element])
            self.norm2 = norm_layer(gs, planes)
        else:
            self.norm2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm='gn', group_element=4, act='mish'):

        super(ResNet, self).__init__()
        norm_layer = nn.GroupNorm if norm == 'gn' else nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.group_element = group_element
        self.norm = norm
        self.act = act

        self.inplanes = 64
        self.dilation = 1
        self.n_classes = num_classes
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        ## For CIFAR/Tiny-IN: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        
        if self.norm == 'gn':
            gs = min([32, self.inplanes//self.group_element])
            self.norm1 = norm_layer(gs, self.inplanes)
        else: 
            self.norm1 = norm_layer(self.inplanes)

        self.activation = nn.Mish(inplace=True) if act == 'mish' else nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.norm == 'gn':
                gs = min([32, planes*block.expansion//self.group_element])
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(gs, planes*block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, 
                            norm=self.norm, group_element=self.group_element, act=self.act))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm=self.norm, group_element=self.group_element, act=self.act))

        return nn.Sequential(*layers)
    
    def embed(self, x):
        x = self.conv1(x) 
        x = self.norm1(x)
        x1 = self.activation(x)     # (bsz, 64, 64, 64)

        x2 = self.layer1(x1)        # (bsz, 64, 64, 64)
        x3 = self.layer2(x2)        # (bsz, 128, 32, 32)
        x4 = self.layer3(x3)        # (bsz, 256, 16, 16)
        x5 = self.layer4(x4)        # (bsz, 512, 8, 8)

        x = self.avgpool(x5)
        x = x.reshape(x.size(0), -1)
        return x, [x1, x2, x3, x4, x5]

    def forward(self, x):
        x, _ = self.embed(x)
        return self.fc(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

if __name__ == "__main__":
    model = resnet18()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 11173962
    y = model(torch.randn(64, 3, 32, 32), 1)
    print(y.size(), model)

