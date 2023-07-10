import torch
from torch import nn
import torch.nn.functional as F
import sys
from typing import List

import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    """

    def __init__(self, 
                num_features : int, 
                eps : float = 1e-5):
        super().__init__()

        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class Bottleneck(nn.Module):
    def __init__(self, 
                in_cha : int, 
                neck_cha : int, 
                out_cha : int, 
                stride : int, 
                has_bias : bool =False):
        super(Bottleneck, self).__init__()

        self.downsample = None
        if in_cha!= out_cha or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, kernel_size=1, stride=stride, bias=has_bias),
                FrozenBatchNorm2d(out_cha),)

        self.conv1 = nn.Conv2d(in_cha, neck_cha, kernel_size=1, stride=1, bias=has_bias)
        
        self.bn1 = FrozenBatchNorm2d(neck_cha)

        self.conv2 = nn.Conv2d(neck_cha, neck_cha, kernel_size=3, stride=stride, padding=1, bias=has_bias)
        self.bn2 = FrozenBatchNorm2d(neck_cha)

        self.conv3 = nn.Conv2d(neck_cha, out_cha, kernel_size=1, bias=has_bias)
        self.bn3 = FrozenBatchNorm2d(out_cha)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = F.relu_(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, 
                 freeze_at,
                 num_classes = 6,
                 has_bias = False):
        super(ResNet50, self).__init__()

        self.has_bias = has_bias
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=has_bias)
        self.bn1 = FrozenBatchNorm2d(64)

        block_counts = [3, 4, 6, 3]
        bottleneck_channels_list = [64, 128, 256, 512]
        out_channels_list = [256, 512, 1024, 2048]
        stride_list = [1, 2, 2, 2]
        in_channels = 64

        self.layer1 = self._make_layer(block_counts[0], 64, bottleneck_channels_list[0], out_channels_list[0], stride_list[0])
        self.layer2 = self._make_layer(block_counts[1], out_channels_list[0], bottleneck_channels_list[1], out_channels_list[1], stride_list[1])
        self.layer3 = self._make_layer(block_counts[2], out_channels_list[1], bottleneck_channels_list[2], out_channels_list[2], stride_list[2])
        self.layer4 = self._make_layer(block_counts[3], out_channels_list[2], bottleneck_channels_list[3], out_channels_list[3], stride_list[3])

        self.extra_conv1 = nn.Conv2d(2048, 256, 3, padding=1)
        self.extra_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.extra_conv3 = nn.Conv2d(256, 6, 3, padding=1)
        self.num_classes = num_classes

        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_normal_(l.weight, mode='fan_out')
                if self.has_bias:
                    nn.init.constant_(l.bias, 0)

        self._freeze_backbone(freeze_at)

    def _make_layer(self, 
                    num_blocks : int, 
                    in_channels : int, 
                    bottleneck_channels : int, 
                    out_channels : int, 
                    stride : int):

        layers = []
        for _ in range(num_blocks):
            layers.append(Bottleneck(in_channels, bottleneck_channels, out_channels, stride, self.has_bias))
            stride = 1
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _freeze_backbone(self, 
                        freeze_at : int):

        if freeze_at < 0:
            return
        if freeze_at >= 1:
            for p in self.conv1.parameters():
                p.requires_grad = False
        if freeze_at >= 2:
            for p in self.layer1.parameters():
                p.requires_grad = False
        if freeze_at >= 3:
            print("Freeze too much layers! Only freeze the first 2 layers.")

    def forward(self, x, label=None, size=None):

        if size is None:
            size = x.size()[2:] # (H, W)

        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.extra_conv1(x)
        x = self.extra_conv2(x)
        x = self.extra_conv3(x)

        logit = self.fc(x) 
        
        if label is None:
            return logit
        else:
            cam = self.cam_normalize(x.detach(), size, label)
            return logit, cam

    def fc(self, x):
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, self.num_classes)
        return x

    def cam_normalize(self, cam, size, label):  # vanila cam
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=True)
        cam /= F.adaptive_max_pool2d(cam, 1) + 1e-5 # normalize 
        cam = cam * label[:, :, None, None] # clean

        return cam

    def _initialize_weights(self):
        import math 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                    
    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups
    
def resnet50_base(pretrained=True, **kwargs):

    model = ResNet50(2, False)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
        print("model pretrained initialized")

    return model
