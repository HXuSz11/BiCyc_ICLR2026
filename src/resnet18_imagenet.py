from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import torchvision.models as models
from torchvision.models import ResNet18_Weights



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, pretrained=False,  feat_dim=512):
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64

        self.layer0_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer0_bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.feature_space_size = 512
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if pretrained:
            self._load_pretrained_weights()
        
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    # def _load_pretrained_weights(self):
    #     """Copy ImageNet weights from torchvision's ResNet-18."""
    #     src = models.resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
    #     dst = self.state_dict()  # own parameters

    #     # Keys that differ between the two models
    #     rename_map = {
    #         'layer0_conv1.weight':            'conv1.weight',
    #         'layer0_bn1.weight':              'bn1.weight',
    #         'layer0_bn1.bias':                'bn1.bias',
    #         'layer0_bn1.running_mean':        'bn1.running_mean',
    #         'layer0_bn1.running_var':         'bn1.running_var',
    #         'layer0_bn1.num_batches_tracked': 'bn1.num_batches_tracked',
    #     }

    #     # Copy renamed keys
    #     for dst_key, src_key in rename_map.items():
    #         dst[dst_key] = src[src_key]

    #     # Copy identical names (layer1-4 etc.)
    #     for k in dst.keys():
    #         if k in rename_map:       # already handled
    #             continue
    #         if k in src:
    #             dst[k] = src[k]

    #     self.load_state_dict(dst)
        # print("✓ Loaded ImageNet-pretrained weights.")
    
    def _load_pretrained_weights(self):
        url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        src = torch.hub.load_state_dict_from_url(url, progress=True)

        rename = {
            'conv1.weight':            'layer0_conv1.weight',
            'bn1.weight':              'layer0_bn1.weight',
            'bn1.bias':                'layer0_bn1.bias',
            'bn1.running_mean':        'layer0_bn1.running_mean',
            'bn1.running_var':         'layer0_bn1.running_var',
            'bn1.num_batches_tracked': 'layer0_bn1.num_batches_tracked',
        }
        state = self.state_dict()
        for k, v in src.items():
            if k.startswith('fc.'):
                continue
            state[rename.get(k, k)] = v
        self.load_state_dict(state, strict=False)
        print("✓ Loaded ImageNet-pretrained weights.")

    def forward(self, x):
        out = F.relu(self.layer0_bn1(self.layer0_conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out_features = out.view(out.size(0), -1)
 
        return out_features


def resnet18_imagenet(pretrained = False, **kwargs):
    return ResNetImageNet(BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)