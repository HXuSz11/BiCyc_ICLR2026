import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights


# ------------------------------------------------------------
# BasicBlock — shared by both variants
# ------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


# ------------------------------------------------------------
# ResNet‑18 backbone
# ------------------------------------------------------------
class ResNet18Backbone(nn.Module):
    """
    * `small_input=True`  → 3 × 3 conv, stride 1 (CIFAR/CUB scale).
    * `small_input=False` → 7 × 7 conv, stride 2 + 3 × 3 max‑pool (ImageNet).
    """

    def __init__(self,
                 block:       nn.Module = BasicBlock,
                 layers:      List[int] = [2, 2, 2, 2],
                 *,
                 feat_dim:    int  = 512,
                 small_input: bool = False,
                 pretrained:  bool = False):
        super().__init__()
        self.in_planes = 64
        self.feat_dim  = feat_dim
        self.small     = small_input

        # ---------- Stem ----------
        if self.small:
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.bn1   = nn.BatchNorm2d(64)
            self.pool  = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1   = nn.BatchNorm2d(64)
            self.pool  = nn.MaxPool2d(3, 2, 1)

        # ---------- Residual stages ----------
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # ---------- 1×1 bottleneck ----------
        self.bottleneck = nn.Conv2d(512, feat_dim, 1, bias=True)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)

        # ---------- Weight init ----------
        self._init_weights()

        # ---------- Optional ImageNet weights ----------
        if pretrained:
            self._load_imagenet_weights()

    # ----------------- helpers -----------------
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def _load_imagenet_weights(self):
        src = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
        own = self.state_dict()

        # Skip FC layer, map matching keys, ignore missing ones when small_input=True
        for k, v in src.items():
            if k.startswith('fc.'):
                continue
            if k in own:
                own[k] = v
        self.load_state_dict(own, strict=False)
        print('✔ ImageNet pretrained weights loaded.')

    # ----------------- public -----------------
    def get_feat_size(self) -> int:
        return self.feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bottleneck(x)
        x = self.avgpool(x).flatten(1)
        return x


# ------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------
def resnet18_cifar(*, feat_dim: int = 512, **kwargs) -> ResNet18Backbone:
    """Equivalent to the original CIFAR‑style network (always random init)."""
    return ResNet18Backbone(feat_dim=feat_dim,
                            small_input=True,
                            pretrained=False,
                            **kwargs)


def resnet18_imagenet(*,
                      feat_dim:  int = 512,
                      pretrained: bool = False,
                      **kwargs) -> ResNet18Backbone:
    """ImageNet‑style stem with optional pretrained weights."""
    return ResNet18Backbone(feat_dim=feat_dim,
                            small_input=False,
                            pretrained=pretrained,
                            **kwargs)


# ------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------
if __name__ == "__main__":
    # CIFAR / CUB configuration
    net_small = resnet18_cifar(feat_dim=512)
    print('CIFAR stem output:', net_small(torch.randn(1, 3, 64, 64)).shape)

    # ImageNet stem with pretrained weights
    net_large = resnet18_imagenet(pretrained=True)
    print('ImageNet stem output:', net_large(torch.randn(1, 3, 224, 224)).shape)
