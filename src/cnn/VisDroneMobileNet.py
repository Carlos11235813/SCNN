import torch
import torch.nn as nn
from torchvision import models
from src.cnn.FitParentClass import FitParentClass

class VisDroneMobileNet(FitParentClass):
    def __init__(self, B_boxes: int = 1, C: int = 10):

        super().__init__()
        self.C = C
        self.B = B_boxes

        out_channels = (self.B * 5) + self.C

        base_model = models.mobilenet_v3_small(weights=None)

        self.backbone = base_model.features

        self.detection_head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.detection_head(x)

        x = x.permute(0, 2, 3, 1)

        return x