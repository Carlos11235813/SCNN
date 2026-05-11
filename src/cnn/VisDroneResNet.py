import torch
import torch.nn as nn
import logging
from src.cnn.FitParentClass import FitParentClass

logger = logging.getLogger(__name__)

class ResBlock(nn.Module):
    """
    A residual block with two convolutional layers and a skip connection

    :param channels: Number of channels in the first convolutional layer.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) module for multiscale feature extraction

    :param channels: Number of channels in the first convolutional layer.
    :param out_channels: Number of output channels.
    :param k: Kernel size and padding size in MaxPool.
    """
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = nn.Conv2d(in_channels, c_, 1)
        self.cv2 = nn.Conv2d(c_ * 4, out_channels, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class VisDroneResNet(FitParentClass):
    """
    A ResNet-based architecture for VisDrone object detection.

    :param in_channels: Number of channels in the input image.
    :param boxes_per_cell: Number of boxes that model outputs per cell.
    :param dtype: Data type that model will use.
    :param num_classes: Number of classes in classification task.
    """
    def __init__(self,
                 num_classes: int = 10,
                 boxes_per_cell: int = 1,
                 dtype=torch.float32,
                 in_channels: int = 3):
        logger.info(f"VisDroneResNet object initialization")
        super().__init__()
        # Number of output channels per grid cell: 1 (obj) + 4 (bbox) + 10 (classes) = 15
        self.out_channels_per_cell = (1 + 4)*boxes_per_cell + num_classes
        self.dtype = dtype
        # Backbone: grid reduction by 8 times
        self.model = nn.Sequential(
            # Input: [Batch, 3, H, W]
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            ResBlock(128),
            
            SPPF(128, 256), # Increase the number of channels for better feature representation
            
            # Detection head: 1x1 convolution mapping channels to the output format
            nn.Conv2d(256, self.out_channels_per_cell, kernel_size=1)
        )
        self.to(dtype=self.dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output from convolution: [Batch, 15, S, S]
        out = self.model(x)
        
        # Changing the order of dimensions to: [Batch, S, S, 15]
        out = out.permute(0, 2, 3, 1).contiguous()
        
        return out
