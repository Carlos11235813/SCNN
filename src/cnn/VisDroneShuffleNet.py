import torch
import torch.nn as nn
from src.cnn.FitParentClass import FitParentClass

class DecoupledHead(nn.Module):
    # A decoupled head for separate classification and regression paths
    def __init__(self, in_channels: int, num_classes: int=10):
        super().__init__()
        # Combined preprocessing for better feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classification path
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1) # Result: 10 classes
        )
        
        # Regression path
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1 + 4, 1) # Objectness (1) + Box (4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        cls_score = self.cls_branch(x)
        reg_score = self.reg_branch(x)
        
        # Joining the tensors
        return torch.cat([reg_score, cls_score], dim=1)

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Shuffling channels for ShuffleNet"""
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # Re-shape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # Transposing group dimensions
    x = torch.transpose(x, 1, 2).contiguous()
    # Flattening back to original shape
    x = x.view(batch_size, -1, height, width)
    return x

class ShuffleUnit(nn.Module):
    # A single ShuffleNet unit that performs channel splitting, depthwise convolution, and channel shuffling
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        branch_features = out_channels // 2
        
        # Path for processing (Branch 2)
        self.branch2 = nn.Sequential(
            # 1x1 Conv (Pointwise)
            nn.Conv2d(branch_features if stride==1 else in_channels, 
                      branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            # 3x3 Depthwise Conv
            nn.Conv2d(branch_features, branch_features, 3, 
                      stride=stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            # 1x1 Conv (Pointwise)
            nn.Conv2d(branch_features, branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
        
        # Path for shortcut connection (Branch 1) - only if we are reducing dimensions (stride=2)
        if stride == 2:
            self.branch1 = nn.Sequential(
                # 3x3 Depthwise Conv
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, 
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                # 1x1 Conv
                nn.Conv2d(in_channels, branch_features, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()

    def forward(self, x):
        if self.stride == 1:
            # Split channels in half
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # Stride 2: both branches process the entire input
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            
        return channel_shuffle(out, 2)

class VisDroneShuffleNet(FitParentClass):
    # A ShuffleNet-based architecture for VisDrone object detection
    def __init__(self, num_classes=10, boxes_per_cell: int = 1, dtype=torch.float32, in_channels: int = 3):
        super().__init__()

        self.out_channels_per_cell = (1 + 4) * boxes_per_cell + num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = ShuffleUnit(24, 116, stride=2)
        
        self.stage2 = nn.Sequential(
            ShuffleUnit(116, 232, stride=2),
            ShuffleUnit(232, 232, stride=1),
            ShuffleUnit(232, 232, stride=1)
        )
        
        # Detection head
        self.head = DecoupledHead(232, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        
        out = self.head(x)
        return out.permute(0, 2, 3, 1).contiguous()
