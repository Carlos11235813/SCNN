import torch
import torch.nn as nn
import logging
from src.cnn.FitParentClass import FitParentClass

logger = logging.getLogger(__name__)

class VisDroneCNN(FitParentClass):
    def __init__(self, B_boxes: int = 1, C: int = 10, dtype=torch.float32):
        """
        B_boxes: number of bounding boxes per grid cell (most models only use 1)
        C: number of classes (here 10)
        """
        logger.info(f"VisDroneCNN object initialization")
        super(VisDroneCNN, self).__init__()
        self.B_boxes = B_boxes
        self.C = C
        self.dtype=dtype
        
        self.output_dim = self.B_boxes * 5 + self.C

        # Feature Extractor (Backbone)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )

        # Detection Head
        self.detection_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, self.output_dim, kernel_size=1)
        )
        self.to(dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.detection_head(x)
        
        # Changing dimensions to fit [B, S, S, 15]
        x = x.permute(0, 2, 3, 1) 
        
        return x