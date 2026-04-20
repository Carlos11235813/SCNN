import torch
import torch.nn as nn

class VisDroneCNN(nn.Module):
    def __init__(self, S: int = 8, B_boxes: int = 1, C: int = 10):
        """
        S: rozmiar siatki (grid size)
        B_boxes: liczba ramek na jedną komórkę (zazwyczaj 1 w prostym modelu)
        C: liczba klas (tu 10)
        """
        super(VisDroneCNN, self).__init__()
        self.S = S
        self.B_boxes = B_boxes
        self.C = C
        
        self.output_dim = self.B_boxes * 5 + self.C

        # Ekstrakcja cech (Backbone)
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

        # Głowica detekcyjna (Detection Head)
        self.detection_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, self.output_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.detection_head(x)
        
        # Zmieniamy kolejność wymiarów, aby pasowała do [B, S, S, 15]
        x = x.permute(0, 2, 3, 1) 
        
        return x