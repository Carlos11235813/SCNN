from dataclasses import dataclass
import torch

@dataclass
class YOLO():
    boxes: torch.Tensor
    labels: torch.Tensor

    def __len__(self):
        """
        :return: Number of boxes on image
        """
        return len(self.labels)

    def __getitem__(self, idx):
        box = self.boxes[idx]
        label = self.labels[idx]
        return box, label
