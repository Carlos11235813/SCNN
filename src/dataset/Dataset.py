import torch
from src.models.YOLO import YOLO
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class VisDrone(Dataset):
    def __init__(self, data_dir, labels_dir, transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.labels_dir = Path(labels_dir)
        self.data = sorted(self.data_dir.glob('*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, YOLO]:
        img_path = self.data[idx]
        label_path = self.labels_dir / img_path.with_suffix(".txt").name

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        boxes = []
        labels = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                lab, cx, cy, w, h = parts
                boxes.append([float(cx), float(cy), float(w), float(h)])
                labels.append(int(lab))

        yolo = YOLO(
            boxes=torch.tensor(boxes, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.long),
        )
        return image, yolo


