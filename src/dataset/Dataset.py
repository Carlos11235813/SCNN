import torch
import logging
from src.models.Yolo import Yolo
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

class VisDrone(Dataset):
    def __init__(self, data_dir: str, labels_dir: str, transform=None):
        logger.info(f"Loading VisDrone dataset from {data_dir}")
        super().__init__()
        self.data_dir = Path(data_dir)
        self.labels_dir = Path(labels_dir)
        self.data = sorted(self.data_dir.glob('*.jpg'))
        tra = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64))
            ]
        )
        if not transform:
            logger.warning(f"Transform not specified, using default transform")
            logger.info(f"Default transform == {transform}")
            self.transform = transform or tra


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Yolo]:
        img_path = self.data[idx]
        label_path = self.labels_dir / img_path.with_suffix(".txt").name

        image = Image.open(img_path).convert("RGB")
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

        yolo = Yolo(
            boxes=torch.tensor(boxes, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.long),
        )
        return image, yolo