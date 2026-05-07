import torch
import os
import logging
import argparse
import sys
import time
from torchvision import transforms

from src.dataset.Dataset import VisDrone
from src.dataset.collate import collate_fn
from src.cnn.VisDroneCNN import VisDroneCNN


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_path(path: str, name: str) -> str:
    if not os.path.exists(path):
        logger.error(f"{name} path does not exist: '{path}'")
        sys.exit(1)
    if not os.path.isdir(path):
        logger.error(f"{name} path is not a directory: '{path}'")
        sys.exit(1)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Train VisDroneCNN model")

    parser.add_argument("--train-data",
                        type=str,
                        default="src/data/VisDrone_Dataset/VisDrone2019-DET-train/images",
                        help="Path to training images directory")
    parser.add_argument("--train-labels",
                        type=str,
                        default="src/data/VisDrone_Dataset/VisDrone2019-DET-train/labels",
                        help="Path to training labels directory")
    parser.add_argument("--val-data",
                        type=str,
                        default="src/data/VisDrone_Dataset/VisDrone2019-DET-val/images",
                        help="Path to validation images directory")
    parser.add_argument("--val-labels",
                        type=str,
                        default="src/data/VisDrone_Dataset/VisDrone2019-DET-val/labels",
                        help="Path to validation labels directory")
    parser.add_argument("--resize",
                        type=int,
                        nargs=2,
                        default=[32, 32],
                        metavar=("HEIGHT", "WIDTH"),
                        help="Resize dimensions as HEIGHT WIDTH (default: 32 32)")
    parser.add_argument("--epochs",
                        type=int,
                        default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Batch size for dataloaders (default: 32)")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="Number of dataloader workers (default: 4)")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="Learning rate for Adam optimizer (default: 1e-3)")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Working directory: {os.getcwd()}")

    validate_path(args.train_data, "train-data")
    validate_path(args.train_labels, "train-labels")
    validate_path(args.val_data, "val-data")
    validate_path(args.val_labels, "val-labels")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(tuple(args.resize))
    ])

    visdrone = VisDrone(data_dir=args.train_data,
                        labels_dir=args.train_labels,
                        transform=transform)
    dataset = torch.utils.data.DataLoader(visdrone,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          collate_fn=collate_fn,
                                          pin_memory=True,
                                          prefetch_factor=2)

    visdrone_val = VisDrone(data_dir=args.val_data,
                            labels_dir=args.val_labels,
                            transform=transform)
    dataset_val = torch.utils.data.DataLoader(visdrone_val,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              collate_fn=collate_fn,
                                              pin_memory=True,
                                              prefetch_factor=2)

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    cnn = VisDroneCNN(B_boxes=1, C=10)
    cnn = cnn.to(device)
    opt = torch.optim.Adam(cnn.parameters(), lr=args.lr)

    wandb_config = {
        "resize": args.resize,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    cnn.fit(epochs=args.epochs,
            optimizer=opt,
            train_loader=dataset,
            val_loader=dataset_val,
            wandb_config=wandb_config)


if __name__ == "__main__":
    main()