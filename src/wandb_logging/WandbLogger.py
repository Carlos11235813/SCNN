import torch
import wandb
import warnings
import logging

from src.cnn.DetectionLosses import DetectionLosses
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# Added since average_precision_score, have no zero_division arg, and throws warnings all the time.
warnings.filterwarnings("ignore", message="No positive class found")

logger = logging.getLogger(__name__)

class WandbLogger:

    @staticmethod
    def log_losses(process: str,
                   classification_loss: float,
                   localization_loss: float,
                   objective_loss: float) -> None:

        """
        The method is used to log the losses during training/validation to wandb.

        :param process: Indicates type of losses, ex: process == \"Training Weighted\" or process == \"Validation Original\".
        :param classification_loss: Classification losses
        :param localization_loss: Localization losses
        :param objective_loss: Objective losses
        :return: None
        """

        logger.info(f"[{process}] Attempting to log losses to wandb.")

        total_loss = classification_loss + localization_loss + objective_loss
        wandb.log({
            f'{process} Loss Classification': classification_loss,
            f'{process} Loss Localization': localization_loss,
            f'{process} Loss Objectness': objective_loss,
            f'{process} Total Loss': total_loss})

    @staticmethod
    def precision_recall_f1(process: str,
                    classify_out: torch.Tensor,
                    classify_tar: torch.Tensor) -> None:

        """
        This method is used to log metrics for classification to wandb. The method logs:
            - Precision
            - Recall
            - F1

        :param process: Indicates type of losses, ex: process == \"Training Weighted\" or process == \"Validation Original\".
        :param classify_out: Classification outputs
        :param classify_tar: Classification targets
        :return: None
        """

        logger.info(f"[{process}] Attempting to log precision/recall/f1 to wandb.")

        out = classify_out.cpu().detach()
        tar = classify_tar.cpu().detach()

        if out.dtype == torch.bfloat16:
            out = out.float()
        if tar.dtype == torch.bfloat16:
            tar = tar.float()

        out = (torch.sigmoid(out) > 0.5).numpy().astype(int)
        tar = tar.numpy().astype(int)

        precision = precision_score(out, tar, average='macro', zero_division=0)
        recall = recall_score(out, tar, average='macro', zero_division=0)
        f1 = f1_score(out, tar, average='macro', zero_division=0)
        ap = average_precision_score(out, tar, average='macro')

        wandb.log({
            f'{process} Classification Precision': precision,
            f'{process} Classification Recall': recall,
            f'{process} Classification F1': f1,
            f'{process} Classification Average Precision': ap,
        })


    @staticmethod
    def ap_metrics(process: str,
                   preds_per_image: list[list[dict]],
                   real_per_image: list[list[dict]],
                   device: torch.device) -> None:
        """
        Computes and logs mean AP@50 and AP@75 to wandb.

        :param process: Indicates type of losses, ex: process == \"Training Weighted\" or process == \"Validation Original\".
        :param preds_per_image: List of predictions per image (each element is a list of pred dicts).
        :param real_per_image: List of ground truth boxes per image (each element is a list of gt dicts).
        :param device: Device to use for computation.
        :return: None
        """

        logger.info(f"[{process}] Attempting to log mAP@50/mAP@75,  to wandb.")

        total_ap50 = 0.0
        total_ap75 = 0.0
        num_images = len(preds_per_image)

        for preds, real in zip(preds_per_image, real_per_image):
            total_ap50 += DetectionLosses.compute_ap(preds, real, device, 0.5)
            total_ap75 += DetectionLosses.compute_ap(preds, real, device, 0.75)

        mean_ap50 = total_ap50 / max(num_images, 1)
        mean_ap75 = total_ap75 / max(num_images, 1)

        wandb.log({
            f'{process} mAP@50': mean_ap50,
            f'{process} mAP@75': mean_ap75,
        })

    @staticmethod
    def log_latency(process: str, latencies: list) -> None:
        """
        Computes and logs average latency and std of latencies.

        :param process: Indicates type of losses, ex: process == \"Training Weighted\" or process == \"Validation Original\".
        :param latencies:  list of latencies measured.
        :return: None
        """

        wandb.log({
            f"{process} Latency Mean [ms]": sum(latencies) / len(latencies),
            f"{process} Latency Std [ms]": torch.tensor(latencies).std().item(),
        })