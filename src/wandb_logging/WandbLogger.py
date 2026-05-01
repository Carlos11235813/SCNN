import torch
import wandb

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

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

        out = classify_out.cpu().detach()
        tar = classify_tar.cpu().detach()

        out = (torch.sigmoid(out) > 0.5).numpy().astype(int)
        tar = tar.numpy().astype(int)
        precision = precision_score(out, tar, average='macro', zero_division=0)
        recall = recall_score(out, tar, average='macro', zero_division=0)
        f1 = f1_score(out, tar, average='macro', zero_division=0)
        ap = average_precision_score(out, tar, average='macro', zero_division=0)

        wandb.log({
            f'{process} Classification Precision': precision,
            f'{process} Classification Recall': recall,
            f'{process} Classification F1': f1,
            f'{process} Average precision': ap,
        })

