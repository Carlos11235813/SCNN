import torch
import wandb

from sklearn.metrics import precision_score, recall_score, f1_score

class WandbLogger:

    @staticmethod
    def log_losses(process: str,
                   classification_loss: torch.Tensor,
                   localization_loss: torch.Tensor,
                   objective_loss: torch.Tensor) -> None:

        """
        The method is used to log the losses during training/validation to wandb.

        :param process: Indicates type of losses, ex: process == \"Training Weighted\" or process == \"Validation Original\".
        :param classification_loss: Classification losses
        :param localization_loss: Localization losses
        :param objective_loss: Objective losses
        :return: None
        """


        total_loss = classification_loss.item() + localization_loss.item() + objective_loss.item()
        wandb.log({
            f'{process} Loss Classification': classification_loss,
            f'{process} Loss Localization': localization_loss,
            f'{process} Loss Objectness': objective_loss,
            f'{process} Total Loss': total_loss})

    @staticmethod
    def precision_recall_f1(process: str,
                    classify_out: torch.Tensor,
                    classify_tar: torch.Tensor) -> None:

        precision = precision_score(classify_out, classify_tar, average='macro')
        recall = recall_score(classify_out, classify_tar, average='macro')
        f1 = f1_score(classify_out, classify_tar, average='macro')

        wandb.log({
            f'{process} Precision': precision,
            f'{process} Recall': recall,
            f'{process} F1': f1,
        })

