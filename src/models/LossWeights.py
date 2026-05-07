from dataclasses import dataclass
import torch

@dataclass
class LossWeights:
    """
    The class is used to store weight for each loss type during training.
    """
    objectness: float = 1
    localization: float = 1
    classification: float = 1

    def auto_weights(self,
                     loss_objectness: torch.Tensor,
                     loss_localization: torch.Tensor,
                     loss_classification: torch.Tensor) -> None:
        loss_sum = loss_objectness + loss_localization + loss_classification
        loss_sum = loss_sum.item()

        self.objectness = (1 / 3) / (loss_objectness.item() / loss_sum)
        self.localization = (1 / 3) / (loss_localization.item() / loss_sum)
        self.classification = (1 / 3) / (loss_classification.item() / loss_sum)

