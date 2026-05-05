import torch
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:

    def __init__(self, patience: int=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False


    def __call__(self, model: torch.nn.Module, epoch: int, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                logger.warning(f"Early stopping, at epoch == {epoch}. [BREAK]")
