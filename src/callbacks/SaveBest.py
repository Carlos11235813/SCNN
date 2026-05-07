import os
import datetime
import torch
import logging

logger = logging.getLogger(__name__)

class SaveBest:

    def __init__(self, path: str = None):
        self.path = path or os.path.join(os.getcwd(), "SaveBest_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".pth")
        self.best_loss = float("inf")

    def __call__(self, model: torch.nn.Module, epoch: int, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(model.state_dict(), self.path)
            logger.info(f"Save Best, at epoch == {epoch}. [Best loss: {loss}]")