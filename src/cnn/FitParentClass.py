import torch
from torch import nn
import wandb

from src.cnn.BuildTargets import BuildTargets
from src.cnn.DetectionLosses import DetectionLosses

from src.callbacks.EarlyStoping import EarlyStopping
from src.callbacks.SaveBest import SaveBest

class FitParentClass(nn.Module):
    def __init__(self):
        super(FitParentClass, self).__init__()

    def _train(self,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
        """
        Iterate once trough dataloader, and updates the weights in the model.
        It logs data to wandb.
        :param dataloader: DataLoader to iterate on
        :param optimizer: Optimizer used in training
        :param device: Specifies the device to run on
        :return: It returns the total loss on Dataloader.
        total loss == localization_loss + classify_loss + objectness_loss
        """
        total_loss = 0
        total_objectness = 0
        total_localization = 0
        total_classification = 0
        for images, yolo in dataloader:
            images = images.to(device)
            optimizer.zero_grad()

            outputs = self(images)

            targets = BuildTargets.build_targets(yolo,
                                          grid_h=outputs.size(1),
                                          grid_w=outputs.size(2),
                                          num_classes=10,
                                          device=device)


            loss_objectness, loss_localization, loss_classification = DetectionLosses.compute_basic_loss(
                                                                                                        outputs=outputs,
                                                                                                        targets=targets)

            total_objectness += loss_objectness.item()
            total_localization += loss_localization.item()
            total_classification += loss_classification.item()

            loss = loss_objectness + loss_localization + loss_classification

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        wandb.log({"Training Loss Objectness": total_objectness,
                   "Training Loss Localization": total_localization,
                   "Training Loss Classification": total_classification,
                   "Training Total Loss": total_loss})


        return total_loss

    def _validate(self,
                  validation_dataloader: torch.utils.data.DataLoader,
                  device: torch.device) -> float:
        """
        Iterate once trough validation_dataloader, does not update the weights in the model.
        It mesures models performance on validation_dataloader.
        It logs data to wandb.

        :param validation_dataloader: DataLoader to iterate on
        :param device: Specifies the device to run on
        :return: It returns the total loss on Dataloader.
        total loss == localization_loss + classify_loss + objectness_loss
        """
        total_loss = 0
        total_objectness = 0
        total_localization = 0
        total_classification = 0
        with torch.no_grad():
            for images, yolo in validation_dataloader:
                images = images.to(device)
                outputs = self(images)

                targets = BuildTargets.build_targets(yolo,
                                              grid_h=outputs.size(1),
                                              grid_w=outputs.size(2),
                                              num_classes=10,
                                              device=device
                                              )
                loss_objectness, loss_localization, loss_classification = DetectionLosses.compute_basic_loss(
                                                                                            outputs=outputs,
                                                                                            targets=targets)
                total_objectness += loss_objectness.item()
                total_localization += loss_localization.item()
                total_classification += loss_classification.item()

                loss = loss_objectness + loss_localization + loss_classification
                total_loss += loss.item()
        wandb.log({"Val Loss Objectness": total_objectness,
                   "Val Loss Localization": total_localization,
                   "Val Loss Classification": total_classification,
                   "Validation Total Loss": total_loss})

        return total_loss

    def fit(self,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            wandb_config: dict = None,
            val_loader: torch.utils.data.DataLoader = None) -> None:
        """
        Training loop that works for specified number of epochs.
        It validates model on validation dataloader, if it is specified.


        :param epochs: Specifies the number of epochs to run
        :param optimizer: Specifies the optimizer to use
        :param train_loader: Specifies the dataloader to iterate on in training mode
        :param wandb_config: Specifies the wandb configuration
        :param val_loader: Specifies the dataloader to iterate on in validation mode
        :return: None
        """
        wandb.init(
            project="SCNN",
            config=wandb_config
        )
        device = next(self.parameters()).device
        early_stopping = EarlyStopping(patience=5)
        save_best = SaveBest()
        callbacks = [early_stopping, save_best]
        for epoch in range(epochs):
            self.train()
            loss = self._train(dataloader=train_loader,
                               optimizer=optimizer,
                               device=device)

            print(f"Epoch {epoch + 1}/{epochs}, Train Total Loss: {loss}")
            if val_loader is not None:
                self.eval()
                val_loss = self._validate(validation_dataloader=val_loader,
                                          device=device)
                print(f"Epoch {epoch + 1}/{epochs}, Val Total Loss: {val_loss}")
                for callback in callbacks:
                    callback(model=self,
                             epoch=epoch + 1,
                             loss=val_loss)

                if any(getattr(cb, "stop", False) for cb in callbacks):
                    break

        wandb.finish()