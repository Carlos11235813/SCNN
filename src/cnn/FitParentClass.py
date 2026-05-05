import torch
from torch import nn
import wandb
import logging

from src.cnn.BuildTargets import BuildTargets
from src.cnn.DetectionLosses import DetectionLosses

from src.callbacks.EarlyStoping import EarlyStopping
from src.callbacks.SaveBest import SaveBest
from src.models.LossWeights import LossWeights
from src.wandb_logging.WandbLogger import WandbLogger

logger = logging.getLogger(__name__)

class FitParentClass(nn.Module):
    def __init__(self):
        super(FitParentClass, self).__init__()

    def _train(self,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                loss_weights: LossWeights) -> float:
        """
        Iterate once trough dataloader, and updates the weights in the model.
        It logs data to wandb.
        If loss_weights are user defined, they are used as is. Otherwise, weights are
        computed automatically each batch to equalize the contribution of each loss component.

        :param dataloader: DataLoader to iterate on
        :param optimizer: Optimizer used in training
        :param device: Specifies the device to run on
        :param loss_weights: Loss weights used to scale objectness, localization and classification losses.
                             If None, weights are computed automatically via auto_weights() each batch.
        :return: It returns the total loss on Dataloader.
        total loss == localization_loss + classify_loss + objectness_loss
        """
        total_loss = 0
        total_objectness = 0
        total_localization = 0
        total_classification = 0


        cumulative_outputs = []
        cumulative_targets = []


        user_defined = True if loss_weights is not None else False
        if not user_defined:
            loss_weights = LossWeights()
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
            obj_mask = targets[:, :, :, 0] == 1
            cumulative_outputs.append(outputs[obj_mask][..., 5:])
            cumulative_targets.append(targets[obj_mask][..., 5:])

            total_objectness += loss_objectness.item()
            total_localization += loss_localization.item()
            total_classification += loss_classification.item()

            if not user_defined:
                loss_weights.auto_weights(loss_objectness,
                                          loss_localization,
                                          loss_classification)

            loss_objectness = loss_weights.objectness * loss_objectness
            loss_localization = loss_weights.localization * loss_localization
            loss_classification = loss_weights.classification * loss_classification


            loss = loss_objectness + loss_localization + loss_classification

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        classify_out = torch.cat(cumulative_outputs, dim=0)
        classify_targets = torch.cat(cumulative_targets, dim=0)

        WandbLogger.precision_recall_f1(process="Validation Weighted",
                                        classify_out=classify_out,
                                        classify_tar=classify_targets)

        WandbLogger.log_losses(process="Train Weighted",
                               classification_loss=total_classification,
                               localization_loss=total_localization,
                               objective_loss=total_objectness)
        return total_loss

    def _validate(self,
                  validation_dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  loss_weights: LossWeights = None) -> float:
        """
        Iterate once trough validation_dataloader, does not update the weights in the model.
        It measures models performance on validation_dataloader.
        It logs data to wandb.
        If loss_weights are user defined, they are used as is. Otherwise, weights are
        computed automatically each batch to equalize the contribution of each loss component.

        :param validation_dataloader: DataLoader to iterate on
        :param device: Specifies the device to run on
        :param loss_weights: Loss weights used to scale objectness, localization and classification losses.
                             If None, weights are computed automatically via auto_weights() each batch.
        :return: It returns the total loss on Dataloader.
        total loss == localization_loss + classify_loss + objectness_loss
        """
        total_loss = 0
        total_objectness = 0
        total_localization = 0
        total_classification = 0

        cumulative_outputs = []
        cumulative_targets = []

        user_defined = True if loss_weights is not None else False
        if not user_defined:
            loss_weights = LossWeights()

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
                obj_mask = targets[:, :, :, 0] == 1
                cumulative_outputs.append(outputs[obj_mask][..., 5:])
                cumulative_targets.append(targets[obj_mask][..., 5:])

                if not user_defined:
                    loss_weights.auto_weights(loss_objectness,
                                              loss_localization,
                                              loss_classification)

                loss_objectness = loss_weights.objectness * loss_objectness
                loss_localization = loss_weights.localization * loss_localization
                loss_classification = loss_weights.classification * loss_classification

                total_objectness += loss_objectness.item()
                total_localization += loss_localization.item()
                total_classification += loss_classification.item()

                loss = loss_objectness + loss_localization + loss_classification
                total_loss += loss.item()

        classify_out = torch.cat(cumulative_outputs, dim=0)
        classify_targets = torch.cat(cumulative_targets, dim=0)

        WandbLogger.precision_recall_f1(process="Validation Weighted",
                                        classify_out=classify_out,
                                        classify_tar=classify_targets)

        WandbLogger.log_losses(process="Validation Weighted",
                               localization_loss=total_localization,
                               classification_loss=total_classification,
                               objective_loss=total_objectness)

        return total_loss

    def fit(self,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            wandb_config: dict = None,
            val_loader: torch.utils.data.DataLoader = None,
            loss_weights: LossWeights = None,
            early_stopping: int = 10,
            save_best: str = None) -> None:
        """
        Training loop that works for specified number of epochs.
        It validates model on validation dataloader, if it is specified.
        If loss_weights are not provided, auto weighting is used and logged to wandb as 'auto'.

        :param epochs: Specifies the number of epochs to run.
        :param optimizer: Specifies the optimizer to use.
        :param train_loader: Specifies the dataloader to iterate on in training mode.
        :param wandb_config: Specifies the wandb configuration.
        :param val_loader: Specifies the dataloader to iterate on in validation mode.
        :param loss_weights: Loss weights used to scale objectness, localization and classification losses.
                             If None, weights are computed automatically via auto_weights() each batch.
        :param early_stopping: Specifies the number of epochs to stop after, if the val loss does not improve.
        :param save_best: Specifies path to save the best model.
        :return: None.
        """
        wandb_config = wandb_config or {}

        if loss_weights is None:
            for k in LossWeights().__dict__.keys():
                wandb_config["initial " + k + " weight"] = "auto"

        else:
            for k in loss_weights.__dict__.keys():
                wandb_config["initial " + k + " weight"] = loss_weights.__dict__[k]

        wandb_config["max epochs"] = epochs
        wandb_config["optimizer"] = optimizer
        wandb_config["train_loader length (num batches)"] = len(train_loader)
        wandb_config["val_loader length (num batches)"] = len(val_loader)

        wandb.init(
            project="SCNN",
            config=wandb_config
        )
        device = next(self.parameters()).device
        early_stopping = EarlyStopping(patience=early_stopping)
        save_best = SaveBest(path=save_best)
        callbacks = [early_stopping, save_best]
        logger.info(f"Starting Model Training")
        logger.info(f"Max epochs == {epochs}")
        logger.info(f"Train dataset len == {len(train_loader)}")
        logger.info(f"Validation dataset len == {len(val_loader)}")
        logger.info(f"Optimizer == {optimizer}")
        logger.info(f"Device == {device}")
        logger.info(f"Loss Weights == {loss_weights}")
        logger.info(f"Early stopping patience == {early_stopping}")
        logger.info(f"Saving best model to == {save_best}")
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.train()
            loss = self._train(dataloader=train_loader,
                               optimizer=optimizer,
                               device=device,
                               loss_weights=loss_weights)

            if val_loader is None:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Train Total Loss: {loss}")
            if val_loader is not None:
                self.eval()
                val_loss = self._validate(validation_dataloader=val_loader,
                                          device=device)
                logger.info(f"Epoch {epoch + 1}/{epochs}, Train Total Loss: {loss}, Val Total Loss: {val_loss}")
                for callback in callbacks:
                    callback(model=self,
                             epoch=epoch + 1,
                             loss=val_loss)

                if any(getattr(cb, "stop", False) for cb in callbacks):
                    break

        wandb.finish()