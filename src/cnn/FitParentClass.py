import torch
from torch import nn
import wandb

from src.cnn.BuildTargets import BuildTargets
from src.cnn.DetectionLosses import DetectionLosses

from src.callbacks.EarlyStoping import EarlyStopping
from src.callbacks.SaveBest import SaveBest
from src.models.LossWeights import LossWeights
from src.cnn.cnn_utils import apply_nms

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

        wandb.log({"Training Loss Objectness": total_objectness,
                   "Training Loss Localization": total_localization,
                   "Training Loss Classification": total_classification,
                   "Training Total Loss": total_loss})


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
        total_ap = 0.0
        num_batches = 0

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

                preds = DetectionLosses.build_preds_from_output(outputs)
                preds = apply_nms(preds)
                real = []
                for y in yolo:
                    for box, label in zip(y.boxes, y.labels):
                        real.append({
                            "bbox": box.tolist(),
                            "class": label.item()})
                ap = DetectionLosses.compute_ap(preds, real)
                total_ap += ap
                num_batches += 1
        mean_ap = total_ap / max(num_batches, 1)
        wandb.log({"Val Loss Objectness": total_objectness,
                   "Val Loss Localization": total_localization,
                   "Val Loss Classification": total_classification,
                   "Validation Total Loss": total_loss,
                   "Validation mAP": mean_ap})

        return total_loss

    def fit(self,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            wandb_config: dict = None,
            val_loader: torch.utils.data.DataLoader = None,
            loss_weights: LossWeights = None) -> None:
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
        early_stopping = EarlyStopping(patience=5)
        save_best = SaveBest()
        callbacks = [early_stopping, save_best]
        for epoch in range(epochs):
            self.train()
            loss = self._train(dataloader=train_loader,
                               optimizer=optimizer,
                               device=device,
                               loss_weights=loss_weights)

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