from typing import Any

import torch
from torch import nn
import wandb



class FitParentClass(nn.Module):
    def __init__(self):
        super(FitParentClass, self).__init__()

    @staticmethod
    def _build_targets(yolo_batch: torch.Tensor,
                       grid_h: int,
                       grid_w: int,
                       num_classes: int,
                       device: torch.device):
        batch_size = len(yolo_batch)
        targets = torch.zeros(batch_size,
                              grid_h,
                              grid_w,
                              5 + num_classes,
                              device=device)
        for idx, yolo in enumerate(yolo_batch):
            boxes = yolo.boxes.to(device)
            labels = yolo.labels.to(device)

            cord_x = (boxes[:, 0] * grid_w).long()
            cord_y = (boxes[:, 1] * grid_h).long()

            targets[idx, cord_y, cord_x, 0] = 1.0

            targets[idx, cord_y, cord_x, 1:5] = boxes
            targets[idx, cord_y, cord_x, 5:] = torch.eye(10)[labels]
        return targets
    
    @staticmethod
    def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-6):
        x1_1 = boxes1[:, 0] - boxes1[:, 2] / 2
        y1_1 = boxes1[:, 1] - boxes1[:, 3] / 2
        x2_1 = boxes1[:, 0] + boxes1[:, 2] / 2
        y2_1 = boxes1[:, 1] + boxes1[:, 3] / 2

        x1_2 = boxes2[:, 0] - boxes2[:, 2] / 2
        y1_2 = boxes2[:, 1] - boxes2[:, 3] / 2
        x2_2 = boxes2[:, 0] + boxes2[:, 2] / 2
        y2_2 = boxes2[:, 1] + boxes2[:, 3] / 2

        x1 = torch.max(x1_1, x1_2)
        y1 = torch.max(y1_1, y1_2)
        x2 = torch.min(x2_1, x2_2)
        y2 = torch.min(y2_1, y2_2)

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)

        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        union = area1 + area2 - inter + eps

        return inter / union


    @staticmethod
    def compute_loss(outputs: torch.Tensor, targets: torch.Tensor) -> tuple[Any, Any, Any]:
        objectness_out = outputs[:, :, :, 0]
        objectness_tar = targets[:, :, :, 0]

        criterion_objectness = nn.BCEWithLogitsLoss()
        loss_objectness = criterion_objectness(objectness_out, objectness_tar)

        obj_mask = targets[:, :, :, 0] == 1

        localization_out = outputs[obj_mask][:, 1:5]
        localization_tar = targets[obj_mask][:, 1:5]

        criterion_localization = nn.MSELoss()
        loss_localization = criterion_localization(localization_out, localization_tar)

        classify_out = outputs[obj_mask][:, 5:]
        classify_tar = targets[obj_mask][:, 5:]

        criterion_classification = nn.BCEWithLogitsLoss()
        loss_classification = criterion_classification(classify_out, classify_tar)

        return loss_objectness, loss_localization, loss_classification

    def _it_over_dataloader(self,
                            dataloader: torch.utils.data.DataLoader,
                            optimizer: torch.optim.Optimizer,
                            device: torch.device) -> float:
        total_loss = 0
        total_objectness = 0
        total_localization = 0
        total_classification = 0
        for images, yolo in dataloader:
            images = images.to(device)
            optimizer.zero_grad()

            outputs = self(images)

            targets = self._build_targets(yolo,
                                          grid_h=outputs.size(1),
                                          grid_w=outputs.size(2),
                                          num_classes=10,
                                          device=device)


            loss_objectness, loss_localization, loss_classification = self.compute_loss(outputs=outputs,
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
        total_loss = 0
        total_objectness = 0
        total_localization = 0
        total_classification = 0
        with torch.no_grad():
            for images, yolo in validation_dataloader:
                images = images.to(device)
                outputs = self(images)

                targets = self._build_targets(yolo,
                                              grid_h=outputs.size(1),
                                              grid_w=outputs.size(2),
                                              num_classes=10,
                                              device=device
                                              )
                loss_objectness, loss_localization, loss_classification = self.compute_loss(outputs=outputs,
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
            wand_config = None,
            val_loader: torch.utils.data.DataLoader = None) -> None:
        wandb.init(
            project="SCNN",
            config=wand_config
        )
        device = next(self.parameters()).device

        for epoch in range(epochs):
            self.train()
            loss = self._it_over_dataloader(dataloader=train_loader,
                                                   optimizer=optimizer,
                                                   device=device)

            print(f"Epoch {epoch + 1}/{epochs}, Train Total Loss: {loss / len(train_loader)}")
            if val_loader is not None:
                self.eval()
                val_loss = self._validate(validation_dataloader=val_loader,
                                          device=device)

                print(f"Epoch {epoch + 1}/{epochs}, Val Total Loss: {val_loss / len(val_loader)}")


        wandb.finish()