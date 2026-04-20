import torch
from torch import nn
import wandb

from src.cnn.DetectionLosses import DetectionLosses

class FitParentClass(nn.Module):
    def __init__(self):
        super(FitParentClass, self).__init__()

    @staticmethod
    def _build_targets(yolo_batch: torch.Tensor,
                       grid_h: int,
                       grid_w: int,
                       num_classes: int,
                       device: torch.device) -> torch.Tensor:
        """
        Transforms batch of Yolo outputs into target boxes.
        That are compatible with models output.

        :param yolo_batch: Batch of Yolo outputs
        :param grid_h: Height of the grid
        :param grid_w: Width of the grid
        :param num_classes: Number of classes in classification part of network
        :param device: Specifies the device to run on
        :return: Tensor of target boxes, shape: (batch_size, grid_h, grid_w, 5 + num_classes)
        """
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
    def generate_anchors(base_size: float=1.0,
                         scales: list[float]=[0.03, 0.06, 0.09],
                         aspect_ratio: list[float]=[0.75, 1.0, 1.25]) -> torch.Tensor:
        anchors=[]
        for scale in scales:
            for ratio in aspect_ratio:
                w=base_size*scale*(ratio**0.5)
                h=base_size*scale/(ratio**0.5)
                anchors.append([w, h])
        return torch.tensor(anchors, dtype=torch.float32) 

    @staticmethod
    def non_maximum_suppression(boxes: torch.Tensor,
                                scores: torch.Tensor,
                                iou_threshold: float=0.5) -> torch.Tensor:
        indices=torch.argsort(scores, descending=True)
        keep=[]
        while indices.numel() > 0:
            current=indices[0]
            keep.append(current)

            if indices.numel() == 1:
                break
            
            remaining_boxes=boxes[indices[1:]]
            current_box = boxes[current].unsqueeze(0).repeat(len(remaining_boxes), 1)
            iou = DetectionLosses().compute_iou(current_box, remaining_boxes)
            indices= indices[1:][iou < iou_threshold]
        
        return torch.tensor(keep, dtype=torch.long)

    def _it_over_dataloader(self,
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

            targets = self._build_targets(yolo,
                                          grid_h=outputs.size(1),
                                          grid_w=outputs.size(2),
                                          num_classes=10,
                                          device=device)


            loss_objectness, loss_localization, loss_classification = DetectionLosses().compute_basic_loss(outputs=outputs,
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

                targets = self._build_targets(yolo,
                                              grid_h=outputs.size(1),
                                              grid_w=outputs.size(2),
                                              num_classes=10,
                                              device=device
                                              )
                loss_objectness, loss_localization, loss_classification = DetectionLosses().compute_basic_loss(
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
            wandb_config = None,
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