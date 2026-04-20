from typing import Any
import torch
from torch import nn

class DetectionLosses:

    @staticmethod
    def compute_basic_loss(outputs: torch.Tensor,
                           targets: torch.Tensor,
                           loc_loss_iou: bool=False) -> tuple[Any, Any, Any]:
        """
        It computes the basic loss function. Both outputs and targets are expected to have the same shape.
        Specified by FitParentClass._build_targets method. shape: (batch_size, grid_h, grid_w, 5 + num_classes)
        Allows to decide weather to use MSE or IoU as localization loss.

        :param outputs: Output of the model. Tensor of shape: (batch_size, grid_h, grid_w, 5 + num_classes).
        :param targets: True info, that model tried to predict.
        :param loc_loss_iou: If set True, IoU will be used for localization loss.
        Else MSE will be used for localization loss.
        :return: Tuple(loss_objectness, loss_localization, loss_classification)
        """
        objectness_out = outputs[:, :, :, 0]
        objectness_tar = targets[:, :, :, 0]

        criterion_objectness = nn.BCEWithLogitsLoss()
        loss_objectness = criterion_objectness(objectness_out, objectness_tar)

        obj_mask = targets[:, :, :, 0] == 1

        localization_out = outputs[obj_mask][:, 1:5]
        localization_tar = targets[obj_mask][:, 1:5]

        if loc_loss_iou:
            if localization_out.numel() == 0:
                loss_localization = torch.tensor(0.0)
            else:
                iou = DetectionLosses().compute_iou(localization_out, localization_tar)
                loss_localization = 1 - iou.mean()
        else:
            criterion_localization = nn.MSELoss()
            loss_localization = criterion_localization(localization_out, localization_tar)

        classify_out = outputs[obj_mask][:, 5:]
        classify_tar = targets[obj_mask][:, 5:]

        criterion_classification = nn.BCEWithLogitsLoss()
        loss_classification = criterion_classification(classify_out, classify_tar)

        return loss_objectness, loss_localization, loss_classification


    @staticmethod
    def compute_iou(boxes1: torch.Tensor,
                    boxes2: torch.Tensor,
                    eps: float=1e-6) -> torch.Tensor:

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