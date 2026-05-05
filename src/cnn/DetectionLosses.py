from typing import Any
import torch
from torch import nn
from src.detection_demo.demo_utils import models_output_to_boxes
from src.detection_demo.demo_utils import decode_prediction


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
                iou = DetectionLosses.compute_iou(localization_out, localization_tar)
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
        """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.

    Both input tensors are expected to have shape [N, 4], where each box is defined
    in (x1, y1, x2, y2) format. The function computes IoU element-wise, meaning
    boxes1[i] is compared with boxes2[i].

    :param boxes1: Tensor of predicted bounding boxes, shape [N, 4].
    :param boxes2: Tensor of ground truth bounding boxes, shape [N, 4].
    :param eps: Small value to avoid division by zero.
    :return: Tensor of IoU values for each corresponding pair of boxes, shape [N].
        """


        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        union = area1 + area2 - inter + eps

        return inter / union
    

    @staticmethod
    def build_preds_from_output(outputs: torch.Tensor) -> list[dict]:
        """
    Builds list of predicted bounding boxes from model outputs.

    The function decodes raw model outputs using sigmoid/softmax activations,
    converts them into absolute pixel coordinates, filters predictions based on
    objectness score, and assigns class labels with confidence scores.

    :param outputs: Raw model output tensor of shape (batch_size, grid_size, grid_size, 5 + num_classes).
    :return: List of predictions, where each prediction is a dictionary:
         {
             "bbox": [x1, y1, x2, y2],
             "score": confidence score,
             "class": predicted class index,
             "image_id": index of image in batch
         }
        """
        cx, cy, w, h, objectness, class_probs = decode_prediction(outputs)
        boxes = models_output_to_boxes(cx, cy, w, h)

        batch_size, grid_size, _, _ = outputs.shape

        preds = []

        for b in range(batch_size):
            mask = objectness[b] > 0.5

            if mask.sum() == 0:
                continue

            b_boxes = boxes[b][mask]
            b_obj = objectness[b][mask]
            b_cls_probs = class_probs[b][mask]

            cls = torch.argmax(b_cls_probs, dim=-1)
            idx = torch.arange(len(cls), device=cls.device)
            scores = b_obj * b_cls_probs[idx, cls]

            for i in range(len(cls)):
                preds.append({
                    "bbox": b_boxes[i].tolist(),
                    "score": scores[i].item(),
                    "class": cls[i].item(),
                    "image_id": b
                })

        return preds

    @staticmethod
    def match_prediction(preds: list[dict],
                         real: list[dict],
                         device: torch.device,
                         iou_threshold: float=0.5) -> list:
        """
    Matches predicted bounding boxes with ground truth boxes using IoU.

    Each prediction is assigned as True Positive (1) or False Positive (0) based on:
    - IoU threshold
    - matching class
    - matching image_id
    - ensuring each ground truth box is matched at most once

    :param preds: List of predicted boxes (dict format with bbox, class, image_id).
    :param real: List of ground truth boxes (same format as preds, without score).
    :param device: Device to use while computing.
    :param iou_threshold: Minimum IoU required to consider a prediction as correct.
    :return: List of integers (1 for True Positive, 0 for False Positive).
        """
        matched=[]
        used=set()

        for pred in preds:
            best_iou=0
            best_reals=-1
            pred_box = torch.tensor([pred["bbox"]], dtype=torch.float32, device=device)
            for i, reals  in  enumerate(real):
                if i in used:
                    continue
                if pred["class"] != reals["class"]:
                    continue
                if pred["image_id"] != reals["image_id"]:
                    continue
                gt_box = torch.tensor([reals["bbox"]], dtype=torch.float32, device=device)
                score = DetectionLosses.compute_iou(pred_box, gt_box)[0].item()
                if score > best_iou :
                    best_iou=score
                    best_reals=i
            if best_iou>=iou_threshold:
                matched.append(1)
                used.add(best_reals)
            else:
                matched.append(0)
        return matched
    
    @staticmethod
    def compute_ap(preds: list[dict],
                   real: list[dict],
                   device: torch.device,
                   iou_threshold: float=0.5) -> float:
        """
    Computes Average Precision (AP) for a single IoU threshold.

    Predictions are sorted by confidence score, then matched with ground truth boxes.
    Precision and recall are calculated cumulatively, and AP is computed as the area
    under the precision-recall curve using interpolation.

    :param preds: List of predicted boxes (dict format with bbox, score, class, image_id).
    :param real: List of ground truth boxes (dict format with bbox, class, image_id).
    :param device: Device to use while computing.
    :param iou_threshold: IoU threshold used to determine True Positives.
    :return: Average Precision (float) for given IoU threshold.
        """
        if len(real) == 0:
            return 0.0

        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        tp = torch.tensor(
            DetectionLosses.match_prediction(preds, real, device, iou_threshold),
            dtype=torch.float32,
            device=device
        )
        fp = 1.0 - tp

        tp_c = torch.cumsum(tp, dim=0)
        fp_c = torch.cumsum(fp, dim=0)

        recalls = tp_c / len(real)
        precisions = tp_c / (tp_c + fp_c + 1e-6)

        zero = torch.zeros(1, device=device)
        one = torch.ones(1, device=device)
        recalls = torch.cat([zero, recalls, one])
        precisions = torch.cat([zero, precisions, zero])

        precisions = torch.flip(
            torch.cummax(torch.flip(precisions, dims=[0]), dim=0).values,
            dims=[0]
        )

        recall_diff = recalls[1:] - recalls[:-1]
        ap = torch.sum(recall_diff * precisions[1:]).item()

        return ap