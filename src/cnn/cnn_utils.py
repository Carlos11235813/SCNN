import torch

from src.cnn.DetectionLosses import DetectionLosses


def generate_anchors(base_size: float=1.0,
                     scales: list[float]=[0.03, 0.06, 0.09],
                     aspect_ratio: list[float]=[0.75, 1.0, 1.25]) -> torch.Tensor:
    """
    Generates a set of anchors (reference boxes) with various scales and aspect ratios.
    Returns a PyTorch tensor containing the width and height of each anchor.

    :param base_size: Base size relative to which anchors are scaled.
    :param scales: List of scaling factors determining anchor sizes.
    :param aspect_ratio: List of width-to-height ratios for the anchors.
    :return: PyTorch tensor of shape (len(scales) * len(aspect_ratios), 2) containing anchor dimensions.
    """
    anchors = []
    for scale in scales:
        for ratio in aspect_ratio:
            w = base_size * scale * (ratio ** 0.5)
            h = base_size * scale / (ratio ** 0.5)
            anchors.append([w, h])
    return torch.tensor(anchors, dtype=torch.float32)



def non_maximum_suppression(boxes: torch.Tensor,
                            scores: torch.Tensor, iou_threshold=0.5) -> torch.Tensor:
    """
    Performs Non-Maximum Suppression (NMS) to remove overlapping boxes with lower scores.
    Returns indices of selected boxes that best represent the detected objects.

    :param boxes: Tensor of boxes in format [N, 4], where each box is (x1, y1, x2, y2).
    :param scores: Tensor of confidence scores for each box, shape [N].
    :param iou_threshold: IoU threshold above which boxes are considered overlapping and discarded.
    :return: Tensor of indices corresponding to the selected boxes.
    """
    indices = torch.argsort(scores, descending=True)
    keep = []
    while indices.numel() > 0:
        current = indices[0]
        keep.append(current)

        if indices.numel() == 1:
            break
        
        remaining_boxes = boxes[indices[1:]]
        current_box = boxes[current].unsqueeze(0).repeat(len(remaining_boxes), 1)
        iou = DetectionLosses.compute_iou(current_box, remaining_boxes)
        indices = indices[1:][iou < iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def apply_nms(preds: list[dict],
              device: torch.device,
              iou_threshold=0.5) -> list:
    """
    Applies Non-Maximum Suppression (NMS) to a list of predicted bounding boxes.

    The function groups predictions by class and removes overlapping boxes based on
    their Intersection over Union (IoU) and confidence scores. For each class, only
    the highest-scoring boxes are kept while suppressing redundant overlapping ones.

    :param preds: List of predicted boxes, where each element is a dictionary:
              {
                  "bbox": [x1, y1, x2, y2],
                  "score": confidence score,
                  "class": predicted class index,
                  "image_id": index of image in batch
              }
    :param device: Torch device (e.g., "cpu" or "cuda") used for tensor operations.
    :param iou_threshold: IoU threshold above which boxes are considered overlapping
                      and suppressed.
    :return: Filtered list of predictions after applying NMS, in the same format as input.
    """
    final_preds = []

    classes = set(p["class"] for p in preds)

    for cls in classes:
        cls_preds = [p for p in preds if p["class"] == cls]

        if len(cls_preds) == 0:
            continue

        boxes = torch.tensor([p["bbox"] for p in cls_preds], dtype=torch.float32, device=device)
        scores = torch.tensor([p["score"] for p in cls_preds], dtype=torch.float32, device=device)

        keep = non_maximum_suppression(boxes, scores, iou_threshold)

        for i in keep:
            final_preds.append(cls_preds[i.item()])

    return final_preds

def calculate_models_size(model: torch.nn.Module) -> float:
    """
    The function checks the size of model passed as argument in MB, and returns it as float.

    :param model: Pytorch model
    :return model_size: indicating number of model parameters in MB
    """

    model_size = 0

    for param in model.parameters():
        if param.data.is_floating_point():
            model_size += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            model_size += param.numel() * torch.iinfo(param.data.dtype).bits

    return model_size