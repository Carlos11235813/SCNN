import torch

from src.cnn.DetectionLosses import DetectionLosses


def generate_anchors(base_size: float = 1.0,
                     scales: list[float] = [0.03, 0.06, 0.09],
                     aspect_ratio: list[float] = [0.75, 1.0, 1.25]) -> torch.Tensor:
    anchors = []
    for scale in scales:
        for ratio in aspect_ratio:
            w = base_size * scale * (ratio ** 0.5)
            h = base_size * scale / (ratio ** 0.5)
            anchors.append([w, h])
    return torch.tensor(anchors, dtype=torch.float32)



def non_maximum_suppression(boxes: torch.Tensor,
                            scores: torch.Tensor,
                            iou_threshold: float = 0.5) -> torch.Tensor:
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

    return torch.tensor(keep, dtype=torch.long)