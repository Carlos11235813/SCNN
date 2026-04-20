import torch

from src.cnn.DetectionLosses import DetectionLosses


def generate_anchors(base_size=1.0, scales=[0.03, 0.06, 0.09], aspect_ratio=[0.75, 1.0, 1.25]):
    """
    Generuje zestaw anchorów (ramki referencyjne) o różnych skalach i proporcjach boków.
    Zwraca tensor PyTorch zawierający szerokość i wysokość każdego anchoru.

    Parametry:
    - base_size (float): bazowy rozmiar, względem którego skalowane są anchory.
    - scales (list[float]): lista współczynników skalowania określających wielkość anchorów.
    - aspect_ratio (list[float]): lista proporcji szerokości do wysokości anchorów.
    """
    anchors = []
    for scale in scales:
        for ratio in aspect_ratio:
            w = base_size * scale * (ratio ** 0.5)
            h = base_size * scale / (ratio ** 0.5)
            anchors.append([w, h])
    return torch.tensor(anchors, dtype=torch.float32)



def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    """
    Wykonuje Non-Maximum Suppression (NMS) w celu usunięcia nakładających się ramek o niższych score’ach.
    Zwraca indeksy wybranych ramek, które najlepiej reprezentują wykryte obiekty.

    Parametry:
    - boxes (Tensor): tensor ramek w formacie [N, 4], gdzie każda ramka to (x1, y1, x2, y2).
    - scores (Tensor): tensor score’ów (pewności) dla każdej ramki o rozmiarze [N].
    - iou_threshold (float): próg IoU, powyżej którego ramki są uznawane za nakładające się i odrzucane.
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
    
    return torch.tensor(keep, dtype=torch.long)