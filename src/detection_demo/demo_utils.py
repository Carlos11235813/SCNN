import torch
import torch.nn.functional as f
from PIL import Image, ImageDraw

def decode_prediction(models_prediction: torch.Tensor) -> tuple:
    """
    The function applies activation function to models output.
    It is done since the last layer of model outputs data which is not transformed with Softmax/Sigmoid, or
    any other activation function.


    :param models_prediction: Model output tensor of shape (batch_size, grid_size, grid_size, 5+num_classes).
    :return: named tuple of models prediction after transformation. Shape is the same as function input.
    """

    batch_center_x = torch.sigmoid(models_prediction[..., 0])
    batch_center_y = torch.sigmoid(models_prediction[..., 1])
    batch_box_width = torch.sigmoid(models_prediction[..., 2])
    batch_box_height = torch.sigmoid(models_prediction[..., 3])
    batch_objectness = torch.sigmoid(models_prediction[..., 4])
    batch_classes_prob = f.softmax(models_prediction[..., 5:], dim=-1)
    return batch_center_x, batch_center_y, batch_box_width, batch_box_height, batch_objectness, batch_classes_prob



def models_output_to_boxes(batch_center_x: torch.Tensor,
                          batch_center_y: torch.Tensor,
                          batch_box_width: torch.Tensor,
                          batch_box_height: torch.Tensor,
                          img_w: int = 256,
                          img_h: int = 256) -> torch.Tensor:
    """
    Converts decoded YOLO model outputs (sigmoid-activated coordinates) into
    absolute pixel coordinates in (x1, y1, x2, y2) format for each image in the batch.

    :param batch_center_x: Sigmoid-activated x center of each box relative to its grid cell, shape [B, S, S].
    :param batch_center_y: Sigmoid-activated y center of each box relative to its grid cell, shape [B, S, S].
    :param batch_box_width: Sigmoid-activated box width relative to the full image width, shape [B, S, S].
    :param batch_box_height: Sigmoid-activated box height relative to the full image height, shape [B, S, S].
    :param img_w: Width of the input image in pixels.
    :param img_h: Height of the input image in pixels.
    :return: Tensor of bounding boxes in pixel coordinates (x1, y1, x2, y2), shape [B, S, S, 4].
    """
    grid = batch_center_x.shape[-1]
    cell_w = img_w / grid
    cell_h = img_h / grid

    offset_x = torch.arange(grid).float().view(1, 1, grid).expand(batch_center_x.shape[0], grid, grid)
    offset_y = torch.arange(grid).float().view(1, grid, 1).expand(batch_center_y.shape[0], grid, grid)

    cx = (offset_x + batch_center_x) * cell_w
    cy = (offset_y + batch_center_y) * cell_h

    bw_px = batch_box_width * img_w
    bh_px = batch_box_height * img_h

    x1 = cx - bw_px / 2
    x2 = cx + bw_px / 2
    y1 = cy - bh_px / 2
    y2 = cy + bh_px / 2

    boxes_on_images_in_batch = torch.stack([x1, y1, x2, y2], dim=-1)

    return boxes_on_images_in_batch


def filter_predictions(boxes: torch.Tensor,
                       batch_objectness: torch.Tensor,
                       batch_classes_prob: torch.Tensor,
                       conf_threshold: float = 0.7) ->list[dict]:
    """
    The function filters model predictions in such a way that only boxes for which sum of objectness and
    class score is higher than the threshold are passed as valid boxes to be printed.

    :param boxes: Bounding boxes in pixel coordinates (x1, y1, x2, y2), shape [B, S, S, 4].
    :param batch_objectness: Sigmoid-activated objectness scores indicating presence of an object, shape [B, S, S].
    :param batch_classes_prob: Softmax-activated class probabilities for each grid cell, shape [B, S, S, 10].
    :param conf_threshold: Minimum confidence score (average of objectness and class probability) for a box to be kept.
    :return: List of dicts with keys 'boxes' [N, 4], 'scores' [N] and 'class_ids' [N] for each image in the batch.
    """
    batch_size = boxes.shape[0]
    filtered_boxes = []
    for box in range(batch_size):
        class_scores, class_ids = batch_classes_prob[box].max(dim=-1)
        scores = (batch_objectness[box] + class_scores) / 2

        mask = scores > conf_threshold

        filtered_boxes.append({
            "boxes": boxes[box][mask],
            "scores": scores[mask],
            "class_ids": class_ids[mask]
        })
    return filtered_boxes

def draw_predictions(image: Image.Image,
                     result: dict,
                     colors: list,
                     class_names: list) -> Image.Image:
    draw = ImageDraw.Draw(image)

    for box, score, cls_id in zip(result["boxes"], result["scores"], result["class_ids"]):
        x1, y1, x2, y2 = box.tolist()
        cls_id = int(cls_id)
        color  = colors[cls_id]
        label  = f"{class_names[cls_id]}: {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255))

    return image