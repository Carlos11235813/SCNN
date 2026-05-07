from src.detection_demo.demo_utils import (decode_prediction, models_output_to_boxes,
                                           filter_predictions, draw_predictions)
import logging
import torch
import os
from PIL import Image

logger = logging.getLogger(__name__)

class VisualizePrediction:

    def __init__(self,
                 input_image_batch: torch.Tensor,
                 output_batch_prediction: torch.Tensor,
                 img_w: int,
                 img_h: int,
                 conf_threshold: float = 0.6,
                 class_names: list = None,
                 colors: list = None):
        logger.info('Initializing VisualizePrediction class')
        self.output_batch_prediction = output_batch_prediction
        self.img_w = img_w
        self.img_h = img_h
        self.conf_threshold = conf_threshold
        self.input_image_batch = input_image_batch
        self.prediction_images: list[Image.Image] = []
        if class_names is None:
            logger.info('Initializing default class names [Names for VisDrone dataset]')
            self.class_names = [
                "pedestrian", "people", "bicycle", "car", "van",
                "truck", "tricycle", "awning-tricycle", "bus", "motor"
                ]
        else:
            self.class_names = class_names
        if colors is None:
            self.colors = [
                (255, 56, 56), (255, 157, 151), (255, 112, 31),
                (255, 178, 29), (207, 210, 49), (72, 249, 10),
                (146, 204, 23), (61, 219, 134), (26, 147, 52),
                (0, 212, 187),
            ]
        else:
            self.colors = colors

        self._postprocess()

    def _postprocess(self) -> None:
        """
        The method is used to postprocess the output of the model.
        And to draw the predictions. The final images are saved as class field, and can be printed later.
        :return: None
        """
        logger.info('Attempting to postprocess data for plotting')
        bx, by, bw, bh, obj, cls = decode_prediction(self.output_batch_prediction)
        boxes = models_output_to_boxes(bx,
                                       by,
                                       bw,
                                       bh,
                                       img_w=self.img_w,
                                       img_h=self.img_h)

        results = filter_predictions(boxes,
                                     obj,
                                     cls,
                                     conf_threshold=self.conf_threshold)

        for i in range(len(self.input_image_batch)):

            img_np = self.input_image_batch[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype("uint8")
            pil_img = Image.fromarray(img_np)

            pil_img = draw_predictions(image = pil_img,
                                       result=results[i],
                                       colors=self.colors,
                                       class_names=self.class_names)

            self.prediction_images.append(pil_img)

    def save_all(self, output_dir: str) -> None:
        """
        Saves all annotated images to the specified folder.
        The folder is created automatically if it does not exist.

        :param output_dir: Path to the folder where images will be saved.
        :return: None
        """
        logger.info('Attempting to save all images to the specified folder')
        os.makedirs(output_dir, exist_ok=True)
        for i, img in enumerate(self.prediction_images):
            img.save(os.path.join(output_dir, f"prediction_{i}.png"))

    def show(self, index: int) -> None:
        """
        Displays a single annotated image by its index in the batch.

        :param index: Index of the image to display.
        :return: None
        """
        if index < 0 or index >= len(self.prediction_images):
            raise IndexError(f"Index {index} out of range – batch contains {len(self.prediction_images)} images.")
        self.prediction_images[index].show()