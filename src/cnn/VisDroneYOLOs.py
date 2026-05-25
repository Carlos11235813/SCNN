import torch
import torch.nn as nn
from yolov5.models.yolo import DetectionModel
from src.cnn.FitParentClass import FitParentClass

class VisDroneYOLO(FitParentClass):
    """
    YOLOv5s-based detector fine-tuned on the VisDrone dataset.
    
    Args:
        yaml_path (str): Path to the YOLOv5s YAML configuration file. Defaults to "yolov5s.yaml".
        weights_path (str): Path to the pre-trained weights file. Defaults to "yolov5s-visdrone.pt".
    """


    def __init__(self, yaml_path: str = "yolov5s.yaml", weights_path: str = "yolov5s-visdrone.pt"):
        super(VisDroneYOLO, self).__init__()
        print(f"Budowanie architektury z pliku: {yaml_path}")
        print(f"Ładowanie wag z pliku: {weights_path}")
        
        # Bulding the base model using the specified YAML configuration
        self.base_model = DetectionModel(cfg=yaml_path, ch=3)
        
        # Loading the weights from the specified path
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Checking the structure of the checkpoint to ensure the loading of the correct part
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.base_model.load_state_dict(checkpoint['model'].state_dict(), strict=False)
        else:
            self.base_model.load_state_dict(checkpoint, strict=False)
            
        # Turning off export for the last layer to ensure getting the raw grid output
        self.base_model.model[-1].export = False 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        input_height = x.shape[2]
        input_width = x.shape[3]
        
        # Scaling is 8 for P3 layer
        grid_h = input_height // 8
        grid_w = input_width // 8
        p3_elements = grid_h * grid_w * 3  # 3 anchors per grid cell
        
        # Putting the input through the base model
        base_output = self.base_model(x)[0]
        
        # Cutting out the P3 part from the output
        p3_flattened = base_output[:, :p3_elements, :]
        p3_structured = p3_flattened.view(batch_size, 3, grid_h, grid_w, 15)
        
        # Taking the first anchor to get a clean tensor [Batch, Grid_H, Grid_W, 15]
        final_grid = p3_structured[:, 0, :, :, :]
        
        return final_grid
