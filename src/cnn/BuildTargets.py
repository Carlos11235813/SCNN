import torch

class BuildTargets:

    @staticmethod
    def build_targets(yolo_batch: torch.Tensor,
                       grid_h: int,
                       grid_w: int,
                       num_classes: int,
                       device: torch.device,
                       dtype=torch.float32) -> torch.Tensor:
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
                              device=device,
                              dtype=dtype)
        for idx, yolo in enumerate(yolo_batch):
            boxes = yolo.boxes.to(device)
            labels = yolo.labels.to(device)

            cord_x = (boxes[:, 0] * grid_w).long()
            cord_y = (boxes[:, 1] * grid_h).long()

            targets[idx, cord_y, cord_x, 0] = 1.0

            targets[idx, cord_y, cord_x, 1:5] = boxes
            targets[idx, cord_y, cord_x, 5:] = torch.eye(10, device=device,dtype=dtype)[labels]
        return targets