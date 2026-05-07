import torch

def collate_fn(batch):
    images, yolos = zip(*batch)
    images = torch.stack(images)  # Stack tensors normally
    return images, list(yolos)