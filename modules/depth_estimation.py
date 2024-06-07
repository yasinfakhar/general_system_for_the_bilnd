import cv2
import torch
import time
import numpy as np
from config import Config

model_type = Config.depth_model_version  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load(Config.depth_model, model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load(Config.depth_model, "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def _preporcess(image):
    return transform(image).to(device)

def depth_estimation(image):
    
    processed_image = _preporcess(image)
 
    with torch.no_grad():
        prediction = midas(processed_image)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    dim = (192*3, 108*4)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    depth_map = cv2.resize(depth_map, dim, interpolation=cv2.INTER_AREA)
    
    return depth_map
    