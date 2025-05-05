import sys
import os
import torch
from csbdeep.utils import Path, normalize
from monai.data import ultrasound_confidence_map
from skimage.filters.rank import threshold
from torch.backends.mkl import verbose
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, AutoModel, AutoProcessor
from codebase.data.dataset import SegDataset, create_dataset, custom_collate_fn
from segment_anything import sam_model_registry, SamPredictor
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from codebase.utils.metrics import calculate_metrics, average_precision
from codebase.utils.visualize import plot_detections_vs_groundtruth, plot_nms, calculate_bbox_accuracy, plot_bboxes
import torch.nn as nn
import monai
from tifffile import imread
import sys
from ultralytics import YOLO
import cv2
from Stardist.stardist.models import StarDist2D
import pandas as pd

def nms(boxes, scores, iou_threshold=0.5):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas of bounding boxes
    areas = (x2 - x1) * (y2 - y1)
    max_area = 16384
    valid_indices = torch.where(areas <= max_area)[0]
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    areas = areas[valid_indices]
    order = scores.argsort(descending=True)  # Sort by score (highest first)

    keep = []
    while order.numel() > 0:
        i = order[0].item()  # Index of the box with the highest score
        keep.append(i)

        # Compute IoU of the kept box with the rest
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than the threshold
        remaining = torch.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return torch.tensor(keep)


def yolo_bboxes(image):
    weights_path = "../weights/yolov8n_dsb18.pt" # "runs/detect/yolov8n_dsb18/weights/best.pt" "runs/detect/yolov8n_fluo/weights/best.pt"
    model = YOLO(weights_path)
    image_resized = torch.tensor(image).permute(0,3,1,2)
    # Perform prediction on the image
    results = model.predict(source=image_resized, save=False, save_txt=False,verbose=False)
    boxes = results[0].boxes.xyxy
    confidence = results[0].boxes.conf
    return boxes, confidence


def pad_predictions(gt_masks, pred_masks):
    pred_num_objects = pred_masks.shape[0]
    gt_num_objects = gt_masks.shape[0]

    if pred_num_objects < gt_num_objects:
        # If there are fewer predicted objects, pad pred_masks to match ground truth
        padding = gt_num_objects - pred_num_objects
        padding_tensor = torch.zeros((padding, *pred_masks.shape[1:]), device=pred_masks.device)
        pred_masks = torch.cat([pred_masks, padding_tensor], dim=0)
    elif pred_num_objects > gt_num_objects:
        # If there are more predicted objects, trim pred_masks to match ground truth
        pred_masks = pred_masks[:gt_num_objects]

    return pred_masks