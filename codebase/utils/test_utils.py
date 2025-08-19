import torch
from ultralytics import YOLO
import numpy as np
from skimage.morphology import remove_small_holes
import os
import pandas as pd
import openpyxl

def nms(boxes, scores, iou_threshold=0.5):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas of bounding boxes
    areas = (x2 - x1) * (y2 - y1)
    max_area = 10000
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


def get_yolo_bboxes(image, weights_path="../weights/yolo/yolov8n_dsb18.pt"):
    model = YOLO(weights_path)
    if image.ndim == 2:
        image_resized = image.unsqueeze(0).repeat(1, 3, 1, 1)

    else:
        image_resized = torch.tensor(image).permute(0, 3, 1, 2)

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

def convert_masks_to_instances(mask_list):
    """
    Converts a list of labeled masks into a list of binary instance masks.

    Input:
        mask_list: list of (256, 256) numpy arrays with labeled integers (0=background, 1+ = object IDs)

    Output:
        instance_masks_list: list of arrays with shape (num_objects, 256, 256)
    """
    instance_masks_list = []

    for mask in mask_list:
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]  # Exclude background

        if len(instance_ids) == 0:
            instance_masks = np.zeros((0, 256, 256), dtype=np.uint8)
        else:
            instance_masks = np.stack([(mask == inst_id).astype(np.uint8) for inst_id in instance_ids])

        instance_masks_list.append(instance_masks)

    return instance_masks_list


def fill_small_holes_in_masks(masks: np.ndarray, area_threshold: int = 100) -> np.ndarray:
    """
    Fill small holes in binary instance masks.

    Args:
        masks (np.ndarray): Array of shape (num_cells, H, W) with binary values (0 or 1).
        area_threshold (int): Maximum area of small holes to fill.

    Returns:
        np.ndarray: Array of same shape as input with small holes filled.
    """
    filled_masks = []
    for i, mask in enumerate(masks):
        filled = remove_small_holes(mask.astype(bool), area_threshold=area_threshold)
        filled_masks.append(filled.astype(np.uint8))

    return np.array(filled_masks)


def save_results_to_excel(results, model, data, filename="ap_threshold_0.5.xlsx"):
    """
    Save results of AP, TP, FP, FN (for IoU threshold 0.5) to Excel.

    Args:
        results (list of dicts): Each dict should contain keys:
            ["Sample", "IoU Threshold", "AP", "TP", "FP", "FN"]
        model (str): Model name
        data (str): Dataset name
        filename (str): Name of the Excel file
    """
    save_dir = os.path.join("..", "logs", "results", model, data)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(results)
    save_path = os.path.join(save_dir, filename)
    df.to_excel(save_path, index=False)

    print(f"[INFO] Saved results to {save_path}")
    return save_path
