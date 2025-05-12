import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import os

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union (IoU) for two binary masks.
    """
    intersection = np.sum(np.logical_and(pred_mask, gt_mask))
    union = np.sum(np.logical_or(pred_mask, gt_mask))
    if union == 0:
        return 0.0  # Avoid division by zero
    return intersection / union

def metrics_semantic_segmentation(pred_mask_bin, gt_mask):
    # Compute TP, FP, FN
    pred_mask_bin = pred_mask_bin.squeeze().bool()
    gt_mask_bin = gt_mask.squeeze().bool()

    TP = torch.logical_and(pred_mask_bin, gt_mask_bin).sum().item()
    FP = torch.logical_and(pred_mask_bin, ~gt_mask_bin).sum().item()
    FN = torch.logical_and(~pred_mask_bin, gt_mask_bin).sum().item()

    return TP, FP, FN

def calculate_metrics(detections, ground_truth, threshold):
    """
       Calculate TP,FP and TN metrics for a set of detections.
       """
    TP = 0
    FP = 0
    FN = 0

    matched_gt = set()
    matched_detections = set()
    detections = (detections > 0.5).astype(np.uint8)

    for det_idx, pred_mask in enumerate(detections):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_mask in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue  # Skip already matched ground truth masks

            iou = calculate_iou(pred_mask, gt_mask)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= threshold:
            TP += 1
            matched_gt.add(best_gt_idx)
            matched_detections.add(det_idx)
        else:
            FP += 1

    FN += len(ground_truth) - len(matched_gt)
    FN = len(ground_truth) - FP - TP
    return TP, FP, FN


def average_precision(TP, FP, FN):
    """
    Calculate average precision.
    """
    if (TP + FP + FN) == 0:
        return 0.0  # Avoid division by zero
    return TP / (TP + FP + FN)