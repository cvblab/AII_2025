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
from codebase.models.sam import test_sam
from codebase.models.stardist import test_stardist
from codebase.models.unettorch import test_unet


if __name__ == "__main__":
    print(torch.version.cuda)
    print("torch version:", torch.__version__)

    data = "fluo"  # fluo  dsb  mixed  breast
    if data == "dsb":
        images_path = "../datasets/dsb2018/test/images/*.tif"
        masks_path = "../datasets/dsb2018/test/masks/*.tif"

    elif data == "fluo":
        images_path = "../datasets/fluorescence_dataset/test/fluorescence/*.tif"
        masks_path = "../datasets/fluorescence_dataset/test/masks/*.tif"

    elif data == "mixed":
        images_path = "../datasets/mixed_dataset/test/source/*.tif"
        masks_path = "../datasets/mixed_dataset/test/target/*.tif"

    elif data == "breast":
        images_path = "../datasets/breast_cancer/test/images/*.tif"
        masks_path = "../datasets/breast_cancer/test/masks/*.tif"

    dataset = create_dataset(images_path, masks_path, preprocess=True, axis_norm=(0, 1))
    print("Acquiring images from " + data + " dataset.")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # Initialize SAMDataset and DataLoader
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    test_dataset = SegDataset(dataset=dataset, processor=processor)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    model_type = "stardist"  # or "unet", "stardist" "sam"
    threshold = 0.7
    nms_iou_threshold = 0.5
    tp_thresholds = [round(th, 2) for th in np.arange(0.5, 1.0, 0.05)]
    model_path = "../weights/sam/sam_model_dsb_best.pth"

    if model_type == "sam":
        test_sam(DEVICE, test_data, model_path, tp_thresholds, nms_iou_threshold, backbone="base")
    elif model_type == "unet":
        test_unet(DEVICE, test_data, model_path, tp_thresholds, nms_iou_threshold)
    elif model_type == "stardist":
        test_stardist(DEVICE, test_data,tp_thresholds)