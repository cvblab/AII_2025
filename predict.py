import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from transformers import SamProcessor
from data.dataset import SegDataset, create_dataset, custom_collate_fn, get_dataset_path, get_model_paths
import numpy as np
from models.sam import test_sam
from models.stardist import test_stardist
from models.unet import test_unet
from models.unet_semantic_segmentation import test_semantic_segmentation
from models.cellpose_model import test_cellpose
from models.run_cellsam import run_cellsam


if __name__ == "__main__":

    print(torch.version.cuda)
    print("torch version:", torch.__version__)

    #choose dataset
    data = "aureus"  # aureus  dsb  breast tcell flow_chamber
    mode = "test"
    images_path, masks_path = get_dataset_path(data, mode)
    dataset = create_dataset(images_path, masks_path, preprocess=True, axis_norm=(0, 1))
    print("Acquiring images from " + data + " dataset.")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # Initialize SAMDataset and DataLoader
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    test_dataset = SegDataset(dataset=dataset, processor=processor)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    model_type = "cellpose"  # or "cellpose", "stardist" "sam", "cellsam
    semantic = False
    threshold = 0.7
    nms_iou_threshold = 0.5
    tp_thresholds = [round(th, 2) for th in np.arange(0.5, 1.0, 0.05)]
    instance_seg_model_path, semantic_seg_model_path, yolo_path, cellpose_path, stardist_path = get_model_paths(data, model_type)

    if model_type == "sam":
        test_sam(DEVICE, data, test_data, instance_seg_model_path, semantic_seg_model_path, yolo_path, tp_thresholds, nms_iou_threshold, backbone="base", semantic=semantic)
    elif model_type == "stardist":
        test_stardist(data,test_data, "combined", stardist_path, tp_thresholds)
    elif model_type == "semantic_segmentation":
        test_semantic_segmentation(DEVICE, test_data, semantic_seg_model_path)
    elif model_type == "cellpose":
        test_cellpose(DEVICE,test_data, tp_thresholds, cellpose_path, data)
    elif model_type == "cellsam":
        run_cellsam(DEVICE, data,test_data, tp_thresholds)