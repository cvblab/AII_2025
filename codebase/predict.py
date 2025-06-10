import torch
from torch.utils.data import DataLoader
from transformers import SamProcessor
from codebase.data.dataset import SegDataset, create_dataset, custom_collate_fn, get_dataset_path
import numpy as np
from codebase.models.sam import test_sam
from codebase.models.stardist import test_stardist
from codebase.models.unet import test_unet
from codebase.models.unet_semantic_segmentation import test_semantic_segmentation
from codebase.models.cellpose_model import test_cellpose


if __name__ == "__main__":
    print(torch.version.cuda)
    print("torch version:", torch.__version__)

    data = "subtilis"  # aureus  dsb  mixed  breast subtilis
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

    model_type = "semantic_segmentation"  # or "unet", "stardist" "sam"
    semantic = False
    threshold = 0.7
    nms_iou_threshold = 0.5
    tp_thresholds = [round(th, 2) for th in np.arange(0.5, 1.0, 0.05)]
    instance_seg_model_path = "../weights/sam/sam_model_dsb_best.pth"
    semantic_seg_model_path = f"../logs/training/semantic2/{data}/unet_final_epoch45.pth"
    yolo_path = f"../logs/training/yolov8_{data}/weights/best.pt"
    cellpose_path = f"models/cellpose/cellpose_{data}"

    if model_type == "sam":
        test_sam(DEVICE, test_data, instance_seg_model_path, semantic_seg_model_path, yolo_path, tp_thresholds, nms_iou_threshold, backbone="base", semantic=semantic)
    elif model_type == "unet":
        test_unet(DEVICE, test_data, instance_seg_model_path, semantic_seg_model_path, yolo_path, tp_thresholds, nms_iou_threshold, semantic=semantic)
    elif model_type == "stardist":
        test_stardist(DEVICE, test_data,tp_thresholds)
    elif model_type == "semantic_segmentation":
        test_semantic_segmentation(DEVICE, test_data, semantic_seg_model_path)
    elif model_type == "cellpose":
        test_cellpose(DEVICE, test_data, tp_thresholds, cellpose_path)