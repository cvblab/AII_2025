import matplotlib.pyplot as plt
import numpy as np
from cellSAM import segment_cellular_image
from codebase.utils.metrics import calculate_metrics,  average_precision
from codebase.utils.visualize import plot_instance_segmentation
from codebase.utils.test_utils import save_results_to_excel
import pandas as pd


def test_cellsam(DEVICE, data, test_data, tp_thresholds):

    all_aps_per_threshold = {threshold: [] for threshold in tp_thresholds}
    results = []
    # Inference loop
    for index, test_sample in enumerate(test_data):
        if test_sample is None or len(test_sample["image"]) == 0:
            print(f"Skipping empty batch {index}.")
            continue

        if test_sample["bounding_boxes"] is None:
            print(f"[WARNING] No bounding boxes for sample: {test_sample['image_path']}")
            continue

        gt_bboxes = test_sample["bounding_boxes"].squeeze(0)
        gt_masks = test_sample['instance_gt_masks'].squeeze(0).float().to(DEVICE)

        # Convert torch tensor to numpy array
        img_tensor = test_sample["image"].squeeze(0)
        img_np = img_tensor.cpu().numpy()

        if img_np.shape[0] == 3:  # CHW to HWC
            img_np = np.transpose(img_np, (1, 2, 0))

        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        # Segment using CellSAM
        pred_mask, boxes, scores = segment_cellular_image(img_np, device=str(DEVICE))

        if pred_mask is None:
            continue
        # Get unique object IDs (excluding background if needed)
        object_ids = np.unique(pred_mask)
        object_ids = object_ids[object_ids != 0]  # Remove background (0)

        # Create binary masks for each object
        binary_pred_masks = np.array([(pred_mask == obj_id).astype(np.uint8) for obj_id in object_ids])

        for threshold in tp_thresholds:

            TP, FP, FN = calculate_metrics(binary_pred_masks, gt_masks.cpu().numpy(), threshold=threshold)
            AP = average_precision(TP, FP, FN)
            all_aps_per_threshold[threshold].append(AP)  # Store AP for this threshold
            print(f"Sample {index}, IoU Threshold: {threshold}, AP: {AP}, TP: {TP}, FP: {FP}, FN: {FN}")

            if threshold == 0.5:
                results.append({
                    "Sample": index,
                    "IoU Threshold": threshold,
                    "AP": AP,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN
                })

        plot_instance_segmentation(
            detections=binary_pred_masks,
            ground_truth=gt_masks.cpu().numpy(),
            image=test_sample["image"].squeeze(0),
            bounding_boxes=[],
            threshold=0.7,
            data= data,
            model = "cellsam",
            index=index
        )

        # Compute mean AP across all samples for each threshold
        mean_aps = {threshold: np.mean(aps) for threshold, aps in all_aps_per_threshold.items()}
        mean_ap_df = pd.DataFrame(list(mean_aps.items()), columns=["IoU Threshold", "Mean AP"])
        print(mean_ap_df)

        save_results_to_excel(results, model="cellsam", data=data)


