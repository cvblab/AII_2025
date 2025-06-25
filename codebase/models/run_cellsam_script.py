import matplotlib.pyplot as plt
import numpy as np
from cellSAM import segment_cellular_image

def test_cellsam(DEVICE, test_data, tp_thresholds, cellpose_path, data):

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
        mask, boxes, scores = segment_cellular_image(img_np, device=str(DEVICE))

        # Visualize
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        axs[0].imshow(img_np)
        axs[0].set_title("Input Image")
        axs[0].axis('off')

        # Predicted mask
        # mask shape depends on CellSAM output, usually (H, W) or (num_instances, H, W)
        if mask.ndim == 3:
            # If multiple masks, combine for visualization
            combined_mask = np.sum(mask, axis=0) > 0  # Binary combined mask
            axs[1].imshow(combined_mask, cmap='gray')
        else:
            axs[1].imshow(mask, cmap='gray')

        axs[1].set_title("Predicted Mask")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()
