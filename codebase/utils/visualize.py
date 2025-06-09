import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import os
from codebase.utils.metrics import calculate_iou
from codebase.utils.test_utils import fill_small_holes_in_masks

def plot_bboxes_nms(ax, image, bboxes):
    ax.imshow(image, cmap="gray")  # Assuming a grayscale image
    for box in bboxes:
        x1, y1, x2, y2 = box[:4]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
    ax.axis("off")

def plot_nms(image, gt_boxes, original_bboxes, filtered_bboxes, iou_threshold):
    """
       Plot bboxes before and after NMS
       """
    # Plot the original and filtered bounding boxes
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # Plot Before NMS
    axs[0].set_title("GT bboxes")
    plot_bboxes_nms(axs[0], image, gt_boxes)

    # Plot Before NMS
    axs[1].set_title("Predictions before NMS")
    plot_bboxes_nms(axs[1], image, original_bboxes)

    # Plot After NMS
    axs[2].set_title(f"Prediction after NMS ({iou_threshold} threshold)")
    plot_bboxes_nms(axs[2], image, filtered_bboxes)

    plt.tight_layout()
    plt.show()

def plot_instance_segmentation(detections, ground_truth, image, bounding_boxes, threshold, epoch, img_name, output_path, mode):
    """
       Detections and ground_truth numpy array (n_objects, 256, 256)
       Image torch tensor (256,256,3)
       Plots original image, ground truth, input_prompt, predictions, and TP,FP
       """
    num_objects = ground_truth.shape[0]  # Number of objects
    num_detections = detections.shape[0]  # Number of detected objects

    # Define a colormap for consistent coloring
    colors = [np.random.randint(0, 255, size=3) for _ in range(max(num_objects, num_detections))]

    # Plot combined ground truth and predicted masks with the same colors for corresponding objects
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    # Transpose image for correct display (if it's in CHW format)
    if image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    # Plot original image
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Combine ground truth masks for visualization (assign consistent colors)
    print("ground_truth shape:", ground_truth.shape)

    if ground_truth.ndim == 2:
        ground_truth = np.expand_dims(ground_truth, axis=0)
    combined_ground_truth_color = np.zeros((ground_truth.shape[1], ground_truth.shape[2], 3), dtype=np.uint8)
    for i in range(num_objects):
        combined_ground_truth_color[ground_truth[i] > 0] = colors[i]  # Use consistent color per object

    axs[1].imshow(combined_ground_truth_color)
    axs[1].set_title("Ground Truth Masks")
    axs[1].axis('off')

    # Combine ground truth masks for visualization (assign consistent colors)
    combined_ground_truth = np.zeros((ground_truth.shape[1], ground_truth.shape[2]), dtype=np.uint8)
    for i in range(num_objects):
        combined_ground_truth[ground_truth[i] > 0] = 1  # Binary representation

    # Plot combined ground truth with bounding boxes
    axs[2].imshow(combined_ground_truth, cmap='gray')
    axs[2].set_title("Ground Truth with Input Prompt")
    axs[2].axis('off')

    # Draw bounding boxes on the combined ground truth mask
    for box in bounding_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=2, edgecolor='red', facecolor='none')
        axs[2].add_patch(rect)

    # Combine predicted masks for visualization (assign same colors)
    detections = (detections > 0.5).astype(np.uint8)
    detections = fill_small_holes_in_masks(detections, area_threshold=100)
    combined_detections = np.zeros((detections.shape[1], detections.shape[2], 3), dtype=np.uint8)
    for i in range(num_detections):
        #combined_detections[detections[i] > 0] = colors[i]  # Use same color for corresponding object
        mask = detections[i, :, :] > 0  # Assumes the mask is in the first channel for each object
        combined_detections[mask] = colors[i % len(colors)]

    axs[3].imshow(combined_detections)
    axs[3].set_title("Predicted Masks")
    axs[3].axis('off')

    tp_fn_fp_mask = np.zeros((detections.shape[1], detections.shape[2], 3), dtype=np.uint8)

    matched_gt = set()
    matched_detections = set()

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
            # True Positive: Mark the predicted mask as green
            tp_fn_fp_mask[pred_mask > 0] = [0, 255, 0]  # Green
            matched_gt.add(best_gt_idx)
            matched_detections.add(det_idx)
        else:
            # False Positive: Mark the unmatched predicted mask as red
            tp_fn_fp_mask[pred_mask > 0] = [255, 0, 0]  # Red

    # Plot the TP and FN visualization
    axs[4].imshow(tp_fn_fp_mask)
    axs[4].set_title(f"TP & FN (Threshold = {threshold})")
    axs[4].axis('off')

    plt.tight_layout()

    if mode != "test":
        output_dir = f"{output_path}/epoch_{epoch}"
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        # Filepath to save the plot
        output_path = os.path.join(output_dir, img_name)
        plt.savefig(output_path, dpi=300)

    # else:
    plt.show()


def plot_semantic_segmentation(pred_mask, gt_mask, input_tensor):
    """Plot input image, ground truth mask, predicted mask, and overlay (pred=white, gt=green)."""
    input_np = input_tensor.detach().cpu().permute(1, 2, 0).numpy()
    pred_np = pred_mask.squeeze().detach().cpu().numpy()
    gt_np = gt_mask.squeeze().detach().cpu().numpy()

    # Create overlay RGB image: black background
    overlay = np.zeros((*gt_np.shape, 3), dtype=np.float32)
    overlay[pred_np == 1] = [1.0, 1.0, 1.0]  # white for prediction
    overlay[gt_np == 1] = [1.0, 0.0, 0.0]    # red for ground truth
    overlay[np.logical_and(pred_np == 1, gt_np == 1)] = [0.0, 1.0, 0.0]  # yellow if both overlap

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(input_np, cmap='gray')
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(gt_np, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    axs[2].imshow(pred_np, cmap='gray')
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    axs[3].imshow(overlay)
    axs[3].set_title("Overlay (White=Pred, Red=GT, Green=Both)")
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()



def visualize_single_cells(input_tensor, gt_masks, preds, binary_preds):
    if torch.is_tensor(input_tensor):
        input_tensor = input_tensor.cpu().numpy()
    if torch.is_tensor(gt_masks):
        gt_masks = gt_masks.cpu().numpy()
    if torch.is_tensor(preds):
        preds = preds.cpu().detach().numpy()
    if torch.is_tensor(binary_preds):
        binary_preds = binary_preds.cpu().detach().numpy()

    for i in range(len(input_tensor)):
        image = input_tensor[i, 0, :, :]  # Grayscale image
        bbox_mask = input_tensor[i, 1, :, :]  # BBox channel

        gt_mask = gt_masks[i]
        pred = preds[i]
        binary_pred = binary_preds[i]

        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1.2])

        ax0 = plt.subplot(gs[0, 0])
        ax0.imshow(image, cmap='gray')
        ax0.set_title("Image")
        ax0.axis('off')

        ax1 = plt.subplot(gs[0, 1])
        ax1.imshow(bbox_mask, cmap='Reds')
        ax1.set_title("BBox Mask")
        ax1.axis('off')

        ax2 = plt.subplot(gs[0, 2])
        ax2.imshow(gt_mask, cmap='Greens')
        ax2.set_title("GT Mask")
        ax2.axis('off')

        ax3 = plt.subplot(gs[0, 3])
        ax3.imshow(pred, cmap='Greens')
        ax3.set_title("Pred")
        ax3.axis('off')

        ax4 = plt.subplot(gs[1, 1])
        ax4.imshow(binary_pred, cmap='Greens')
        ax4.set_title("Binary Prediction")
        ax4.axis('off')

        # Histogram of all logits
        ax5 = plt.subplot(gs[1, 2])
        ax5.hist(pred.flatten(), bins=100, color='blue', alpha=0.7)
        ax5.set_title("Logit Value Distribution (All Pixels)")
        ax5.set_xlabel("Logit value")
        ax5.set_ylabel("Frequency")
        ax5.grid(True)

        # Histogram of only high logits (above 95th percentile)
        threshold = np.percentile(pred, 95)
        high_vals = pred[pred > threshold]

        ax6 = plt.subplot(gs[1, 3:])
        ax6.hist(high_vals.flatten(), bins=50, color='orange', alpha=0.7)
        ax6.set_title(f"High Logits > 95th Percentile ({threshold:.2f})")
        ax6.set_xlabel("Logit value")
        ax6.set_ylabel("Frequency")
        ax6.grid(True)

        plt.suptitle(f"Visualization for Sample {i + 1}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def plot_loss(epoch_losses_list,num_epochs,output_path):
    # Plot the metrics after training
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, num_epochs + 1), epoch_losses_list, marker='o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Loss')
    plt.legend()
    # Save the plot
    output_folder = f"{output_path}/performance"
    os.makedirs(output_folder, exist_ok=True)
    plot_save_path = f"{output_folder}/{num_epochs}_loss.png"
    plt.savefig(plot_save_path)
    #plt.show()

def plot_ap(average_precisions_list,num_epochs,output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, num_epochs + 1), average_precisions_list, marker='o', color='orange', label='AP')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.title('Average Precision per Epoch')
    plt.legend()

    # Save the plot
    output_folder = f"{output_path}/performance"
    os.makedirs(output_folder, exist_ok=True)
    plot_save_path = f"{output_folder}/{num_epochs}_ap.png"
    plt.savefig(plot_save_path)
    #plt.show()

def calculate_bbox_accuracy(ground_truth_boxes, predicted_boxes, iou_threshold=0.5):
    """
       Calculate metrics for bbox predictions
       """
    def calculate_iou_tensor(box1, box2):
        """Calculate IoU for two bounding boxes (PyTorch tensors)."""
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        # Calculate intersection area
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Calculate areas of the individual boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute IoU
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    TP = 0
    FP = 0
    FN = 0

    tp_indices = []
    fp_indices = []
    matched_gt = set()  # To track matched ground truth boxes

    for pred_idx in range(predicted_boxes.shape[0]):
        pred_box = predicted_boxes[pred_idx]
        best_iou = 0
        best_gt_idx = -1

        # Find the best matching ground truth box
        for gt_idx in range(ground_truth_boxes.shape[0]):
            if gt_idx in matched_gt:
                continue  # Skip already matched ground truth boxes

            iou = calculate_iou_tensor(pred_box, ground_truth_boxes[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if IoU is above the threshold
        if best_iou >= iou_threshold:
            TP += 1
            tp_indices.append(pred_idx)
            matched_gt.add(best_gt_idx)
        else:
            FP += 1
            fp_indices.append(pred_idx)

    # Calculate False Negatives
    FN = ground_truth_boxes.shape[0] - len(matched_gt)

    # Calculate Precision, Recall, and F1 score
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"Boxes: {len(ground_truth_boxes)}, TP: {TP}, FP: {FP}, FN: {FN}, Precision: {precision}, Recall: {recall}, F1: {f1_score}")
    return TP,FP,FN,precision,recall,f1_score,tp_indices,fp_indices

def plot_bboxes(image, gt_bboxes, pred_bboxes, tp_indices, fp_indices):
    """
    Plots ground truth boxes, predicted boxes, and TP/FP boxes on an image.

    Args:
        image (torch.Tensor or np.array): Image to plot on (H, W, C).
        gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (num_gt, 4).
        pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (num_pred, 4).
        tp_indices (list): Indices of true positive predicted boxes.
        fp_indices (list): Indices of false positive predicted boxes.
    """

    def add_boxes(ax, boxes, color, label):
        for box in boxes:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=color, facecolor='none', label=label
            )
            ax.add_patch(rect)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['Ground Truth Boxes', 'Predicted Boxes', 'TP (Green) and FP (Red)']

    # Plot each type
    for idx, ax in enumerate(axes):
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(titles[idx])

    # Plot ground truth boxes
    add_boxes(axes[0], gt_bboxes, 'blue', 'GT')

    # Plot predicted boxes
    add_boxes(axes[1], pred_bboxes, 'orange', 'Pred')

    # Plot TP and FP boxes
    add_boxes(axes[2], pred_bboxes[tp_indices], 'green', 'TP')
    add_boxes(axes[2], pred_bboxes[fp_indices], 'red', 'FP')

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm



def plot_imgs(data):

    for batch_index, batch in enumerate(tqdm(data)):
        # Assumes a batched dictionary          # (B, C, H, W) tensor

        if batch is None or len(batch["image"]) == 0:
            print(f"Skipping empty batch {batch_index}.")
            continue

        for item_index in range(len(batch["image"])):

            num_objects = batch["num_objects_per_image"][item_index]
            print("objects",num_objects)
            valid_bboxes = batch["bounding_boxes"][item_index][:num_objects]
            valid_gt_masks = batch['instance_gt_masks'][item_index][:num_objects].float().unsqueeze(-1)


            plot_instance_segmentation(
                detections=valid_gt_masks.cpu().numpy().squeeze(),
                ground_truth=valid_gt_masks.cpu().numpy().squeeze(),
                image=batch["image"][item_index],
                bounding_boxes=valid_bboxes,
                threshold=0.5,
                epoch=0,
                img_name="",
                output_path="",
                mode="test"
            )


