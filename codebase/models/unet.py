import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
import os
import numpy as np
from codebase.utils.metrics import calculate_metrics,  average_precision
from codebase.utils.test_utils import pad_predictions
from codebase.utils.visualize import plot_ap,plot_instance_segmentation,plot_loss, visualize_single_cells, calculate_bbox_accuracy,plot_semantic_segmentation
from codebase.models.unet_semantic_segmentation import predict_binary_mask
import torch.nn as nn
from codebase.utils.test_utils import get_yolo_bboxes, nms, pad_predictions
from torchvision.transforms.functional import rgb_to_grayscale
import segmentation_models_pytorch as smp
from tqdm import tqdm


def create_bbox_mask(image_shape, bbox):
    """
    Create a binary mask for a given bounding box.
    bbox: [xmin, ymin, xmax, ymax]
    """
    bbox_mask = np.zeros(image_shape, dtype=np.uint8)
     # Convert bbox coordinates to integers
    x1, y1, x2, y2 = map(int, bbox)

    # Mark the region inside the bbox as 1
    bbox_mask[y1:y2, x1:x2] = 1   # Mark the region inside the bbox

    return bbox_mask


def get_unet(DEVICE):
    model = smp.Segformer(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=2,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    # model = smp.Unet(
    #     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=2,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=1,  # model output channels (number of classes in your dataset)
    # )
    model.to(DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return model, optimizer, bce_loss_fn


def train_unet(DEVICE, train_data, num_epochs, threshold, output_path):

    model, optimizer, bce_loss_fn = get_unet(DEVICE)
    print("Training U-NET")

    last_model_path = f'{output_path}/weights/last.pth'
    best_model_path = f'{output_path}/weights/best.pth'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/weights", exist_ok=True)
    losses_list = []
    average_precisions_list = []
    best_ap = 0
    sub_batch_size = 8

    for epoch in range(num_epochs):
        total_loss_epoch = 0
        epoch_losses = []
        total_TP, total_FP, total_FN = 0, 0, 0

        for batch_index, batch in enumerate(tqdm(train_data, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            optimizer.zero_grad()
            batch_loss = 0

            for item_index in range(len(batch["image"])):
                num_objects = batch['num_objects_per_image'][item_index]
                img_name = str(batch['path'][item_index]).split("\\")[-1]
                valid_bboxes = batch["bounding_boxes"][item_index][:num_objects]
                valid_gt_masks = batch['instance_gt_masks'][item_index][:num_objects].float().unsqueeze(-1).to(DEVICE)
                img_and_bboxes = []

                for bbox in valid_bboxes:
                    bbox_mask = torch.tensor(create_bbox_mask(batch["image"][item_index].shape[:2], bbox)).unsqueeze(0)
                    img_gray = rgb_to_grayscale(batch["image"][item_index].permute(2, 0, 1))  # [C=1, H, W]
                    img_plus_bbox = torch.cat([img_gray, bbox_mask], dim=0)  # [2, H, W]
                    img_and_bboxes.append(img_plus_bbox)

                img_and_bboxes = torch.stack(img_and_bboxes).to(DEVICE)
                optimizer.zero_grad()  # Moved outside the loop: one step per image
                total_loss_image = 0
                total_loss_tensor = 0  # Accumulate loss as a tensor for .backward()
                pred_masks = []

                # Iterate over sub-batches
                for j in range(0, len(img_and_bboxes), sub_batch_size):
                    input_sub_batch = img_and_bboxes[j:j + sub_batch_size]
                    masks_sub_batch = valid_gt_masks[j:j + sub_batch_size]

                    batched_cells_predictions = model(input_sub_batch.float())
                    batched_cells_predictions = batched_cells_predictions.permute(0, 2, 3, 1)

                    probs = torch.sigmoid(batched_cells_predictions)
                    binary_preds = (probs > 0.98).float()
                    visualize_single_cells(input_tensor=input_sub_batch, gt_masks=masks_sub_batch,
                                      preds=batched_cells_predictions, binary_preds=binary_preds)

                    sub_batch_loss = bce_loss_fn(masks_sub_batch, batched_cells_predictions)

                    print("loss: " ,sub_batch_loss)

                    #pred_masks.extend(batched_cells_predictions.detach().cpu()) # detach for metric computation
                    pred_masks.extend(binary_preds.detach().cpu())
                    total_loss_tensor += sub_batch_loss   # accumulate tensor
                    total_loss_image += sub_batch_loss.item()  # accumulate for logging

                # Backpropagation and optimization step per image
                total_loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        print(
                            f"{name} grad mean: {param.grad.mean().item():.6f}, grad std: {param.grad.std().item():.6f}")
                        break  # Just show one for brevity
                    elif param.requires_grad:
                        print(f"{name} has no grad")
                        break

                optimizer.step()

                # Step 2: Stack into a single tensor
                pred_masks = torch.stack(pred_masks)

                if pred_masks.shape[0] != valid_gt_masks.shape[0]:
                    pred_masks = pad_predictions(valid_gt_masks, pred_masks)

                TP, FP, FN = calculate_metrics(pred_masks.detach().cpu().numpy(), valid_gt_masks.cpu().numpy(),
                                               threshold=threshold)
                print(
                    f"Batch {batch_index}, Item {item_index} Number of cells:{num_objects}, TP: {TP}, FP: {FP}, FN: {FN}")

                total_TP += TP
                total_FP += FP
                total_FN += FN

                plot_instance_segmentation(
                    detections=pred_masks.detach().cpu().numpy().squeeze(),
                    ground_truth=valid_gt_masks.cpu().numpy().squeeze(),
                    image=batch["image"][item_index],
                    bounding_boxes=[],
                    threshold=threshold,
                    epoch=0,
                    img_name="",
                    output_path="",
                    mode="test"
                )

            batch_loss += total_loss_image
            # epoch_losses.append(batch_loss.item())

            print(f"Batch {batch_index}/{len(train_data)} loss: {batch_loss}")
            total_loss_epoch += batch_loss
            epoch_losses.append(batch_loss)

        AP = average_precision(total_TP, total_FP, total_FN)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss_epoch:.4f}, Epoch Losses: {epoch_losses}, Average Precision: {AP:.4f}")

        # Save last model weights
        torch.save(model.state_dict(), last_model_path)
        print(f"Last epoch weights saved at {last_model_path}")

        # Save best model weights
        if AP > best_ap:
            best_ap = AP
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model weights updated with AP: {best_ap:.4f}, saved at {best_model_path}")

        losses_list.append(total_loss_epoch)
        average_precisions_list.append(AP)
        plot_loss(losses_list, epoch, output_path)
        plot_ap(average_precisions_list, epoch, output_path)



def test_unet(DEVICE, test_data, unet_model_path, semantic_seg_model_path, yolo_path, tp_thresholds, nms_iou_threshold, semantic=False):

    model, model_processor, optimizer, bce_loss_fn, seg_loss = get_unet(DEVICE)
    print("Testing U-Net")

    state_dict = torch.load(unet_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    all_aps_per_threshold = {threshold: [] for threshold in tp_thresholds}
    sub_batch_size = 8

    for index, test_sample in enumerate(test_data):
        gt_bboxes = test_sample["bounding_boxes"].squeeze(0)
        gt_masks = test_sample['instance_gt_masks'].squeeze(0).float().to(DEVICE)

        if semantic==True:
            binary_prediction, gt_binary_mask = predict_binary_mask(DEVICE, test_sample["image"].squeeze(0), test_sample['original_gt_masks'].float(), semantic_seg_model_path)
            plot_semantic_segmentation(binary_prediction, gt_binary_mask, test_sample["image"].squeeze(0).permute(2, 0, 1).to(DEVICE))
            yolo_boxes, confs = get_yolo_bboxes(binary_prediction, yolo_path)

        else:
            yolo_boxes, confs = get_yolo_bboxes(test_sample["image"],yolo_path)

        keep_indices = nms(yolo_boxes, confs, iou_threshold=nms_iou_threshold)
        yolo_boxes_nms = yolo_boxes[keep_indices.long()]
        print(f"Number of gt cells: {len(gt_bboxes)}")
        print(f"Number of predicted cells: {len(yolo_boxes_nms)}")
        calculate_bbox_accuracy(gt_bboxes, yolo_boxes_nms, iou_threshold=0.5)
        img_and_bboxes = []
        pred_masks = []

        for bbox in yolo_boxes_nms:
            bbox_mask = torch.tensor(create_bbox_mask(test_sample["image"][index].shape[:2], bbox)).unsqueeze(0)
            img_gray = rgb_to_grayscale(test_sample["image"][index].permute(2, 0, 1))  # [C=1, H, W]
            img_plus_bbox = torch.cat([img_gray, bbox_mask], dim=0)  # [2, H, W]
            img_and_bboxes.append(img_plus_bbox)

            # Iterate over sub-batches
        for j in range(0, len(img_and_bboxes), sub_batch_size):
            input_sub_batch = img_and_bboxes[j:j + sub_batch_size]
            masks_sub_batch = gt_masks[j:j + sub_batch_size]

            batched_cells_predictions = model(input_sub_batch.float())
            batched_cells_predictions = batched_cells_predictions.permute(0, 2, 3, 1)

            print("Pred min:", batched_cells_predictions.min(), "max:", batched_cells_predictions.max())
            # probs = torch.sigmoid(batched_cells_predictions)

            visualize_single_cells(input_tensor=input_sub_batch, gt_masks=masks_sub_batch,
                                   preds=batched_cells_predictions)

            pred_masks.extend(batched_cells_predictions.detach().cpu())

        pred_masks = torch.stack(pred_masks)
        for threshold in tp_thresholds:
            TP, FP, FN = calculate_metrics(pred_masks.detach().cpu().numpy(), gt_masks.cpu().numpy(),
                                           threshold=threshold)
            AP = average_precision(TP, FP, FN)
            all_aps_per_threshold[threshold].append(AP)  # Store AP for this threshold
            print(f"Sample {index}, IoU Threshold: {threshold}, AP: {AP}, TP: {TP}, FP: {FP}, FN: {FN}")

        plot_instance_segmentation(
           detections=pred_masks.detach().cpu().numpy(),
           ground_truth=gt_masks.cpu().numpy(),
           image=test_sample["image"].squeeze(0),
           bounding_boxes=yolo_boxes_nms.cpu().numpy(),
           threshold=threshold,
           epoch=0,
           img_name="",
           output_path = "",
           mode = "test"
        )
    # Compute mean AP across all samples for each threshold
    mean_aps = {threshold: np.mean(aps) for threshold, aps in all_aps_per_threshold.items()}

    mean_ap_df = pd.DataFrame(list(mean_aps.items()), columns=["IoU Threshold", "Mean AP"])

    # Print the DataFrame
    print(mean_ap_df)