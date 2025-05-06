import torch
from skimage.filters.rank import threshold
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, AutoModel, AutoProcessor
from codebase.data.dataset import SegDataset, create_dataset, custom_collate_fn
from codebase.data.preprocess import augment_dataset_with_original, augmenter
from segment_anything import sam_model_registry, SamPredictor
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from codebase.utils.metrics import calculate_metrics,  average_precision
from codebase.utils.visualize import plot_ap,plot_detections_vs_groundtruth,plot_loss, calculate_bbox_accuracy
from predict_instance_segmentation import stardist_centroids
from codebase.utils.test_utils import yolo_bboxes, nms, pad_predictions
import torch.nn as nn
import monai
import sys
import pandas as pd

def predict_masks(DEVICE, model, model_processor, image, bboxes, test=False):
    result_masks = []

    for idx, box in enumerate(bboxes):

        print(f"\r{idx + 1}/{len(bboxes)}", end=" ")
        sys.stdout.flush()
        inputs = model_processor(images=image, input_boxes=[[box.tolist()]], return_tensors="pt",do_rescale=False).to(DEVICE)

        if test == True:
            with torch.no_grad():
                outputs = model(**inputs)

        else:
            outputs = model(**inputs)
        pred_mask = outputs["pred_masks"].squeeze()
        result_masks.append(pred_mask)

    result_masks = torch.stack(result_masks)[:, 0, :, :].float()
    return result_masks


def train_sam(DEVICE, train_data, num_epochs, threshold, backbone, output_path):
    model,model_processor, optimizer, bce_loss_fn, seg_loss = get_sam_model(DEVICE, backbone)
    print("Training SAM")
    last_model_path = f'{output_path}/weights/last.pth'
    best_model_path = f'{output_path}/weights/best.pth'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/weights", exist_ok=True)
    losses_list = []
    average_precisions_list = []
    best_ap = 0

    for epoch in range(num_epochs):
        total_loss = 0
        epoch_losses = []
        total_TP, total_FP, total_FN = 0, 0, 0

        for batch_index, batch in enumerate(train_data):
            batch_loss = 0

            if batch is None or len(batch["image"]) == 0:
                print(f"Skipping empty batch {batch_index}.")
                continue

            for item_index in range(len(batch["image"])):
                num_objects = batch['num_objects_per_image'][item_index]
                img_name = str(batch['path'][item_index]).split("\\")[1]
                valid_bboxes = batch["bounding_boxes"][item_index][:num_objects]
                valid_gt_masks = batch['instance_gt_masks'][item_index][:num_objects].float().to(DEVICE)


                pred_masks = predict_masks(DEVICE, model, model_processor, batch["image"][item_index].to(DEVICE),
                                           valid_bboxes).to(DEVICE)

                if pred_masks.shape[0] != valid_gt_masks.shape[0]:
                    pred_masks = pad_predictions(valid_gt_masks, pred_masks)
                # loss = bce_loss_fn(pred_masks.float(), valid_gt_masks.float())
                loss = seg_loss(pred_masks, valid_gt_masks)
                batch_loss += loss

                TP, FP, FN = calculate_metrics(pred_masks.detach().cpu().numpy(), valid_gt_masks.cpu().numpy(),
                                               threshold=threshold)
                print(
                    f"Batch {batch_index}, Item {item_index} Number of cells:{num_objects}, TP: {TP}, FP: {FP}, FN: {FN}")

                total_TP += TP
                total_FP += FP
                total_FN += FN

                # Optional visualization on last epoch
                if epoch % 5 == 0 or epoch == num_epochs - 1:
                    plot_detections_vs_groundtruth(
                        detections=pred_masks.detach().cpu().numpy(),
                        ground_truth=valid_gt_masks.cpu().numpy(),
                        image=batch["image"][item_index],
                        bounding_boxes=batch["bounding_boxes"][item_index],
                        threshold=threshold,
                        epoch=epoch,
                        img_name=img_name,
                        output_path=output_path,
                        mode="train"
                    )

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            epoch_losses.append(batch_loss.item())

        AP = average_precision(total_TP, total_FP, total_FN)
        losses_list.append(total_loss)
        average_precisions_list.append(AP)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Epoch Losses: {epoch_losses}, Average Precision: {AP:.4f}")

        # Save last model weights
        torch.save(model.state_dict(), last_model_path)
        print(f"Last epoch weights saved at {last_model_path}")

        # Save best model weights
        if AP > best_ap:
            best_ap = AP
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model weights updated with AP: {best_ap:.4f}, saved at {best_model_path}")

        plot_loss(losses_list, epoch, output_path)
        plot_ap(average_precisions_list, epoch, output_path)


def test_sam(DEVICE, test_data, model_path, tp_thresholds, nms_iou_threshold, backbone):
    model, model_processor, optimizer, bce_loss_fn, seg_loss = get_sam_model(DEVICE, backbone)
    print("Testing SAM")

    state_dict = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    all_APs = []
    all_aps_per_threshold = {threshold: [] for threshold in tp_thresholds}

    for index, test_sample in enumerate(test_data):
        gt_bboxes = test_sample["bounding_boxes"].squeeze(0)
        gt_masks = test_sample['instance_gt_masks'].squeeze(0).float().to(DEVICE)

        yolo_boxes, confs = yolo_bboxes(test_sample["image"])
        keep_indices = nms(yolo_boxes, confs, iou_threshold=nms_iou_threshold)
        yolo_boxes_nms = yolo_boxes[keep_indices.long()]
        print(f"Number of gt cells: {len(gt_bboxes)}")
        print(f"Number of predicted cells: {len(yolo_boxes_nms)}")
        calculate_bbox_accuracy(gt_bboxes, yolo_boxes_nms, iou_threshold=0.5)
        pred_masks = predict_masks(DEVICE, model, model_processor, test_sample["image"].to(DEVICE), bboxes=yolo_boxes_nms, test=True).to(DEVICE)

        for threshold in tp_thresholds:
            TP, FP, FN = calculate_metrics(pred_masks.detach().cpu().numpy(), gt_masks.cpu().numpy(),
                                           threshold=threshold)
            AP = average_precision(TP, FP, FN)
            all_aps_per_threshold[threshold].append(AP)  # Store AP for this threshold
            print(f"Sample {index}, IoU Threshold: {threshold}, AP: {AP}, TP: {TP}, FP: {FP}, FN: {FN}")

        plot_detections_vs_groundtruth(
           detections=pred_masks.detach().cpu().numpy(),
           ground_truth=gt_masks.cpu().numpy(),
           image=test_sample["image"].squeeze(0),
           bounding_boxes=yolo_boxes_nms.cpu().numpy(),
           threshold=0.7,
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


def get_sam_model(DEVICE, backbone):
    model_path = "facebook/sam-vit-" + backbone
    model = SamModel.from_pretrained(model_path)
    model_processor = AutoProcessor.from_pretrained(model_path)
    for name, param in model.named_parameters():
        if name.startswith(("vision_encoder", "prompt_encoder","image_encoder")):
            param.requires_grad_(False)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    model.to(DEVICE)
    model.train()

    return model, model_processor, optimizer, bce_loss_fn, seg_loss



