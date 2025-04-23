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
from codebase.utils.visualize import plot_ap,plot_detections_vs_groundtruth,plot_loss
from predict_instance_segmentation import stardist_centroids
import torch.nn as nn
import monai
import sys

def predict_masks(DEVICE, model, image, bboxes) -> np.ndarray:
    result_masks = []

    for idx, box in enumerate(bboxes):

        print(f"\r{idx + 1}/{len(bboxes)}", end=" ")
        sys.stdout.flush()
        processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
        inputs = processor(images=image, input_boxes=[[box.tolist()]], return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        pred_mask = outputs["pred_masks"].squeeze()
        result_masks.append(pred_mask)

    result_masks = torch.stack(result_masks)[:, 0, :, :].float()
    return result_masks


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


def train_sam(DEVICE, train_data, num_epochs, threshold, output_path):
    model,optimizer, bce_loss_fn, seg_loss = get_sam_model(DEVICE)
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

            for item_index in range(len(batch["image"])):
                num_objects = batch['num_objects_per_image'][item_index]
                img_name = str(batch['path'][item_index]).split("\\")[1]
                valid_bboxes = batch["bounding_boxes"][item_index][:num_objects]
                valid_gt_masks = batch['instance_gt_masks'][item_index][:num_objects].float().to(DEVICE)


                pred_masks = predict_masks(DEVICE, model,batch["image"][item_index].to(DEVICE),
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
                        input_points=[],
                        prompt="bounding_boxes",
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


def get_sam_model(DEVICE):
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    for name, param in model.named_parameters():
        if name.startswith(("vision_encoder", "prompt_encoder","image_encoder")):
            param.requires_grad_(False)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    # state_dict = torch.load("results/training/dsb18/weights/sam_model_dsb_best.pth",
    #                         map_location="cuda" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.train()

    return model, optimizer, bce_loss_fn, seg_loss



