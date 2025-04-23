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
from torchvision.transforms.functional import rgb_to_grayscale
from torch.utils.data import DataLoader
from tqdm import tqdm

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

bce_loss_fn = nn.BCEWithLogitsLoss()
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# ---------- 1. U-Net MODEL ----------
class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final(dec1))

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

import matplotlib.pyplot as plt

def visualize_samples(input_tensor, gt_masks, preds):
    if torch.is_tensor(input_tensor):
        input_tensor = input_tensor.cpu().numpy()
    if torch.is_tensor(gt_masks):
        gt_masks = gt_masks.cpu().numpy()

    for i in range(len(input_tensor)):
        image = input_tensor[i, 0, :, :]  # First channel: grayscale image
        bbox_mask = input_tensor[i, 1, :, :]  # Second channel: bbox mask

        gt_mask = gt_masks[i]
        pred = preds[i]

        fig, axs = plt.subplots(1, 4, figsize=(12, 4))

        axs[0].imshow(image, cmap='gray')
        axs[0].set_title(f"Image {i + 1}")
        axs[0].axis('off')

        axs[1].imshow(bbox_mask, cmap='Reds')
        axs[1].set_title("BBox Mask")
        axs[1].axis('off')

        axs[2].imshow(gt_mask, cmap='Greens')
        axs[2].set_title("GT Mask")
        axs[2].axis('off')

        axs[3].imshow(pred.detach().cpu().squeeze().numpy(), cmap='Greens')
        axs[3].set_title("pred")
        axs[3].axis('off')

        plt.tight_layout()
        plt.show()
def train_unett(DEVICE, train_data, num_epochs, threshold, output_path):
    model = UNet(in_channels=2, out_channels=1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
                pred_masks = []
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

                # Check only the image channel (first channel) of img_and_bboxes
                image_channel = img_and_bboxes[:, 0, :, :]
                print(f"image_channel min: {image_channel.min().item():.4f}, max: {image_channel.max().item():.4f}")

                print(f"valid_gt_masks min: {valid_gt_masks.min().item():.4f}, max: {valid_gt_masks.max().item():.4f}")


                total_loss_image = 0  # Initialize total loss for the current image

                # # Iterate over sub-batches
                # for j in range(0, len(img_and_bboxes), sub_batch_size):
                #     input_sub_batch = img_and_bboxes[j:j + sub_batch_size]
                #     masks_sub_batch = valid_gt_masks[j:j + sub_batch_size]
                #
                #
                #     batched_cells_predictions = model(input_sub_batch.float())  # Obtain predictions for a sub-batch
                #     visualize_samples(input_tensor=input_sub_batch, gt_masks=masks_sub_batch,
                #                       preds=batched_cells_predictions)
                #     batched_cells_predictions = batched_cells_predictions.permute(0, 2, 3, 1) # Add channel dimension
                #
                #     #sub_batch_loss = seg_loss(masks_sub_batch, batched_cells_predictions)
                #     sub_batch_loss = bce_loss_fn(masks_sub_batch, batched_cells_predictions)
                #     pred_masks.append(batched_cells_predictions)
                #
                #     optimizer.zero_grad()
                #     sub_batch_loss.backward()
                #     optimizer.step()
                #
                #     total_loss_image += sub_batch_loss.item()
                #
                # pred_masks = [tensor for sublist in pred_masks for tensor in sublist]

                optimizer.zero_grad()  # Moved outside the loop: one step per image
                total_loss_image = 0
                total_loss_tensor = 0  # Accumulate loss as a tensor for .backward()
                total_loss_tensor = torch.tensor(0.0, requires_grad=True, device=DEVICE)
                pred_masks = []

                # Iterate over sub-batches
                for j in range(0, len(img_and_bboxes), sub_batch_size):
                    input_sub_batch = img_and_bboxes[j:j + sub_batch_size]
                    masks_sub_batch = valid_gt_masks[j:j + sub_batch_size]

                    batched_cells_predictions = model(input_sub_batch.float())
                    batched_cells_predictions = batched_cells_predictions.permute(0, 2, 3, 1)

                    print(input_sub_batch.min(), input_sub_batch.max())
                    print(masks_sub_batch.min(), masks_sub_batch.max())
                    print(batched_cells_predictions.min(), batched_cells_predictions.max())

                    visualize_samples(input_tensor=input_sub_batch, gt_masks=masks_sub_batch,
                                      preds=batched_cells_predictions)


                    sub_batch_loss = bce_loss_fn(masks_sub_batch, batched_cells_predictions)
                    pred_masks.extend(batched_cells_predictions.detach().cpu()) # detach for metric computation

                    total_loss_tensor = total_loss_tensor + sub_batch_loss   # accumulate tensor
                    total_loss_image += sub_batch_loss.item()  # accumulate for logging

                # Backpropagation and optimization step per image
                total_loss_tensor.backward()

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

            batch_loss += total_loss_image
            # epoch_losses.append(batch_loss.item())

            print(f"Batch {batch_index}/{len(train_data)} loss: {batch_loss}")
            total_loss_epoch += batch_loss
        print(f"Epoch {epoch + 1} completed. Total Loss: {total_loss_epoch}")

