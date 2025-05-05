import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import torch
from skimage.filters.rank import threshold
from tensorflow.python.framework.test_ops import None_
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, AutoModel, AutoProcessor
from codebase.data.dataset import SegDataset, create_dataset, custom_collate_fn
from codebase.data.preprocess import augment_dataset_with_original, augmenter
from segment_anything import sam_model_registry, SamPredictor
import os
import matplotlib.pyplot as plt
from .sam import pad_predictions
import numpy as np
import matplotlib.patches as patches
from codebase.utils.metrics import calculate_metrics,  average_precision
from codebase.utils.visualize import plot_ap,plot_detections_vs_groundtruth,plot_loss
from predict_instance_segmentation import stardist_centroids
import torch.nn as nn
import monai
import sys

def unet_model(input_shape, num_classes):
    """
    Builds a U-Net model.

    Args:
        input_shape (tuple): Shape of the input image (H, W, C).
        num_classes (int): Number of output classes. For binary segmentation, use 1.

    Returns:
        keras.Model: A compiled U-Net model.
    """
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output Layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid' if num_classes == 1 else 'softmax')(c9)

    model = Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
                  metrics=['accuracy'])

    return model

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

def train_unet(DEVICE, train_data, num_epochs, threshold, output_path):
    print("Training U-NET")
    last_model_path = f'{output_path}/weights/last.pth'
    best_model_path = f'{output_path}/weights/best.pth'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/weights", exist_ok=True)
    losses_list = []
    average_precisions_list = []
    best_ap = 0

    num_classes = 1  # Set to >1 for multi-class segmentation

    input_shape = (256, 256, 2)  # Image of (256,256) and object's bounding box mask of (256,256)
    model = unet_model(input_shape=input_shape, num_classes=num_classes)
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    model.compile(optimizer=optimizer, loss=loss)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)  # Define the optimizer
    sub_batch_size = 8  # Amount of cells that will be detected at once

    for epoch in range(num_epochs):
        total_loss_epoch = 0
        epoch_losses = []
        total_TP, total_FP, total_FN = 0, 0, 0

        for batch_index, batch in enumerate(train_data):
            batch_loss = 0

            if batch is None or len(batch["image"]) == 0:
                print(f"Skipping empty batch {batch_index}.")
                continue

            for item_index in range(len(batch["image"])):
                pred_masks = []
                num_objects = batch['num_objects_per_image'][item_index]
                print(num_objects)

                img_name = str(batch['path'][item_index]).split("\\")[1]
                valid_bboxes = batch["bounding_boxes"][item_index][:num_objects]
                valid_gt_masks = batch['instance_gt_masks'][item_index][:num_objects].float()
                img_and_bboxes = []

                for bbox in valid_bboxes:
                    bbox_mask = create_bbox_mask(batch["image"][item_index].shape[:2], bbox)  # Create bounding box mask
                    img_and_bbox = np.concatenate([tf.image.rgb_to_grayscale(batch["image"][item_index]), np.expand_dims(bbox_mask, axis=-1)], axis=-1)  # Combine image + bbox mask
                    img_and_bboxes.append(img_and_bbox)

                img_and_bboxes = tf.convert_to_tensor(np.array(img_and_bboxes), dtype=tf.float32)

                valid_gt_masks = valid_gt_masks.numpy()

                # Convert the NumPy array to a TensorFlow tensor
                valid_gt_masks = tf.convert_to_tensor(valid_gt_masks, dtype=tf.float32)
                valid_gt_masks = tf.expand_dims(valid_gt_masks, axis=-1)
                if img_and_bboxes.shape[0] != valid_gt_masks.shape[0]:
                    #input_all_cells, instance_masks = pad_masks(input_all_cells, instance_masks)
                    print(f"Skipping sample {item_index} from batch {batch_index} due to mismatch in number of objects: "
                          f"{img_and_bboxes.shape[0]} predicted, {valid_gt_masks.shape[0]} ground truth.")
                    continue  # Skip this sample and move to the next one

                total_loss_image = 0  # Initialize total loss for the current image

                print("Image min/max:", tf.reduce_min(img_and_bboxes[:,:,:,0]).numpy(), tf.reduce_max(img_and_bboxes[:,:,:,0]).numpy())
                print("Mask min/max:", tf.reduce_min(valid_gt_masks).numpy(), tf.reduce_max(valid_gt_masks).numpy())

                # Iterate over sub-batches
                for j in range(0, len(img_and_bboxes), sub_batch_size):
                    input_sub_batch = img_and_bboxes[j:j+sub_batch_size]
                    masks_sub_batch = valid_gt_masks[j:j+sub_batch_size]

                    with tf.GradientTape() as tape:
                        batched_cells_predictions = model(input_sub_batch,
                                                          training=True)  # Obtain predictions for a sub-batch

                        print("Pred min:", tf.reduce_min(batched_cells_predictions).numpy(),
                              "max:", tf.reduce_max(batched_cells_predictions).numpy())

                        visualize_samples(input_tensor=input_sub_batch, gt_masks=masks_sub_batch,
                                          preds=batched_cells_predictions, num_samples=8)

                        loss = tf.keras.losses.binary_crossentropy(masks_sub_batch, batched_cells_predictions)
                        mini_batch_loss = tf.reduce_mean(loss)

                    gradients = tape.gradient(mini_batch_loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, model.trainable_variables))  # Apply gradients and update weights
                    total_loss_image += mini_batch_loss.numpy()  # Add batch loss to image loss
                    pred_masks.append(batched_cells_predictions)

                #total_loss_epoch += total_loss_image  # Add image loss to epoch loss
                batch_loss += total_loss_image
                pred_masks = [torch.from_numpy(p.numpy()).squeeze(-1) for p in pred_masks]  # Now shape [N, 256, 256]
                # # Now you can concatenate
                pred_masks = torch.cat(pred_masks, dim=0)
                #
                TP, FP, FN = calculate_metrics(pred_masks.detach().cpu().numpy(), np.squeeze(valid_gt_masks.cpu().numpy(), axis=-1),
                                                threshold=threshold)
                print(f"Batch {batch_index}, Item {item_index} Number of cells:{num_objects}, TP: {TP}, FP: {FP}, FN: {FN}")
                #
                # total_TP += TP
                # total_FP += FP
                # total_FN += FN

                # Optional visualization on last epoch
                #if epoch % 5 == 0 or epoch == num_epochs - 1:

                # plot_detections_vs_groundtruth(
                #     detections=pred_masks.detach().cpu().numpy(),
                #     ground_truth=valid_gt_masks.cpu().numpy(),
                #     image=batch["image"][item_index],
                #     bounding_boxes=batch["bounding_boxes"][item_index],
                #     input_points=[],
                #     prompt="bounding_boxes",
                #     threshold=threshold,
                #     epoch=epoch,
                #     img_name=img_name,
                #     output_path=output_path,
                #     mode="train"
                # )

            print(f"Batch {batch_index}/{len(train_data)} loss: {batch_loss}")
            total_loss_epoch += batch_loss
        print(f"Epoch {epoch + 1} completed. Total Loss: {total_loss_epoch}")  # Print total loss for the epoch

        # AP = average_precision(total_TP, total_FP, total_FN)
        # losses_list.append(total_loss)
        # average_precisions_list.append(AP)
        # print(
        #     f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Epoch Losses: {epoch_losses}, Average Precision: {AP:.4f}")
        #
        # # Save last model weights
        # torch.save(model.state_dict(), last_model_path)
        # print(f"Last epoch weights saved at {last_model_path}")
        #
        # # Save best model weights
        # if AP > best_ap:
        #     best_ap = AP
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f"Best model weights updated with AP: {best_ap:.4f}, saved at {best_model_path}")
        #
        # plot_loss(losses_list, epoch, output_path)
        # plot_ap(average_precisions_list, epoch, output_path)

    # Save model in .keras format
    save_path_keras = "unet_instance_segmentation.keras"
    model.save(save_path_keras)
    print(f"Model saved to {save_path_keras}")


import matplotlib.pyplot as plt

def visualize_samples(input_tensor, gt_masks, preds, num_samples=3):
    if tf.is_tensor(input_tensor):
        input_tensor = input_tensor.numpy()
    if tf.is_tensor(gt_masks):
        gt_masks = gt_masks.numpy()
    if tf.is_tensor(preds):
        preds = preds.numpy()

    for i in range(min(num_samples, len(input_tensor))):
        image = input_tensor[i, :, :, 0]
        bbox_mask = input_tensor[i, :, :, 1]
        gt_mask = gt_masks[i]
        pred = preds[i]

        # Apply threshold to predictions to get binary mask for values above 0.5
        pred_above_05 = (pred > 0.5).astype(float)

        fig, axs = plt.subplots(1, 5, figsize=(15, 4))  # Now 5 plots

        axs[0].imshow(image, cmap='gray')
        axs[0].set_title(f"Image {i + 1}")
        axs[0].axis('off')

        axs[1].imshow(bbox_mask, cmap='Reds')
        axs[1].set_title("BBox Mask")
        axs[1].axis('off')

        axs[2].imshow(gt_mask, cmap='Greens')
        axs[2].set_title("GT Mask")
        axs[2].axis('off')

        axs[3].imshow(pred, cmap='Greens')
        axs[3].set_title("Prediction")
        axs[3].axis('off')

        axs[4].imshow(pred_above_05, cmap='Blues')
        axs[4].set_title("Pred > 0.5")
        axs[4].axis('off')

        plt.tight_layout()
        plt.show()