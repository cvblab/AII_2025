import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt
import torch
from skimage.filters.rank import threshold
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, AutoModel, AutoProcessor
from codebase.data.dataset import SegDataset, create_dataset, custom_collate_fn, get_dataset_path
import os
import matplotlib.patches as patches

def plot_bboxes(ground_truth, image, bounding_boxes):
    num_objects = ground_truth.shape[0]  # Number of objects # Number of detected objects

    # Define a colormap for consistent coloring
    colors = [np.random.randint(0, 255, size=3) for _ in range(num_objects)]

    # Plot combined ground truth and predicted masks with the same colors for corresponding objects
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    # Transpose image for correct display (if it's in CHW format)
    if image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    print(image.shape)
    # Plot original image
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Combine ground truth masks for visualization (assign consistent colors)
    combined_ground_truth = np.zeros((ground_truth.shape[1], ground_truth.shape[2]), dtype=np.uint8)
    for i in range(num_objects):
        combined_ground_truth[ground_truth[i] > 0] = 1  # Binary representation

    # Plot combined ground truth with bounding boxes
    axs[1].imshow(combined_ground_truth, cmap='gray')
    axs[1].set_title("Ground Truth with Bounding Boxes")
    axs[1].axis('off')

    # Draw bounding boxes on the combined ground truth mask
    for box in bounding_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=2, edgecolor='red', facecolor='none')
        axs[1].add_patch(rect)

    # Combine ground truth masks for visualization (assign consistent colors)
    combined_ground_truth_color = np.zeros((ground_truth.shape[1], ground_truth.shape[2], 3), dtype=np.uint8)
    for i in range(num_objects):
        combined_ground_truth_color[ground_truth[i] > 0] = colors[i]  # Use consistent color per object

    axs[2].imshow(combined_ground_truth_color)
    axs[2].set_title("Combined Ground Truth Masks")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

def save_labels(labels_output_path, images_output_path, image_file, image, bounding_boxes):
    # Ensure the image is a NumPy array
    if isinstance(image, torch.Tensor):
        image = image.numpy()  # Convert CHW to HWC

    image = (image * 255).astype(np.uint8)
 # Convert to uint8 for OpenCV

    # Set output path for the image (with .tiff extension)
    single_image_output_path = os.path.join(images_output_path, os.path.splitext(os.path.basename(image_file))[0] + '.tiff')

    # Save the image with bounding boxes as a .tiff file using OpenCV
    cv2.imwrite(single_image_output_path, image)

    # Get image dimensions
    image_height, image_width = image.shape[:2]
    print(image_height, image_width)

    # Save bounding boxes to a text file in YOLO format
    txt_file_name = os.path.splitext(os.path.basename(image_file))[0] + '.txt'
    with open(os.path.join(labels_output_path, txt_file_name), 'w') as txt_file:
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

            # Normalize coordinates to be between 0 and 1
            x_center_normalized = (x_min + x_max) / 2 / image_width
            y_center_normalized = (y_min + y_max) / 2 / image_height
            width_normalized = (x_max - x_min) / image_width
            height_normalized = (y_max - y_min) / image_height

            # Write the class ID (assuming class 0) and normalized coordinates
            txt_file.write(f"0 {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n")

    print(f"Image min: {image.min()}, max: {image.max()}, shape: {image.shape}")

    # Draw bounding boxes on the image
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        # Draw a red rectangle for each bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Plot the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Image with Bounding Boxes")
    plt.axis('off')

    # Show the plot
    plt.show()


if __name__ == "__main__":

    data = "combined"  # aureus  dsb  mixed  breast subtilis neurips
    mode = "train"
    images_path, masks_path = get_dataset_path(data, mode)
    labels_output_path = f"data/train_{data}/labels"  # Path to save the txt files (create this directory)
    images_output_path = f"data/train_{data}/images"

    dataset = create_dataset(f"../{images_path}", f"../{masks_path}")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SegDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    os.makedirs(labels_output_path, exist_ok=True)
    os.makedirs(images_output_path, exist_ok=True)

    print(len(train_dataloader))

    for batch_index, batch in enumerate(train_dataloader):

        if batch is None or len(batch["image"]) == 0:
            print(f"Skipping empty batch {batch_index}.")
            continue

        plot_bboxes(
            ground_truth=batch["instance_gt_masks"][0],
            image=batch["image"][0],
            bounding_boxes=batch["bounding_boxes"][0]
        )

        save_labels(labels_output_path,images_output_path, str(batch["path"][0]), batch["image"][0], batch["bounding_boxes"][0])




