from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, AutoModel, AutoProcessor
from codebase.data.dataset import SegDataset, create_dataset, custom_collate_fn
from codebase.utils.visualize import calculate_bbox_accuracy, plot_bboxes
from codebase.utils.test_utils import get_yolo_bboxes, nms
from codebase.data.dataset import get_dataset_path
from codebase.models.unet_semantic_segmentation import predict_binary_mask
import pandas as pd
import glob


def get_detection_metrics(data, mode, yolo_weights_path, semantic_seg_model_path, input_type="images"):
    images_path, masks_path = get_dataset_path(data, mode)
    dataset = create_dataset(f"../{images_path}", f"../{masks_path}", preprocess=True, axis_norm=(0, 1))
    print("Acquiring images from " + data + " dataset.")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # Initialize SAMDataset and DataLoader
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    test_dataset = SegDataset(dataset=dataset, processor=processor)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    nms_iou_threshold = 0.5
    metrics = []

    for index, test_sample in enumerate(test_data):

        if input_type=="binary_masks":
            mask = test_sample['original_gt_masks'].squeeze(0).float()
            binary_mask = (mask > 0).float() # float32 tensor with values 0 or 255
            yolo_boxes, confs = get_yolo_bboxes(binary_mask, yolo_weights_path)

        elif input_type=="predicted_masks":
            binary_prediction, gt_binary_mask = predict_binary_mask(DEVICE, test_sample["image"].squeeze(0), test_sample['original_gt_masks'].float(), semantic_seg_model_path)
            yolo_boxes, confs = get_yolo_bboxes(binary_prediction, yolo_weights_path)

        else:
            yolo_boxes, confs = get_yolo_bboxes(test_sample["image"], yolo_weights_path)

        gt_bboxes = test_sample["bounding_boxes"].squeeze(0)
        keep_indices = nms(yolo_boxes, confs, iou_threshold=nms_iou_threshold)
        yolo_boxes_nms = yolo_boxes[keep_indices.long()]
        TP, FP, FN, precision, recall, f1, tp_indices, fp_indices = calculate_bbox_accuracy(gt_bboxes, yolo_boxes_nms,
                                                                                      iou_threshold=0.5)
        metrics.append({
            "image_id": index,
            "num_gt": len(gt_bboxes),
            "num_pred": len(yolo_boxes_nms),
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2)
        })

        plot_bboxes(test_sample["image"].squeeze(), gt_bboxes.cpu().numpy(), yolo_boxes_nms.cpu().numpy(), tp_indices,
                    fp_indices)

    # Create DataFrame
    df_metrics = pd.DataFrame(metrics)

    # Summary
    summary = {
        "mean_precision": df_metrics["precision"].mean(),
        "mean_recall": df_metrics["recall"].mean(),
        "mean_f1": df_metrics["f1"].mean()
    }

    print(df_metrics)
    print("\nSummary:")
    print(pd.Series(summary))


def predict_and_visualize(folder_path, model):
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(image_path)
        # Perform prediction on the image
        results = model.predict(source=image_path, save=False, save_txt=False)

        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

        # Get predictions
        boxes = results[0].boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)

        # Create a subplot with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Plot the original image
        axs[0].imshow(image_rgb)
        axs[0].axis('off')
        axs[0].set_title("Original Image")

        # Plot the image with bounding boxes
        axs[1].imshow(image_rgb)
        ax = axs[1]
        # Add bounding boxes to the second subplot
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            # Draw rectangle (red color, no label or confidence)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
        ax.axis('off')
        axs[1].set_title("Image with Bounding Boxes")
        plt.show()
        plt.close()


def train_model(name):
    model = YOLO("yolov8n.pt")  # yolov8x.pt
    results = model.train(data="mydata.yaml", epochs=300, project="runs", name=name)
    return results


def plot_yolo_with_masks(images_dir, masks_dir, labels_dir, image_ext=".jpg", mask_ext=".png"):
    print("Directory exists:", os.path.isdir(images_dir))
    print("Files in dir:", os.listdir(images_dir))

    image_paths = sorted(glob.glob(f"{images_dir}/*{image_ext}"))


    for image_path in image_paths:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(masks_dir, f"{filename}{mask_ext}")
        label_path = os.path.join(labels_dir, f"{filename}.txt")

        if not (os.path.exists(mask_path) and os.path.exists(label_path)):
            print(f"Missing mask or label for {filename}, skipping.")
            continue

        # Load image and mask
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image_with_boxes = image_rgb.copy()
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_with_boxes = mask_color.copy()

        height, width = image.shape[:2]

        # Read YOLO labels and draw boxes
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            class_id, x_center, y_center, w, h = map(float, line.strip().split())
            x_center *= width
            y_center *= height
            w *= width
            h *= height

            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            for canvas in [image_with_boxes, mask_with_boxes]:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(canvas, str(int(class_id)), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Plot all 4
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(image_rgb)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title("Mask")
        axs[1].axis("off")

        axs[2].imshow(image_with_boxes)
        axs[2].set_title("Image with Boxes")
        axs[2].axis("off")

        axs[3].imshow(mask_with_boxes)
        axs[3].set_title("Mask with Boxes")
        axs[3].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    plot_yolo_with_masks(
        images_dir="data/train_dsb18/images",
        masks_dir="data/train_dsb18/masks/images",
        labels_dir="data/train_dsb18/masks/labels",
        image_ext=".tiff",  # or .png
        mask_ext=".tif"
    )