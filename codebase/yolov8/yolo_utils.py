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
import pandas as pd


def get_detection_metrics(data, weights_path):
    if data == "dsb":
        images_path = "../../datasets/dsb2018/test/images/*.tif"
        masks_path = "../../datasets/dsb2018/test/masks/*.tif"

    elif data == "fluo":
        images_path = "../../datasets/fluorescence_dataset/test/fluorescence/*.tif"
        masks_path = "../../datasets/fluorescence_dataset/test/masks/*.tif"

    elif data == "mixed":
        images_path = "../../datasets/mixed_dataset/test/source/*.tif"
        masks_path = "../../datasets/mixed_dataset/test/target/*.tif"

    elif data == "breast":
        images_path = "../../datasets/breast_cancer/test/images/*.tif"
        masks_path = "../../datasets/breast_cancer/test/masks/*.tif"

    dataset = create_dataset(images_path, masks_path, preprocess=True, axis_norm=(0, 1))
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
        gt_bboxes = test_sample["bounding_boxes"].squeeze(0)
        yolo_boxes, confs = get_yolo_bboxes(test_sample["image"], weights_path)
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
            "precision": precision,
            "recall": recall,
            "f1": f1
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