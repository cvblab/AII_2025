import glob
import tifffile
import numpy as np
from PIL import Image
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import CenterCrop
import torch
from .preprocess import get_bounding_boxes, preprocess_data
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt



def get_dataset_path(data, mode):
    if os.path.exists("/workspace/cell_segmentation/datasets"):
        # Running inside Docker
        base_dataset_path = "/workspace/cell_segmentation/datasets"
    else:
        # Running locally (venv)
        base_dataset_path = "../datasets"

    if data == "dsb":
        images_path = os.path.join(base_dataset_path, f"dsb2018/{mode}/images/*.tif")
        masks_path = os.path.join(base_dataset_path, f"dsb2018/{mode}/masks/*.tif")

    elif data == "aureus":
        images_path = os.path.join(base_dataset_path, f"aureus/{mode}/fluorescence/*.tif")
        masks_path = os.path.join(base_dataset_path, f"aureus/{mode}/masks/*.tif")

    elif data == "mixed":
        images_path = os.path.join(base_dataset_path, f"mixed_dataset/{mode}/source/*.tif")
        masks_path = os.path.join(base_dataset_path, f"mixed_dataset/{mode}/target/*.tif")

    elif data == "breast":
        images_path = os.path.join(base_dataset_path, f"breast_cancer/{mode}/images/*.tif")
        masks_path = os.path.join(base_dataset_path, f"breast_cancer/{mode}/masks/*.tif")

    elif data == "subtilis":
        images_path = os.path.join(base_dataset_path, f"subtilis/{mode}/fluorescence/*.png")
        masks_path = os.path.join(base_dataset_path, f"subtilis/{mode}/masks/*.png")

    elif data == "neurips":
        images_path = os.path.join(base_dataset_path, f"neurips/{mode}/images/*")
        masks_path = os.path.join(base_dataset_path, f"neurips/{mode}/labels/*")

    else:
        raise ValueError(f"Unknown dataset: {data}")

    return images_path, masks_path



def get_model_paths(data):
    if os.path.exists("/workspace/cell_segmentation/datasets"):
        # Running inside Docker
        base_logs_path = "/workspace/cell_segmentation/logs"
        base_models_path = "/workspace/cell_segmentation/codebase"
    else:
        # Running locally (venv)
        base_logs_path = "../logs"
        base_models_path = "../codebase"

    instance_seg_model_path = os.path.join(base_logs_path, "training", "sam_old", "sam_model_dsb_best.pth")
    semantic_seg_model_path = os.path.join(base_logs_path, "training", "semantic2", data, "best_unet.pth")
    yolo_path = os.path.join(base_logs_path, "training", "yolo", f"yolov8_{data}", "weights", "best.pt")
    #yolo_path = os.path.join(base_logs_path, "training", "yolo", f"yolov8_dsb", "weights", "best.pt")
    cellpose_path = os.path.join(base_models_path, "models", "cellpose", f"cellpose_{data}")

    return instance_seg_model_path, semantic_seg_model_path, yolo_path, cellpose_path



def plot_images_and_masks(images, masks, max_samples=10):
    n = min(len(images), len(masks), max_samples)
    plt.figure(figsize=(10, 2 * n))

    for i in range(n):
        img = images[i]
        mask = masks[i]

        # Normalize image to [0,1] based on min and max per image
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val - min_val > 1e-8:
            img_norm = (img - min_val) / (max_val - min_val)
        else:
            img_norm = np.zeros_like(img)

        # Convert grayscale to RGB for plotting
        if img_norm.ndim == 2:
            img_rgb = np.stack([img_norm] * 3, axis=-1)
        else:
            img_rgb = img_norm  # already 3-channel?

        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Image {i}")
        plt.axis("off")

        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(mask, cmap="nipy_spectral")
        plt.title(f"Mask {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



def normalize_image(img):
    img = img.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    denom = max_val - min_val
    if denom < 1e-8:
        return np.zeros_like(img)
    else:
        return (img - min_val) / denom



def create_dataset(images_path, masks_path, preprocess=True, axis_norm=(0, 1)):
    images, masks, paths = load_data(images_path, masks_path)
    images = [normalize_image(img) for img in images]
    # Preprocess the images and masks
    if preprocess:
        images, masks = preprocess_data(images, masks, axis_norm)

    plot_images_and_masks(images, masks)

    dataset_dict = {
        "image": [Image.fromarray((img * 255).astype(np.uint8)) for img in images],
        "label": [Image.fromarray(np.uint8(mask)) for mask in masks],
        "path": paths,
    }

    # Create the Dataset object from the dictionary
    dataset = Dataset.from_dict(dataset_dict)

    return dataset



def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff']:
        img = tifffile.imread(path)
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        img = imageio.imread(path)
    else:
        print(f"Skipping unsupported file: {path}")
        return None

    # Convert RGB to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    return img.astype(np.float32)



def load_data(images_path, masks_path):
    image_files = sorted(glob.glob(images_path))
    mask_files = sorted(glob.glob(masks_path))

    filtered_images = []
    filtered_image_files = []
    for path in image_files:
        img = read_image(path)
        if img is not None:
            filtered_images.append(img)
            filtered_image_files.append(path)

    filtered_masks = []
    filtered_mask_files = []
    for path in mask_files:
        img = read_image(path)
        if img is not None:
            filtered_masks.append(img.astype(np.int32))
            filtered_mask_files.append(path)

    return filtered_images, filtered_masks, filtered_image_files



class SegDataset(TorchDataset):
    """
    This class creates a dataset that serves input images and individual masks for instance segmentation.
    """
    def __init__(self, dataset, processor, crop_size=256):
        self.dataset = dataset
        self.processor = processor
        self.crop_size = crop_size
        self.cropper = CenterCrop((self.crop_size, self.crop_size))  # Center cropping to 256x256

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = np.array(item["image"]).astype(np.float32) / 255.0  # Assumes image was saved as uint8
        mask = np.array(item["label"])
        path = item["path"]

        # Convert grayscale to RGB if needed
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)  # HWC

        # Crop both image and mask
        if image.shape[0] > self.crop_size or image.shape[1] > self.crop_size:
            image_pil = Image.fromarray((image * 255).astype(np.uint8))  # convert to PIL for cropping
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            image = np.array(self.cropper(image_pil)).astype(np.float32) / 255.0
            mask = np.array(self.cropper(mask_pil))

        # Check final shape
        if image.shape != (self.crop_size, self.crop_size, 3):
            print(f"[INFO] Skipping image with invalid shape: {image.shape}, path: {path}")
            return None

        # Process instance masks and bounding boxes
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]
        if len(object_ids) == 0:
            print(f"[INFO] No objects found in mask: {path}")
            return None

        instance_masks = []
        bounding_boxes = []
        for obj_id in object_ids:
            instance_mask = (mask == obj_id).astype(np.uint8)
            instance_masks.append(instance_mask)
            for bbox in get_bounding_boxes(instance_mask):
                if len(bbox) == 4:
                    bounding_boxes.append(np.array(bbox))
                else:
                    print(f"[WARNING] Invalid bounding box for object {obj_id}: {bbox}")

        instance_masks = np.stack(instance_masks, axis=0)

        # Sanity check normalized image
        if not np.all(np.isfinite(image)):
            print(f"[WARNING] NaN or Inf found in image after normalization. Skipping: {path}")
            return None

        # Prepare SAM inputs
        nested_bboxes = [bbox.flatten().tolist() for bbox in bounding_boxes]
        try:
            inputs = self.processor(image, input_boxes=[nested_bboxes], return_tensors="pt", do_rescale=False)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        except Exception as e:
            print(f"[ERROR] Processor failed on sample {path}: {e}")
            return None

        return {
            "image_float": torch.tensor(image),  # float32 [0,1]
            "image_uint8": torch.tensor((image * 255).astype(np.uint8)),  # uint8 image
            "original_mask": torch.tensor(mask, dtype=torch.float32),
            "single_instance_masks": torch.tensor(instance_masks, dtype=torch.float32),
            "bounding_boxes": torch.tensor(np.array(bounding_boxes)),
            "path": path,
            **inputs
        }



def custom_collate_fn(batch):
    # Remove None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    # Stack fixed-size tensors
    images_float = torch.stack([item['image_float'] for item in batch])
    images_uint8 = torch.stack([item['image_uint8'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    original_masks = torch.stack([item['original_mask'] for item in batch])
    paths = [item['path'] for item in batch]

    # Handle variable-sized instance masks
    num_objects_per_image = [item['single_instance_masks'].shape[0] for item in batch]
    max_num_masks = max(num_objects_per_image)

    padded_masks = []
    for item in batch:
        masks = item['single_instance_masks']
        num_masks = masks.shape[0]
        if num_masks < max_num_masks:
            pad = torch.zeros((max_num_masks - num_masks, *masks.shape[1:]), dtype=masks.dtype)
            masks = torch.cat([masks, pad], dim=0)
        padded_masks.append(masks)
    padded_masks = torch.stack(padded_masks)  # [B, max_instances, H, W]

    # Handle bounding boxes
    max_num_boxes = max(len(item['bounding_boxes']) for item in batch)
    padded_boxes = []
    for item in batch:
        boxes = item['bounding_boxes']
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)  # Handle single box case
        num_boxes = boxes.shape[0]
        if num_boxes < max_num_boxes:
            pad = torch.zeros((max_num_boxes - num_boxes, 4), dtype=boxes.dtype)
            boxes = torch.cat([boxes, pad], dim=0)
        padded_boxes.append(boxes)
    padded_boxes = torch.stack(padded_boxes)  # [B, max_boxes, 4]

    return {
        'pixel_values': pixel_values,
        'image': images_float,
        'image_uint8': images_uint8,
        'original_gt_masks': original_masks,
        'bounding_boxes': padded_boxes,
        'instance_gt_masks': padded_masks,
        'num_objects_per_image': num_objects_per_image,
        'path': paths,
    }
