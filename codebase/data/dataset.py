import glob
import tifffile
import numpy as np
from PIL import Image
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import CenterCrop
import torch
from .preprocess import get_bounding_boxes, preprocess_data
import cv2
import os
import imageio.v2 as imageio  # imageio.v2 avoids deprecation warnings
from csbdeep.utils import normalize
#from Stardist.stardist import fill_label_holes
from codebase.Stardist.stardist import fill_label_holes
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_dataset_path(data, mode):

    if data == "dsb":
        images_path = f"../datasets/dsb2018/{mode}/images/*.tif"
        masks_path = f"../datasets/dsb2018/{mode}/masks/*.tif"

    elif data == "aureus":
        images_path = f"../datasets/aureus/{mode}/fluorescence/*.tif"
        masks_path = f"../datasets/aureus/{mode}/masks/*.tif"

    elif data == "mixed":
        images_path = f"../datasets/mixed_dataset/{mode}/source/*.tif"
        masks_path = f"../datasets/mixed_dataset/{mode}/target/*.tif"

    elif data == "breast":
        images_path = f"../datasets/breast_cancer/{mode}/images/*.tif"
        masks_path = f"../datasets/breast_cancer/{mode}/masks/*.tif"

    elif data == "subtilis":
        images_path = f"../datasets/subtilis/{mode}/fluorescence/*.png"
        masks_path = f"../datasets/subtilis/{mode}/masks/*.png"

    elif data == "neurips":
        images_path = f"../datasets/neurips/{mode}/images/*"
        masks_path = f"../datasets/neurips/{mode}/labels/*"

    return images_path, masks_path


def create_dataset(images_path, masks_path, preprocess=True, axis_norm=(0, 1)):
    images, masks, paths = load_data(images_path, masks_path)

    # Preprocess the images and masks
    if preprocess:
        images, masks = preprocess_data(images, masks, axis_norm)

    # Create a dictionary with images and masks as PIL images
    # dataset_dict = {
    #     "image": [Image.fromarray(img) for img in images],
    #     "label": [Image.fromarray(mask) for mask in masks],
    #     "path": paths,
    # }

    dataset_dict = {
        "image": [Image.fromarray(np.uint8(img)) for img in images],
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
        image = np.array(item["image"])
        mask = np.array(item["label"])
        path = np.array(item["path"])

        # Convert grayscale to RGB
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=2)

        # Crop if necessary
        if image.shape[0] > self.crop_size or image.shape[1] > self.crop_size:
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 1.0) * 255.0
                image = image.astype(np.uint8)

            image = self.cropper(Image.fromarray(image))
            mask = self.cropper(Image.fromarray(mask))
            image = np.array(image)
            mask = np.array(mask)

        if image.shape != (self.crop_size, self.crop_size, 3):
            print(f"[INFO] Skipping image with invalid shape: {image.shape}, path: {path}")
            return None

        # Handle instance masks and bounding boxes
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]
        if len(object_ids) == 0:
            print(f"[INFO] No objects found in mask: {path}")
            return None

        instance_masks = []
        bounding_boxes = []
        for obj_id in object_ids:
            instance_mask = np.where(mask == obj_id, 1, 0)
            instance_masks.append(instance_mask)

            for bbox in get_bounding_boxes(instance_mask):
                if len(bbox) == 4:
                    bounding_boxes.append(np.array(bbox))
                else:
                    print(f"Unexpected bounding box format for object {obj_id}: {bbox}")

        instance_masks = np.stack(instance_masks, axis=0)

        # Convert image to float32 for normalization and processing
        image_float = image.astype(np.float32)

        # Check for NaNs or Infs
        if not np.all(np.isfinite(image_float)):
            print(f"[WARNING] Image has NaN or Inf before normalization. Skipping: {path}")
            return None

        min_val = image_float.min()
        max_val = image_float.max()
        denominator = max_val - min_val

        if denominator < 1e-8:
            image_norm = np.zeros_like(image_float)
        else:
            image_norm = (image_float - min_val) / denominator

        # Sanity check normalized image
        if not np.all(np.isfinite(image_norm)):
            print(f"[WARNING] Image has NaN or Inf after normalization. Skipping: {path}")
            return None

        # Prepare inputs for SAM processor using normalized image
        nested_bounding_boxes = [bbox.flatten().tolist() for bbox in bounding_boxes]
        try:
            inputs = self.processor(image_norm, input_boxes=[nested_bounding_boxes], return_tensors="pt", do_rescale=False)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        except Exception as e:
            print(f"[ERROR] Processor failed on sample {path}: {e}")
            return None

        return {
            "image_norm": torch.tensor(image_norm),  # Normalized float image [0,1]
            "image_uint8": torch.tensor(image, dtype=torch.uint8),  # Original image uint8 [0,255]
            "original_mask": torch.tensor(mask, dtype=torch.float32),
            "single_instance_masks": torch.tensor(instance_masks, dtype=torch.float32),
            "bounding_boxes": torch.tensor(np.array(bounding_boxes)),
            "path": path,
            **inputs
        }



def custom_collate_fn(batch):
    # Collate images
    batch = [item for item in batch if item is not None]

    # If after removing None, the batch is empty, return None (skip this batch)
    if len(batch) == 0:
        return None

    images = torch.stack([item['image_norm'] for item in batch])
    images_original = torch.stack([item['image_uint8'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    original_masks = torch.stack([item['original_mask'] for item in batch])
    path = [item['path'] for item in batch]

    # Track the number of objects (masks) per image
    num_objects_per_image = [item['single_instance_masks'].shape[0] for item in batch]

    # Get the maximum number of masks in the batch
    max_num_masks = max(num_objects_per_image)

    # Pad masks with zeros to match the max number of instance masks
    padded_masks = []
    for item in batch:
        masks = item['single_instance_masks']
        num_masks = masks.shape[0]

        # If there are fewer masks than the max, pad with zeros
        if num_masks < max_num_masks:
            padding = torch.zeros((max_num_masks - num_masks, masks.shape[1], masks.shape[2]), dtype=masks.dtype)
            masks = torch.cat([masks, padding], dim=0)

        padded_masks.append(masks)

    # Stack the padded masks
    padded_masks = torch.stack(padded_masks)  # Shape: [batch_size, max_num_masks, H, W]

    # Handle bounding boxes (prompt)
    max_num_boxes = max(len(item['bounding_boxes']) for item in batch)
    padded_boxes = []

    for item in batch:
        boxes = item['bounding_boxes']  # Assuming boxes are in a list
        boxes = torch.tensor(boxes, dtype=torch.float32) if isinstance(boxes, list) else boxes.clone().detach()  # Ensure boxes are a tensor
        num_boxes = boxes.shape[0]

        # Pad boxes with zeros to match the max number of boxes
        if num_boxes < max_num_boxes:
            padding = torch.zeros((max_num_boxes - num_boxes, 4), dtype=boxes.dtype)
            boxes = torch.cat([boxes, padding], dim=0)

        padded_boxes.append(boxes)

    # Stack padded boxes
    padded_boxes = torch.stack(padded_boxes)  # Shape: [batch_size, max_num_boxes, 4]

    # Return the batched data, including the number of objects
    return {
        'pixel_values': pixel_values,
        'image': images,
        'original_image':images_original,
        'original_gt_masks': original_masks,
        'bounding_boxes': padded_boxes,
        'instance_gt_masks': padded_masks,
        'num_objects_per_image': num_objects_per_image,
        'path':path# New field to track number of objects per image
    }