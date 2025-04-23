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
from csbdeep.utils import normalize
from Stardist.stardist import fill_label_holes
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_dataset(images_path, masks_path, preprocess=True, axis_norm=(0, 1)):
    images, masks, paths = load_data(images_path, masks_path)

    # Preprocess the images and masks
    if preprocess:
        images, masks = preprocess_data(images, masks, axis_norm)

    # Create a dictionary with images and masks as PIL images
    dataset_dict = {
        "image": [Image.fromarray(img) for img in images],
        "label": [Image.fromarray(mask) for mask in masks],
        "path": paths,
    }

    # Create the Dataset object from the dictionary
    dataset = Dataset.from_dict(dataset_dict)

    return dataset

def load_data(images_path, masks_path):
    # Use glob to get sorted lists of image and mask file paths
    image_files = sorted(glob.glob(images_path))
    mask_files = sorted(glob.glob(masks_path))

    # Read the images and masks using tifffile
    images = [tifffile.imread(img) for img in image_files]  # Store images in a list
    masks = [tifffile.imread(mask) for mask in mask_files]  # Store masks in a list

    return images, masks, image_files


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

        # Check if the image is grayscale and convert it to RGB
        if image.ndim == 2:  # Image is grayscale
            image = np.expand_dims(image, axis=-1)  # Expand dimensions to (H, W, 1)
            image = np.repeat(image, 3, axis=2)  # Repeat the grayscale values across the new channel dimension

        # If the image is larger than crop size, crop it at the center

        if image.shape[0] > self.crop_size or image.shape[1] > self.crop_size:

            if image.dtype != np.uint8:
                # image = np.array(image, dtype=np.float32)  # Convert to float to prevent truncation
                # image = np.clip(image / 256, 0, 255).astype(np.uint8)  # Normalize and convert to uint8
                image = np.clip(image, 0, 1.0) * 255.0
                image = image.astype(np.uint8)

            image = self.cropper(Image.fromarray(image))  # Center crop image
            mask = self.cropper(Image.fromarray(mask))  # Center crop mask
            image = np.array(image)
            mask = np.array(mask)

        # Generate individual instance masks (assuming objects are labeled with unique values)
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]
        instance_masks = []
        bounding_boxes = []

        for obj_id in object_ids:
            instance_mask = np.where(mask == obj_id, 1, 0)  # Create binary mask for each object
            instance_masks.append(instance_mask)

            # Get bounding box for the object
            bounding_box_list = get_bounding_boxes(instance_mask)

            for bbox in bounding_box_list:
                # Only append if the bounding box has exactly 4 coordinates
                if len(bbox) == 4:
                    if isinstance(bbox, list):
                        bbox = np.array(bbox)
                    bounding_boxes.append(bbox)
                else:
                    print(f"Unexpected bounding box format for object {obj_id}: {bbox}")  # Debug print

        # Convert instance_masks to a tensor of shape [num_objects, H, W]
        instance_masks = np.stack(instance_masks, axis=0)  # Shape: [num_objects, H, W]

        # Normalize the image to the range 0-255
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
        threshold = 50  # Tune this value
        image[image < threshold] = 0
        image = image / np.max(image)

        nested_bounding_boxes = [bbox.flatten().tolist() for bbox in bounding_boxes]  # List of boxes for each object

        inputs = self.processor(image, input_boxes=[nested_bounding_boxes], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return {
            "image": torch.tensor(image),  # Original image
            "original_mask": torch.tensor(mask, dtype=torch.float32),
            "single_instance_masks": torch.tensor(instance_masks, dtype=torch.float32),  # Instance masks
            "bounding_boxes": torch.tensor(np.array(bounding_boxes)),
            "path": path,
            **inputs
        }


def custom_collate_fn(batch):
    # Collate images
    images = torch.stack([item['image'] for item in batch])
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
        'original_gt_masks': original_masks,
        'bounding_boxes': padded_boxes,
        'instance_gt_masks': padded_masks,
        'num_objects_per_image': num_objects_per_image,
        'path':path# New field to track number of objects per image
    }