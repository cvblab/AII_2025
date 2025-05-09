import glob
import tifffile
import numpy as np
from PIL import Image
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import CenterCrop
import torch
import cv2
import os
from csbdeep.utils import normalize
#from Stardist.stardist import fill_label_holes
from codebase.Stardist.stardist import fill_label_holes
from tqdm import tqdm
import matplotlib.pyplot as plt



def preprocess_data(images, masks, axis_norm=(0, 1)):
    """
    Normalize images and fill label holes in masks.
    """
    n_channel = 1 if images[0].ndim == 2 else images[0].shape[-1]
    if n_channel > 1:
        print(
            "Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

    # Normalize images and fill holes in masks
    images = [normalize(img, 1, 99.8, axis=axis_norm) for img in tqdm(images, desc="Normalizing Images")]
    masks = [fill_label_holes(mask) for mask in tqdm(masks, desc="Filling Mask Holes")]

    return images, masks



def augment_dataset_with_original(dataset, augmenter):
    """
    Apply augmentations to the Dataset object and retain original samples.

    Parameters:
        dataset (Dataset): Hugging Face Dataset object containing images and masks.
        augmenter (function): Augmentation function.

    Returns:
        Dataset: A new dataset containing both original and augmented samples.
    """
    augmented_data = {"image": [], "label": [], "path": []}

    for sample in dataset:
        img, mask = np.array(sample["image"]), np.array(sample["label"])

        # Add original sample
        augmented_data["image"].append(sample["image"])  # Original image as PIL
        augmented_data["label"].append(sample["label"])  # Original label as PIL
        augmented_data["path"].append(sample["path"])  # Original path

        # Add augmented sample
        aug_img, aug_mask = augmenter(img, mask)

        # Ensure augmented path has the correct extension
        base, ext = os.path.splitext(sample["path"])
        augmented_data["image"].append(Image.fromarray(aug_img))  # Augmented image
        augmented_data["label"].append(Image.fromarray(aug_mask))  # Augmented label
        augmented_data["path"].append(base + "_aug" + ext)  # Append "_aug" before the extension

    return Dataset.from_dict(augmented_data)

def random_fliprot(img, mask):
    """
    Apply random flips and rotations to image and mask.
    """
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)

    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    """
    Apply random intensity changes to the image.
    """
    return img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)


def augmenter(x, y):
    """
    Augment a single input/label image pair.
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y

def get_bounding_boxes(instance_mask):
    # Find contours of the instance mask
    contours, _ = cv2.findContours(instance_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box
        bounding_boxes.append([x, y, x + w, y + h])  # Append in [x_min, y_min, x_max, y_max] format

    return bounding_boxes


def create_input_points(grid_size, array_size):
    x = np.linspace(0, array_size - 1, grid_size)
    y = np.linspace(0, array_size - 1, grid_size)
    xv, yv = np.meshgrid(x, y)
    return torch.tensor([[int(x), int(y)] for x, y in zip(xv.flatten(), yv.flatten())]).view(1, 1, -1, 2)


def process_image_patch(image_patch):
    if image_patch.ndim == 2:
        image_patch = np.stack([image_patch] * 3, axis=-1)
    if image_patch.dtype == np.uint16:
        image_patch = (image_patch / 65535.0 * 255).astype(np.uint8)
    return Image.fromarray(image_patch)