import torch
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, AutoModel, AutoProcessor
from codebase.data.dataset import create_dataset, SegDataset, custom_collate_fn, get_dataset_path
from models.sam import train_sam
from models.unet import train_unet
from models.stardist import train_stardist
from models.unet_semantic_segmentation import train_semantic_seg
from models.cellpose_model import train_cellpose
from codebase.utils.visualize import plot_imgs
import os

if __name__ == "__main__":
    print(os.getcwd())
    print("torch version:", torch.__version__)
    print(torch.cuda.is_available())  # True
    print(torch.version.cuda)  # '12.5'

    data = "neurips" # aureus  dsb  mixed  breast subtilis neurips
    mode = "train"
    images_path, masks_path = get_dataset_path(data, mode)
    dataset = create_dataset(images_path, masks_path, preprocess=True, axis_norm=(0, 1))
    print("Acquiring images from "+ data + " dataset.")
    #augmented_dataset = augment_dataset_with_original(dataset, augmenter=augmenter).shuffle(seed=42)
    #plot_augmented_dataset(augmented_dataset)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # Initialize SAMDataset and DataLoader
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SegDataset(dataset=dataset, processor=processor)
    train_data = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    #plot_imgs(train_data)
    model_type = ""  # or "unet", "stardist" "sam"
    num_epochs = 50
    threshold = 0.7

    if os.path.exists("/workspace/cell_segmentation/datasets"):
        # Running inside Docker
        base_output_path = "/workspace/cell_segmentation/logs"
    else:
        # Running locally (venv)
        base_output_path = "../logs"

    instance_seg_model_path = os.path.join(base_output_path, "training", model_type, data)
    semantic_seg_model_path = os.path.join(base_output_path, "training", model_type, data)
    output_path = os.path.join(base_output_path, "training", model_type, data)

    if model_type == "sam": # base large huge
        train_sam(DEVICE, train_data, num_epochs, threshold, backbone="base", output_path=output_path)
    elif model_type == "unet":
        train_unet(DEVICE, train_data, num_epochs, threshold, output_path=output_path)
    elif model_type == "stardist":
        train_stardist(DEVICE, train_data, num_epochs, threshold, output_path=output_path)
    elif model_type == "semantic":
        train_semantic_seg(DEVICE, train_data, num_epochs, output_path=output_path)
    elif model_type == "cellpose":
        train_cellpose(DEVICE, train_data, num_epochs, data)
