import torch
from torch.utils.data import DataLoader
from transformers import SamProcessor
from codebase.data.dataset import create_dataset, SegDataset, custom_collate_fn, get_dataset_path
from models.sam import train_sam
from models.unet import train_unet
from models.stardist import train_stardist
from models.unet_semantic_segmentation import train_semantic_seg
from models.cellpose_model import train_cellpose
import os


if __name__ == "__main__":
    print(os.getcwd())
    print("torch version:", torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    # choose dataset
    data = "combined" # aureus  dsb breast tcell flow_chamber combined (dsb, aureus and tcell)
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
    model_type = "cellpose"  # or "unet", "stardist" "sam", "cellpose"
    num_epochs = 50
    threshold = 0.7
    env = os.environ.get("ENV", "LOCAL").lower()

    if env == "docker":
        base_output_path = "/workspace/cell_segmentation/logs"
    else:
        base_output_path = "../logs"

    instance_seg_model_path = os.path.join(base_output_path, "training", model_type, data)
    semantic_seg_model_path = os.path.join(base_output_path, "training", model_type, data)
    output_path = os.path.join(base_output_path, "training", model_type, data)
    os.makedirs(output_path, exist_ok=True)

    if model_type == "sam": # base large huge
        train_sam(DEVICE, train_data, num_epochs, threshold, backbone="base", output_path=output_path)
    elif model_type == "stardist":
        train_stardist(DEVICE, train_data, num_epochs, threshold, output_path=output_path)
    elif model_type == "semantic":
        train_semantic_seg(DEVICE, train_data, num_epochs, output_path=output_path)
    elif model_type == "cellpose":
        train_cellpose(num_epochs, data, train_data)
