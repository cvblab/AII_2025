import torch
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, AutoModel, AutoProcessor
from codebase.data.dataset import create_dataset, SegDataset, custom_collate_fn, get_dataset_path
from models.sam import train_sam
from models.unettorch import train_unet
from models.stardist import train_stardist
from models.unet_semantic_segmentation import train_semantic_seg

if __name__ == "__main__":
    print(torch.version.cuda)
    print("torch version:", torch.__version__)

    data = "dsb" # aureus  dsb  mixed  breast subtilis
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
    train_data = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    model_type = "semantic"  # or "unet", "stardist" "sam"
    num_epochs = 50
    threshold = 0.7
    output_path = "../logs/training/" + model_type + "/" + data

    if model_type == "sam": # base large huge
        train_sam(DEVICE, train_data, num_epochs, threshold, backbone="base", output_path=output_path)
    elif model_type == "unett":
        train_unet(DEVICE, train_data, num_epochs, threshold, output_path=output_path)
    elif model_type == "stardist":
        train_stardist(DEVICE, train_data, num_epochs, threshold, output_path=output_path)
    elif model_type == "semantic":
        train_semantic_seg(DEVICE, train_data, num_epochs, output_path=output_path)
