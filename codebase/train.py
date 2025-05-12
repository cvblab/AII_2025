import torch
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, AutoModel, AutoProcessor
from codebase.data.dataset import create_dataset, SegDataset, custom_collate_fn
from models.sam import train_sam
from models.unet import train_unet
from models.unettorch import train_unett
from models.stardist import train_stardist

if __name__ == "__main__":
    print(torch.version.cuda)
    print("torch version:", torch.__version__)

    data = "aureus" # fluo  dsb  mixed  breast
    if data == "dsb":
        images_path = "../datasets/dsb2018/train/images/*.tif"
        masks_path = "../datasets/dsb2018/train/masks/*.tif"

    elif data == "aureus":
        images_path = "../datasets/aureus/train/patches/fluorescence/*.tif"
        masks_path = "../datasets/aureus/train/patches/masks/*.tif"

    elif data == "mixed":
        images_path = "../datasets/mixed_dataset/training/source/*.tif"
        masks_path = "../datasets/mixed_dataset/training/target/*.tif"

    elif data == "breast":
        images_path = "../datasets/breast_cancer/train/images/*.tif"
        masks_path = "../datasets/breast_cancer/train /masks/*.tif"

    elif data == "subtilis":
        images_path = "../datasets/subtilis/train/fluorescence/*.png"
        masks_path = "../datasets/subtilis/train/masks/*.png"

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

    model_type = "sam"  # or "unet", "stardist" "sam"
    num_epochs = 20
    threshold = 0.7
    output_path = "../logs/training/" + data + "/" + model_type

    if model_type == "sam": # base large huge
        train_sam(DEVICE, train_data, num_epochs, threshold, backbone="base", output_path=output_path)
    elif model_type == "unet":
        train_unet(DEVICE, train_data, num_epochs, threshold, output_path=output_path)
    elif model_type == "unett":
        train_unett(DEVICE, train_data, num_epochs, threshold, output_path=output_path)
    elif model_type == "stardist":
        train_stardist(DEVICE, train_data, num_epochs, threshold, output_path=output_path)
