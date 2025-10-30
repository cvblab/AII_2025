import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import SamProcessor
from codebase.data.dataset import create_dataset, SegDataset, custom_collate_fn, get_dataset_path


def convert_instance_masks_to_binary(images_dir,masks_dir, output_dir, visualize=True):
    os.makedirs(output_dir, exist_ok=True)
    dataset = create_dataset(images_dir, masks_dir, preprocess=True, axis_norm=(0, 1))
    # Initialize SAMDataset and DataLoader
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    test_dataset = SegDataset(dataset=dataset, processor=processor)
    train_data = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    for index, train_sample in enumerate(train_data):
        if train_sample is None or len(train_sample["image"]) == 0:
            print(f"Skipping empty batch {index}.")
            continue

        image = train_sample["image"].squeeze(0)
        mask = train_sample['original_gt_masks'].squeeze(0).float()
        name = os.path.basename(str(train_sample['path'][0]))

        # Correct binary mask conversion
        binary_mask = ((mask > 0).byte().cpu().numpy()) * 255

        ext = os.path.splitext(name)[1].lower()
        if ext not in ['.tif', '.tiff']:
            name = os.path.splitext(name)[0] + '.tiff'  # change extension to .tiff

        # Save binary mask
        out_path = os.path.join(output_dir, name)
        Image.fromarray(binary_mask).save(out_path)

        # Visualization
        if visualize:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(image)
            axs[0].set_title("Original Image")
            axs[0].axis('off')

            axs[1].imshow(mask, cmap='nipy_spectral')
            axs[1].set_title("Instance Mask")
            axs[1].axis('off')

            axs[2].imshow(binary_mask, cmap='gray')
            axs[2].set_title("Binary Mask")
            axs[2].axis('off')

            plt.suptitle(name)
            plt.tight_layout()
            plt.show()

    print(f"Binary masks saved to: {output_dir}")


if __name__ == "__main__":
    data = "neurips"  # aureus  dsb  mixed  breast subtilis neurips
    mode = "train"
    images_path, masks_path = get_dataset_path(data, mode)
    output_dir = f"../yolov8/data/train_{data}/masks/images"
    convert_instance_masks_to_binary(f"../{images_path}", f"../{masks_path}", output_dir, visualize=True)
