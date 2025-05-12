import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from codebase.data.dataset import get_dataset_path
import glob

def convert_instance_masks_to_binary(input_dir, output_dir, visualize=True):
    os.makedirs(output_dir, exist_ok=True)
    file_list = glob.glob(input_dir)
    for path in file_list:

        filename = os.path.basename(path)
        mask = np.array(Image.open(path))

        # Convert to binary: 0 for background, 1 for any object
        binary_mask = (mask > 0).astype(np.uint8) * 255  # Scale to 255 for saving/viewing

        # Save binary mask
        out_path = os.path.join(output_dir, filename)
        Image.fromarray(binary_mask).save(out_path)

        # Visualization
        if visualize:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(mask, cmap='nipy_spectral')
            axs[0].set_title("Original Instance Mask")
            axs[0].axis('off')

            axs[1].imshow(binary_mask, cmap='gray')
            axs[1].set_title("Binary Mask")
            axs[1].axis('off')

            plt.suptitle(filename)
            plt.tight_layout()
            plt.show()

    print(f"Binary masks saved to: {output_dir}")

if __name__ == "__main__":
    data = "breast"  # aureus  dsb  mixed  breast subtilis
    mode = "train"
    images_path, masks_path = get_dataset_path(data, mode)
    output_dir = f"../yolov8/data/train_{data}/masks"
    convert_instance_masks_to_binary(f"../{masks_path}", output_dir, visualize=True)