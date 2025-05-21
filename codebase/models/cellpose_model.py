import numpy as np
from cellpose import models, core, io, plot, train, metrics
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import torch
from PIL import Image
import numpy as np

# Setup logging to see Cellpose output
io.logger_setup()

# Check GPU access
if not core.use_gpu():
    raise RuntimeError("No GPU detected. Please make sure CUDA is available.")

def prepare_cellpose_folder(data, target_dir, mask_suffix="_mask"):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for batch_index, batch in enumerate(data):
        if batch is None or len(batch["image"]) == 0:
            print(f"Skipping empty batch {batch_index}.")
            continue

        for item_index in range(len(batch["image"])):
            image = batch['image'][item_index]
            mask = batch['original_gt_masks'][item_index]
            img_name = str(batch['path'][item_index]).split("\\")[-1]

            # Convert image to PIL and save
            if isinstance(image, torch.Tensor):
                # convert tensor (C,H,W) or (H,W,C) to numpy
                img_np = image.cpu().numpy()
                if img_np.shape[0] == 3 and img_np.shape[1] == 256:  # probably (C,H,W)
                    img_np = np.transpose(img_np, (1, 2, 0))
            else:
                img_np = np.array(image)


            img = (img_np * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img, mode='RGB')
            img_pil.save(target_dir / img_name)

            # Convert mask tensor to PIL and save
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)

            mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L")
            mask_name = f"{Path(img_name).stem}{mask_suffix}{Path(img_name).suffix}"
            mask_pil.save(target_dir / mask_name)

    print(f"Saved images and masks in {target_dir}")


# Train the model
def train_cellpose(DEVICE, train_data, num_epochs, data,  model_name="cellpose_custom_model"):
    learning_rate = 1e-5
    weight_decay = 0.1
    batch_size = 1
    model_name = f"cellpose/cellpose_{data}"
    train_dir = Path("tmp/cellpose_train")
    masks_ext = "_mask"
    prepare_cellpose_folder(train_data, train_dir, mask_suffix=masks_ext)
    output = io.load_train_test_data(str(train_dir),None,
                                     mask_filter=masks_ext)
    train_data, train_labels, _, test_data, test_labels, _ = output
    model = models.CellposeModel(gpu=True)
    new_model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        batch_size=batch_size,
        n_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        nimg_per_epoch=max(2, len(train_data)),
        model_name=model_name
    )

    shutil.rmtree(train_dir, ignore_errors=True)


# Evaluate on test data (optional)
def test_cellpose(DEVICE, test_data, model_path):
    test_dir = Path("tmp/cellpose_test")
    masks_ext = "_mask"
    prepare_cellpose_folder(test_data, test_dir, mask_suffix=masks_ext)
    output = io.load_train_test_data(str(test_dir), None,
                                     mask_filter=masks_ext)
    test_data, test_labels, _, _, _, _ = output

    # Reload trained model
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    masks = model.eval(test_data, batch_size=32)[0]
    ap = metrics.average_precision(test_labels, masks)[0]
    print(f'\n>>> Average Precision @ IoU 0.5: {ap[:, 0].mean():.3f}')

    # Plot predictions
    plt.figure(figsize=(12, 8), dpi=150)
    for k, img in enumerate(test_data):
        plt.subplot(3, len(test_data), k + 1)
        plt.imshow(img)
        plt.axis('off')
        if k == 0:
            plt.title('image')

        plt.subplot(3, len(test_data), len(test_data) + k + 1)
        plt.imshow(masks[k])
        plt.axis('off')
        if k == 0:
            plt.title('predicted labels')

        plt.subplot(3, len(test_data), 2 * len(test_data) + k + 1)
        plt.imshow(test_labels[k])
        plt.axis('off')
        if k == 0:
            plt.title('true labels')
    plt.tight_layout()
    plt.show()

    shutil.rmtree(test_dir, ignore_errors=True)
