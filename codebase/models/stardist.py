from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from csbdeep.utils import normalize
from codebase.Stardist.stardist import gputools_available
from codebase.Stardist.stardist.models import Config2D, StarDist2D
from codebase.Stardist.stardist import random_label_cmap
from codebase.utils.metrics import calculate_metrics,  average_precision
import pandas as pd

lbl_cmap = random_label_cmap()

def split_data(X, Y, val_split=0.15):
    """
    Split the datasets into training and validation sets.
    """
    assert len(X) > 1, "Not enough training datasets"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(val_split * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

    print('Number of images: %3d' % len(X))
    print('- Training:       %3d' % len(X_trn))
    print('- Validation:     %3d' % len(X_val))

    return X_trn, Y_trn, X_val, Y_val


def label_to_binary_masks(label_image):

    ids = np.unique(label_image)
    ids = ids[ids != 0]  # remove background
    masks = np.stack([(label_image == i).astype(np.uint8) for i in ids], axis=0)
    return masks  # Shape: (num_objects, H, W)


def train_stardist(DEVICE, train_data, num_epochs, threshold, output_path):
    """
    Train the model with the given training and validation datasets.
    """
    all_images = []
    all_masks = []
    for batch in train_data:
        for item_index in range(len(batch["image"])):
            # Get image and convert to numpy
            image = batch["image"][item_index].cpu().numpy()
            image = image.mean(axis=-1, keepdims=True)
            all_images.append(image)

            gt_masks = batch["original_gt_masks"][item_index].cpu().numpy().astype(np.uint16)
            all_masks.append(gt_masks)

    X_trn, Y_trn, X_val, Y_val = split_data(all_images, all_masks)

    use_gpu = str(DEVICE).startswith("cuda") and gputools_available()

    conf = Config2D(
        n_rays=32,
        grid=(2, 2),
        use_gpu=use_gpu,
        n_channel_in=1,
    )

    model = StarDist2D(conf, name="stardist",basedir=output_path)
    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), epochs=200, steps_per_epoch=24)

    return model

def test_stardist(test_data, tp_thresholds):
    model = StarDist2D(None, name="stardist_fluorescence", basedir="../weights/")
    #model = StarDist2D.from_pretrained('2D_paper_dsb2018')
    all_aps_per_threshold = {threshold: [] for threshold in tp_thresholds}

    for index, test_sample in enumerate(test_data):
        img = test_sample["image"].squeeze()
        grayscale_image = img.permute(2, 0, 1).float().mean(dim=0)
        axis_norm = (0, 1)  # normalize channels independently
        image = normalize(grayscale_image.numpy(), 1, 99.8, axis=axis_norm)
        gt_mask = test_sample['original_gt_masks'].squeeze().numpy()

        labels, details = model.predict_instances(image)

        print("Detected cells:", len(np.unique(labels)))
        print("Groundtruth cells:", len(np.unique(gt_mask)))

        # Collect all unique label IDs from both masks
        all_labels = np.unique(np.concatenate((np.unique(test_sample['original_gt_masks'].squeeze().numpy()), np.unique(labels))))
        num_labels = all_labels.max() + 1  # Ensure we include all label indices

        # Create a consistent colormap
        cmap = plt.get_cmap('nipy_spectral', num_labels)

        # Plot side-by-side comparison with shared colormap
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        im0 = axes[0].imshow(gt_mask, cmap=cmap, vmin=0, vmax=num_labels - 1)
        axes[0].set_title('Ground Truth Mask')
        axes[0].axis('off')

        im1 = axes[1].imshow(labels, cmap=cmap, vmin=0, vmax=num_labels - 1)
        axes[1].set_title('Predicted Labels')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        det_masks = label_to_binary_masks(labels)
        gt_masks = label_to_binary_masks(gt_mask)

        for threshold in tp_thresholds:
            TP, FP, FN = calculate_metrics(det_masks, gt_masks,
                                           threshold=threshold)
            AP = average_precision(TP, FP, FN)
            all_aps_per_threshold[threshold].append(AP)
            print(f"Sample {index}, IoU Threshold: {threshold}, AP: {AP}, TP: {TP}, FP: {FP}, FN: {FN}")

        # Compute mean AP across all samples for each threshold
    mean_aps = {threshold: np.mean(aps) for threshold, aps in all_aps_per_threshold.items()}

    mean_ap_df = pd.DataFrame(list(mean_aps.items()), columns=["IoU Threshold", "Mean AP"])

    # Print the DataFrame
    print(mean_ap_df)