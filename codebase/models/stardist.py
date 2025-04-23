from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D
from stardist.matching import matching_dataset

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

def train_stardist(DEVICE, train_data, num_epochs, threshold, output_path):
    """
    Train the model with the given training and validation datasets.
    """
    # Use OpenCL-based computations for datasets generator during training (requires 'gputools')

    all_images = []
    all_masks = []

    for batch in train_data:
        for item_index in range(len(batch["image"])):
            # Get image and convert to numpy
            image = batch["image"][item_index].cpu().numpy()
            image = image.mean(axis=-1, keepdims=True)
            all_images.append(image)

            # Get ground truth masks and convert to numpy
            num_objects = batch["num_objects_per_image"][item_index]
            gt_masks = batch["original_gt_masks"][item_index].cpu().numpy().astype(np.uint16)

            all_masks.append(gt_masks)

    X_trn, Y_trn, X_val, Y_val = split_data(all_images, all_masks)

    use_gpu = False and gputools_available()

    conf = Config2D(
        n_rays=32,
        grid=(2, 2),
        use_gpu=use_gpu,
        n_channel_in=1,
    )

    model = StarDist2D(conf, name="stardist",basedir=output_path)
    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), epochs=200, steps_per_epoch=24)

    return model