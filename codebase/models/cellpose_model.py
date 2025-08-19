from cellpose import models, core, io, plot, train, metrics
from pathlib import Path
import shutil
import torch
from PIL import Image
import numpy as np
from codebase.utils.visualize import plot_ap,plot_instance_segmentation
from codebase.utils.test_utils import convert_masks_to_instances, save_results_to_excel
from codebase.utils.metrics import calculate_metrics,  average_precision
import pandas as pd
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
            img_name_raw = Path(batch['path'][item_index]).name
            img_name = img_name_raw.replace("_mask", "")
            img_stem = Path(img_name).stem

            # Convert image to PIL and save
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
                if img_np.shape[0] == 3 and img_np.shape[1] == 256:
                    img_np = np.transpose(img_np, (1, 2, 0))
            else:
                img_np = np.array(image)

            img = (img_np * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img, mode='RGB')
            img_pil.save(target_dir / f"{img_stem}.tif")

            # Convert mask tensor to PIL and save
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)

            mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L")
            mask_name = f"{img_stem}{mask_suffix}.tif"
            mask_pil.save(target_dir / mask_name)

    print(f"Saved images and masks in {target_dir}")




# Train the model
def train_cellpose(num_epochs, data, train_data):

    learning_rate = 1e-5
    weight_decay = 0.1
    batch_size = 2
    masks_ext = "_mask"

    model_dir = Path(f"../logs/training/cellpose/{data}")
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(model_dir)
    #train_data_dir = Path(f"/workspace/cell_segmentation/datasets/cellpose_data/{data}/train")
    train_data_dir = Path(f"../datasets/cellpose_data/{data}/train")#
    #train_data_dir.mkdir(parents=True, exist_ok=True)
    #prepare_cellpose_folder(train_data, train_data_dir, mask_suffix=masks_ext)

    output = io.load_train_test_data(str(train_data_dir),None,
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
        save_path=save_path,
        model_name="best.pt"
    )


# Evaluate on test data (optional)
def test_cellpose(DEVICE, test_data, tp_thresholds, model_path, data):
    #test_dir = Path(f"/workspace/cell_segmentation/datasets/cellpose_data/{data}/test")
    test_dir = Path(f"../datasets/cellpose_data/{data}/test")
    #test_dir.mkdir(parents=True, exist_ok=True)
    masks_ext = "_mask"
    prepare_cellpose_folder(test_data, test_dir, mask_suffix=masks_ext)
    output = io.load_train_test_data(str(test_dir), None,
                                     mask_filter=masks_ext)
    test_images, test_labels, _, _, _, _ = output

    # Reload trained model
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    #model = models.CellposeModel(gpu=True, model_type='nuclei')
    model = models.CellposeModel(gpu=True)

    pred_masks = model.eval(test_images, batch_size=32)[0]
    ap = metrics.average_precision(test_labels, pred_masks)[0]
    print(f'\n>>> Average Precision @ IoU 0.5: {ap[:, 0].mean():.3f}')

    gt_instances = convert_masks_to_instances(test_labels)  # List of (n_gt_objects, 256, 256)
    pred_instances = convert_masks_to_instances(pred_masks)

    all_aps_per_threshold = {threshold: [] for threshold in tp_thresholds}
    results = []

    for i in range(len(gt_instances)):

        for threshold in tp_thresholds:
            TP, FP, FN = calculate_metrics(pred_instances[i], gt_instances[i],
                                           threshold=threshold)
            AP = average_precision(TP, FP, FN)

            all_aps_per_threshold[threshold].append(AP)  # Store AP for this threshold
            print(f"Sample {i}, IoU Threshold: {threshold}, AP: {AP}, TP: {TP}, FP: {FP}, FN: {FN}")

            if threshold == 0.5:
                results.append({
                    "Sample": i,
                    "IoU Threshold": threshold,
                    "AP": AP,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN
                })

        plot_instance_segmentation(
            detections=pred_instances[i],
            ground_truth=gt_instances[i],
            image=torch.from_numpy(test_images[i]),
            bounding_boxes=[],
            threshold=0.7,
            data=data,
            model="cellpose",
            index = i
        )
    # Compute mean AP across all samples for each threshold
    mean_aps = {threshold: np.mean(aps) for threshold, aps in all_aps_per_threshold.items()}

    mean_ap_df = pd.DataFrame(list(mean_aps.items()), columns=["IoU Threshold", "Mean AP"])

    # Print the DataFrame
    print(mean_ap_df)
    save_results_to_excel(results, model="cellpose", data=data)
    shutil.rmtree(test_dir, ignore_errors=True)
