import torch
from codebase.utils.visualize import plot_semantic_segmentation
from codebase.utils.metrics import metrics_semantic_segmentation
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import pandas as pd


def train_semantic_seg(DEVICE, train_data, num_epochs, output_path):
    model = smp.Unet(
        encoder_name="resnet34",  # Choose encoder
        encoder_weights="imagenet",  # Use pretrained weights
        in_channels=3,  # Your input has 2 channels
        classes=1  # Binary segmentation (1 output channel)
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs(output_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_TP, total_FP, total_FN = 0, 0, 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for batch_index, batch in enumerate(tqdm(train_data, desc="Training")):
            if batch is None or len(batch["image"]) == 0:
                print(f"Skipping empty batch {batch_index}.")
                continue

            for item_index in range(len(batch["image"])):
                input_tensor = batch["image"][item_index].float().to(DEVICE)
                input_tensor = input_tensor.permute(2, 0, 1) # shape: [2¡3, H, W]
                original_gt_masks = batch['original_gt_masks'][item_index].float().to(DEVICE)
                gt_mask = (original_gt_masks > 0).float().unsqueeze(0)

                optimizer.zero_grad()
                pred = model(input_tensor.unsqueeze(0))  # add batch dim: [1, 2, H, W] → [1, 1, H, W]
                loss = criterion(pred, gt_mask.unsqueeze(0))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Binarize prediction
                pred_mask_bin = (torch.sigmoid(pred) > 0.5).float()

                TP, FP, FN = metrics_semantic_segmentation(pred_mask_bin, gt_mask)

                total_TP += TP
                total_FP += FP
                total_FN += FN

                if epoch % 5 == 0 or epoch == num_epochs - 1:
                    plot_semantic_segmentation(pred_mask_bin, gt_mask, input_tensor)

        precision = total_TP / (total_TP + total_FP + 1e-8)
        recall = total_TP / (total_TP + total_FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"Loss: {total_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    torch.save(model.state_dict(), os.path.join(output_path, f"unet_epoch{epoch + 1}.pth"))


def test_semantic_segmentation(DEVICE, test_data, model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    total_TP, total_FP, total_FN = 0, 0, 0
    results = []

    for index, test_sample in enumerate(test_data):
        input_tensor = test_sample['image'].float().permute(2, 0, 1).to(DEVICE)
        original_gt_masks = test_sample['original_gt_masks'].float().to(DEVICE)
        gt_mask = (original_gt_masks > 0).float().unsqueeze(0)

        with torch.no_grad():
            input_tensor_batch = input_tensor.unsqueeze(0)
            pred = model(input_tensor_batch)
            pred_mask_bin = (torch.sigmoid(pred) > 0.5).float().squeeze().cpu()

        TP, FP, FN = metrics_semantic_segmentation(pred_mask_bin, gt_mask.cpu())

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results.append({
            "sample_index": index,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        total_TP += TP
        total_FP += FP
        total_FN += FN

        plot_semantic_segmentation(pred_mask_bin, gt_mask, input_tensor)

    # Global metrics
    global_precision = total_TP / (total_TP + total_FP + 1e-8)
    global_recall = total_TP / (total_TP + total_FN + 1e-8)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-8)

    print(f"\nGlobal Precision: {global_precision:.4f} | Recall: {global_recall:.4f} | F1: {global_f1:.4f}")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    print("\nPer-sample metrics:")
    print(df_results)

    return df_results
