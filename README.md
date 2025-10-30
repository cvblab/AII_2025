# Microscopy Cell Segmentation: Review and Benchmarking of Task-Specific and Foundation Models

![](https://github.com/cvblab/AII_2025/raw/d5ca52d2c13b087168c415a8f8f85f827f667f83/workflow.png)
## Overview
Cell segmentation plays a key role in a wide range of biomedical imaging applications, from single-cell analysis to pathology assessment. While classical deep learning architectures such as U-Net, StarDist, and HoVer-Net have set strong baselines, their reliance on domain-specific training limits generalization across diverse microscopy modalities. The emergence of foundation models, particularly the Segment Anything Model (SAM) and its derivatives, has introduced a paradigm shift toward more universal and adaptable segmentation frameworks. 

In this review, we summarize key advances in microscopy cell segmentation, highlighting both traditional methods and recent foundation model–based approaches. Beyond surveying the literature, we present an experimental comparison of four representative models—our proposed YOLO-SAM, along with CellSAM, Cellpose-SAM, and StarDist—tested on both fluorescence and brightfield microscopy spanning diverse cell populations and shapes. Our findings illustrate trade-offs between accuracy, robustness, and adaptability, with foundation-based models showing particular promise for cross-domain performance.

By combining a comprehensive review with systematic benchmarking, this work provides practical guidance for researchers and outlines current challenges and future opportunities in developing robust, generalizable cell segmentation methods for microscopy.

## Usage

### 1. Clone the repository

      git clone https://github.com/cvblab/All_2025

### 2. Install dependencies

Make sure you have Python 3.12+ installed. Then run:

      pip install -r requirements.txt

### 3. Code structure

```
AII_2025/
├─ data/                  # data processing
├─ models/                # segmentation models
├─ utils/                 # helper functions
├─ yolov8/                # detection model
├─ requirements.txt
├─ train.py
├─ predict.py
├─ README.md
```

## Data

All datasets are public. Access links in the reference section.

![](https://github.com/cvblab/AII_2025/raw/d5ca52d2c13b087168c415a8f8f85f827f667f83/datasets.png)


