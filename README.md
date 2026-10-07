# GAN-Based Image Augmentation for Abnormal Cell Detection

A deep learning project focused on improving abnormal cell segmentation through GAN-based microscopy image augmentation.

The project uses the **LIVECell dataset** and explores synthetic image generation with **Pix2Pix (cGAN)** and **cGAN-Seg**, followed by semantic segmentation using **DeepLabV3 with a ResNet50 backbone**.

## Results

- **SH-SY5Y IoU:** 72.8%
- **8-class mean IoU (mIoU):** 84.1%

## Overview

Microscopy datasets often contain limited or imbalanced examples of specific cell types.

This project explores whether GAN-generated microscopy images can be used as additional training data to improve cell segmentation performance.

The workflow combines conditional image generation with semantic segmentation:

```text
LIVECell Dataset
      ↓
Data Preparation
      ↓
Pix2Pix / cGAN-Seg
      ↓
Synthetic Microscopy Images
      ↓
Training Data Augmentation
      ↓
DeepLabV3 + ResNet50
      ↓
Cell Segmentation
      ↓
IoU / mIoU Evaluation
```

## Features

- Microscopy image preprocessing
- LIVECell dataset processing
- COCO-format annotation handling
- Conditional GAN-based image generation
- Pix2Pix training pipeline
- cGAN-Seg experiments
- Synthetic image augmentation
- Multi-class cell segmentation
- DeepLabV3 with ResNet50 backbone
- IoU and mean IoU evaluation

## Tech Stack

- Python
- PyTorch
- Computer Vision
- Pix2Pix
- Conditional GANs
- DeepLabV3
- ResNet50
- LIVECell
- COCO format

## Dataset

The project uses the **LIVECell dataset**, which contains microscopy images and annotations for multiple cell types.

The experiments include both individual cell-type evaluation and multi-class segmentation.

## Methodology

### 1. Data Preparation

LIVECell microscopy images and annotations are processed into the required training format.

### 2. Image Generation

Conditional GAN architectures are used to generate synthetic microscopy samples.

The experiments include:

- Pix2Pix
- cGAN-Seg

### 3. Data Augmentation

Generated images are added to the training data to increase dataset diversity and improve segmentation robustness.

### 4. Semantic Segmentation

A **DeepLabV3 model with a ResNet50 backbone** is trained for cell segmentation.

### 5. Evaluation

Segmentation quality is evaluated using:

- Intersection over Union (IoU)
- Mean Intersection over Union (mIoU)

## Performance

| Metric | Result |
|---|---:|
| SH-SY5Y IoU | 72.8% |
| 8-class mIoU | 84.1% |

## Project Structure

```text
gan-cell-image-augmentation/
├── scripts/          # Training and utility scripts
├── src/              # Core project code
├── unet_scratch/     # Segmentation experiments
├── requirements.txt  # Python dependencies
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/azacrafts/gan-cell-image-augmentation.git
cd gan-cell-image-augmentation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Future Improvements

- Evaluate additional GAN architectures
- Improve synthetic image quality
- Compare real-only vs augmented training datasets
- Perform more detailed per-class evaluation
- Add additional segmentation architectures
- Improve experiment reproducibility and configuration management
