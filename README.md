# U-Net from Scratch in PyTorch

This repository contains a **functional implementation of U-Net** for image segmentation in PyTorch. It was implemented from scratch, with the ability to train on custom datasets and save predicted masks for validation images.

---

## Features

- U-Net architecture implemented **from scratch** using PyTorch
- Custom dataset support
- Training and validation scripts included
- Saves predicted images for easy visualization
- Minimal dependencies

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Mystic703/U-net-from-scracth.git
cd U-net-from-scracth

2.Install dependencies:
pip install -r requirements.txt

Training
python train.py
Modify dataset paths in train.py as needed.

Predictions
Predicted masks are saved in the saved_images/ folder for validation images.

Dataset
The dataset is not included in this repository.
Place your dataset in the data/ folder and update the paths in train.py.

Notes
Large model checkpoints (checkpoint.pth.tar) are not included due to GitHub size limits.
Recommended to use GPU for faster training.
IDE config files (.idea/) are ignored.

Requirements
torch
torchvision
numpy
tqdm
Pillow

