import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model.unet import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_prediction_as_imgs,
)


#HYPERPARAMETERS
LERANING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGTH = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR="data/train"
MASK_IMG_DIR="data/train_masks"
VAL_IMG_DIR="data/val"
VAL_MASK_IMG_DIR="data/val_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = target.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGTH, IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(IMAGE_HEIGTH, IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    model = UNET(in_ch=3, out_ch=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LERANING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        MASK_IMG_DIR,
        VAL_IMG_DIR,
        VAL_MASK_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_EPOCHS,
        PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint={
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict()
        }

        save_checkpoint(checkpoint)
        check_accuracy(val_loader, model, device=DEVICE)

        save_prediction_as_imgs(
            val_loader,model,folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
