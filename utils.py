import torch
import torchvision
from datasets.CarvanaDataset import CaravanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print('Saving checkpoint...')
    torch.save(state, filename)

def load_checkpoint(checkpoint,model):
    print("loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=2,
        pin_memory= True,
):
    train_ds= CaravanaDataset(
        img_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds= CaravanaDataset(
        img_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model,device="cuda"):
    num_correct = 0
    num_pixel = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixel += torch.numel(preds)
            dice_score += ((preds * y)*2).sum() / ((preds +y).sum() + 1e-8)

        print(f"Got {num_pixel} / {num_pixel} with accuracy {num_correct / num_pixel *100:.2f}% ")
        print(f"Dice score: {dice_score / len(loader):.2f}")
        model.train()

def save_prediction_as_imgs(
        loader,model,device="cuda",folder= "saved_images/"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x=x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, f"{folder}preds_{idx}.png")
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}y_{idx}.png")
    model.train()