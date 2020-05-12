import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils import data
from torchvision import transforms

from train_utils import (NUM_PTS, ChangeBrightnessContrast,
                         CropRectangle, CropFrame, FlipHorizontal, Rotator,
                         ScaleToSize, ThousandLandmarksDataset,
                         TransformByKeys, create_submission,
                         restore_landmarks_batch)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "--name", "-n",
        help="Experiment name (for saving checkpoints and submits).",
        default="baseline"
    )
    parser.add_argument(
        "--data", "-d",
        help="Path to dir with target images & landmarks.",
        default='data'
    )
    parser.add_argument("--batch-size", "-b", default=256, type=int)
    parser.add_argument("--epochs", "-e", default=2, type=int)
    parser.add_argument("--learning-rate", "-lr", default=3e-4, type=float)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def create_model(device):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = True
    return model


# Взвешенный по фактору изменения размера изображения в предобработке MSE loss
def weighted_mse_loss(preds, ground_true, weights):
    return torch.mean(weights * torch.mean((preds - ground_true) ** 2, axis=1))


# Взвешенный по фактору изменения размера изображения в предобработке MAE loss
def weighted_mae_loss(preds, ground_true, weights):
    return torch.mean(weights * torch.mean(torch.abs(preds - ground_true),
                                           axis=1
                                           )
                      )


# loss из статьи https://arxiv.org/abs/1711.06753
def wing_loss(preds, ground_true, weights=None, w=10, eps=2):
    t = torch.abs(preds - ground_true)
    C = w - w * np.log(1 + w / eps)
    # if weights is None:
    #     return torch.mean(torch.where(t < w, w * torch.log(1 + t / eps), t - C))
    # else:
    #     return torch.mean(torch.where(t < w, w * torch.log(1 + t / eps), t - C) * weights)
    return torch.mean(torch.where(t < w, w * torch.log(1 + t / eps), t - C))


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_mse_loss = []
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
            weights_mse = (1 / batch['scale_coef']) ** 2
            mse_loss = weighted_mse_loss(pred_landmarks,
                                         landmarks,
                                         weights_mse)
            loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_mse_loss.append(mse_loss.item())
        val_loss.append(loss.item())
    return (np.mean(val_loss), np.mean(val_mse_loss))


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader,
                                        total=len(loader),
                                        desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2)) # noqa E501

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        frames_x = batch["crop_left"].numpy()
        frames_y = batch["crop_top"].numpy()
        prediction = restore_landmarks_batch(pred_landmarks, fs,
                                             margins_x, margins_y,
                                             frames_x, frames_y) # noqa E501
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction # noqa E501

    predictions = np.clip(predictions, a_min=0, a_max=None)
    return predictions


def main(args):
    # 1. prepare data & models
    crop_size = (224, 224)
    train_transforms = transforms.Compose([
        CropFrame(9),
        ScaleToSize(crop_size),
        FlipHorizontal(),
        Rotator(30),
        CropRectangle(crop_size),
        ChangeBrightnessContrast(alpha_std=0.05, beta_std=10),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                        ("image",)
                        ),
    ])

    valid_transforms = transforms.Compose([
        CropFrame(9),
        ScaleToSize(crop_size),
        CropRectangle(crop_size),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                        ("image",)
                        ),
    ])

    print("Reading data...")
    print("Reading train data...")
    train_dataset = ThousandLandmarksDataset(
        os.path.join(args.data, 'train'),
        train_transforms,
        split="train",
        exclude_bad_landmarks=False
    )
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=4, pin_memory=True,
        shuffle=True, drop_last=True
    )
    print(f"Read {len(train_dataset)} of items")
    print("Reading valid data...")
    val_dataset = ThousandLandmarksDataset(
        os.path.join(args.data, 'train'),
        valid_transforms,
        split="val",
        exclude_bad_landmarks=False
    )
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=4, pin_memory=True,
        shuffle=False, drop_last=False
    )
    print(f"Read {len(val_dataset)} of items")
    print("Creating model...")
    device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
    model = create_model(device)

    best_val_mse_losses = np.inf
    best_epoch = 0

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=1/np.sqrt(10),
        patience=4,
        verbose=True, threshold=0.01,
        threshold_mode='abs', cooldown=0,
        min_lr=1e-6, eps=1e-08
    )
    loss_fn = fnn.l1_loss

    # 2. train & validate
    print("Ready for training...")

    train_losses = []
    val_losses = []
    val_mse_losses = []

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}')
        time.sleep(0.5)

        # Train
        current_train_loss = train(
            model, train_dataloader,
            loss_fn, optimizer, device
        )
        train_losses.append(current_train_loss)

        # Validation
        current_loss, current_mse_loss = validate(
            model, val_dataloader,
            loss_fn, device
        )
        val_losses.append(current_loss)
        val_mse_losses.append(current_mse_loss)

        print(f'Train loss:          {train_losses[-1]:.4f}')
        print(f'Validation loss:     {val_losses[-1]:.4f}')
        print(f'Validation mse loss: {val_mse_losses[-1]:.4f}')

        scheduler.step(val_losses[-1])

        losses = pd.DataFrame(
            list(zip(train_losses, val_losses, val_mse_losses)),
            columns=['Train', 'Validation', 'Validation MSE']
        )
        losses.to_csv(f'{args.name}_losses.csv', index=False)

        # Save best model
        if val_mse_losses[-1] < best_val_mse_losses:
            best_val_mse_losses = val_mse_losses[-1]
            best_epoch = epoch
            with open(f"{args.name}_best.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)

    print(f'Best epoch: {best_epoch}')

    # 3. predict
    test_dataset = ThousandLandmarksDataset(
        os.path.join(args.data, 'test'),
        valid_transforms,
        split="test",
        exclude_bad_landmarks=False
    )
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=4, pin_memory=True,
        shuffle=False, drop_last=False
    )

    with open(f"{args.name}_best.pth", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    # with open(f"{args.name}_test_predictions.pkl", "wb") as fp:
    #     pickle.dump({"image_names": test_dataset.image_names,
    #                  "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, f"{args.name}_submit.csv")


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
