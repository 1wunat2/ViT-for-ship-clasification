import os
import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.utils
from sklearn.model_selection import train_test_split
import random
import ViT
import evaluation
from evaluation import conf_matrix_plot, evaluation_metrics
from ViT import VisionTransformer, perform_inference
import pandas as pd
from pathlib import Path

model_default_args = dict(
    in_channels=1,           
    patch_size=7,
    emb_size=64,
    n_heads=8,
    n_layers=6,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
wandb_project = "LearningRate"

class Args:
    batch_size = 1
    n_epochs = 10
    seed = 500
    learning_rate = 3e-4
    n_cpu = 4
    input_dir = "/home/cwu/Documents/NathanProject/archive"
    compile = False
    output_dir = "/home/cwu/Documents/NathanProject/archive"
    resume_training = False
    saved_model_path = "./model_output/saved_models"

    def __init__(self, lr=3e-4, n_epochs=10, batch_size=1) -> None:
        self.learning_rate = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

args = Args()

random.seed(args.seed)

ship_map = {
            "Cargo": 1,
            "Military": 2,
            "Carrier": 3,
            "Cruise": 4,
            "Tankers": 5
        }
rev_map = dict([[v,k] for k,v in ship_map.items()])


class ShipDataset(Dataset):
    def __init__(self, path, data, labels):
        self.path = path  # Path to the dataset
        self.data = data # list of all the files in the dataset
        self.labels = labels # list of all the labels        

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),  # Convert the image to grayscale
                transforms.Resize((28, 28)),  # Resize the image to 28x28
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.data[idx]))
        img = self.transform(img)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        return {"img": img, "label": label}
    
def train(model, optimiser, args, model_args, loss_fn, train_dataloader, valid_dataloader, epoch_start, epoch_end, best_loss_val):
    model.train()
    print(model)
    if args.compile:
        model = torch.compile(model)
    wandb_run_name = (
        f"VisionTransformer_lr_{args.learning_rate}_batch_{args.batch_size}"
    )
    wandb.init(project=wandb_project, name=wandb_run_name, config=args)

    batches_done = 0
    print("[train.py]: Training started...")
    print(
        f"[train.py]: Total Epochs: {args.n_epochs} \t Batches per epoch: {len(train_dataloader)} "
        f"\t Total batches: {len(train_dataloader) * args.n_epochs}"
    )
    for epoch in range(epoch_start, epoch_end):
        y_preds = []
        y_train = []
        losses = []
        print(f"[train.py] Training Epoch {epoch}")
        for i, data in tqdm(enumerate(train_dataloader)):
            if batches_done == 0:
                # Log the first batch of images
                img_grid = torchvision.utils.make_grid(data["img"], nrow=16)
                wandb.log({"Images from Batch 0": wandb.Image(img_grid)})
                wandb.watch(model, loss_fn, log="all", log_freq=100)

            inputs, labels = data["img"].to(device), data["label"].to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            preds = F.softmax(outputs, dim=1).argmax(dim=1)
            y_preds.append(preds.cpu().numpy())
            y_train.append(labels.cpu().numpy())
            losses.append(loss.item())

            batches_done += 1

        # Train Metrics
        loss_train = torch.tensor(losses).mean().item()
        y_train, y_preds = np.concatenate(y_train), np.concatenate(y_preds)
        train_metrics, train_conf_matrix = evaluation_metrics(y_train, y_preds)

        # Validation metrics
        y_val, y_preds_val, loss_val = perform_inference(
            model, valid_dataloader, device, loss_fn
        )
        val_metrics, val_conf_matrix = evaluation_metrics(y_val, y_preds_val)

        # wandb logging
        train_metrics["Loss"], val_metrics["Loss"] = loss_train, loss_val
        wandb_log = {"epoch": epoch}
        for metric in train_metrics:
            wandb_log[f"{metric}_train"] = train_metrics[metric]
            wandb_log[f"{metric}_validation"] = val_metrics[metric]
        fig1 = conf_matrix_plot(train_conf_matrix, "Train")
        fig2 = conf_matrix_plot(val_conf_matrix, "Validation")
        wandb_log["Train Confusion Matrix"] = wandb.Image(fig1)
        wandb_log["Validation Confusion Matrix"] = wandb.Image(fig2)
        wandb.log(wandb_log)

        print(f"EPOCH: {epoch}")
        print(
            f'[TRAINING METRICS] Loss: {loss_train} | Accuracy: {train_metrics["accuracy"]} | '
            f'F1: {train_metrics["f1_score"]} | Precision: {train_metrics[f"precision"]} | Recall:'
            f'{train_metrics["recall"]}'
        )
        print(
            f'[VALIDATION METRICS] Loss: {loss_val} | Accuracy: {val_metrics["accuracy"]} | '
            f'F1: {val_metrics["f1_score"]} | Precision: {val_metrics[f"precision"]} | Recall:'
            f'{val_metrics["recall"]}'
        )

        checkpoint = {
            "epoch": epoch,
            "model_args": model_args,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "best_loss_val": best_loss_val,
        }
        print(f"[train.py]: Saving model at epoch {epoch}...")
        torch.save(
            checkpoint,
            os.path.join(args.output_dir, "saved_models", f"ckpt_epoch_{epoch}.pt"),
        )
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            print(f"[train.py]: Found new best model at epoch {epoch}. Saving model...")
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, "saved_models", "0.pt"),
            )
    return

def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, time)
    os.makedirs(os.path.join(args.output_dir, "saved_models"), exist_ok=True)

    image_folder = r"/home/cwu/Documents/AMATH445_Project/archive/train/images"

    data_df = pd.read_csv(r"/home/cwu/Documents/AMATH445_Project/archive/sample_submission_ns2btKE.csv")
    X, y = data_df['image'], data_df['category']

    # Split train/val/test as 75/15/10
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)    
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1/(0.1 + 0.15)) 

    train_df = pd.read_csv(r"/home/cwu/Documents/AMATH445_Project/archive/train/train.csv")
    X, y = train_df['image'], train_df['category']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    # test_df = pd.read_csv(r"/home/cwu/Documents/AMATH445_Project/archive/test_ApKoW4T.csv")
    # X_test, y_test = test_df['image'], test_df['category']

    train_dataset = ShipDataset(image_folder, list(X_train), list(y_train))
    valid_dataset = ShipDataset(image_folder, list(X_val), list(y_val))
    # test_dataset = ShipDataset(image_folder, list(X_test), list(y_test))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=0)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=0)
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=args.batch_size, num_workers=0)

    epoch_start, epoch_end = 0, args.n_epochs
    best_loss_val = float("inf")
    class_freqs = np.bincount([sample["label"] for sample in train_dataset]) / len(
        train_dataset
    )
    class_freqs = torch.tensor(class_freqs, device=device, dtype=torch.float32)
    n_classes = len(class_freqs)
    n_patches = (                                # H x W / patch_size^2
        train_dataset[0]["img"].shape[-1]
        * train_dataset[0]["img"].shape[-2]
        // model_default_args["patch_size"] ** 2
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    if args.resume_training:
        checkpoint = torch.load(args.saved_model_path, map_location=device)
        model_args = checkpoint["model_args"]
        epoch_start = checkpoint["epoch"] + 1
        epoch_end = args.n_epochs + epoch_start
        best_loss_val = checkpoint["best_loss_val"]
        model = VisionTransformer(**model_args).to(device)
        state_dict = checkpoint["model_state_dict"]
        # fix the keys of the state dictionary.
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        print("[train.py]: Resuming training...")
    else:
        model_args = dict(
            n_classes=n_classes, class_freqs=class_freqs, n_patches=n_patches, **model_default_args
        )
        model = VisionTransformer(**model_args).to(device)
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, optimiser, args, model_args, loss_fn, train_dataloader, valid_dataloader, epoch_start, epoch_end, best_loss_val)

    return

torch.cuda.empty_cache()  
main(args)

if __name__ == "__main__":

    # Tune learning rate
    arg_1 = Args(0.1)
    main(arg_1)

    # arg_2 = Args(0.01)
    # main(arg_2)

    # arg_3 = Args(0.001)
    # main(arg_3)
    
    # main(args)