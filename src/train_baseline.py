import argparse, json, random
import numpy as np
import torch
import torch.nn as nn
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from .models import SmallCNN
from .project import MODEL_DIR, ensure_output_dirs

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_one(seed=42, max_epochs=30, patience=5):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03)),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.ToTensor()
    train_ds = PneumoniaMNIST(split="train", download=True, transform=train_transform, size=64)
    val_ds = PneumoniaMNIST(split="val", download=True, transform=eval_transform, size=64)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    model = SmallCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train(); total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device).view(-1)
            optimizer.zero_grad(); loss = criterion(model(images), labels)
            loss.backward(); optimizer.step(); total += loss.item() * len(images)
        train_loss = total / len(train_loader.dataset)
        model.eval(); total = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device).view(-1)
                total += criterion(model(images), labels).item() * len(images)
        val_loss = total / len(val_loader.dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    ensure_output_dirs()
    checkpoint = MODEL_DIR / f"smallcnn_pneumoniamnist_seed_{seed}.pt"
    torch.save(model.state_dict(), checkpoint)
    with open(MODEL_DIR / f"smallcnn_pneumoniamnist_seed_{seed}.json","w",encoding="utf-8") as f:
        json.dump({"seed":seed,"best_validation_loss":best_loss,"epochs_ran":len(history),"history":history},f,indent=2)
    return checkpoint, history

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42); p.add_argument("--max-epochs",type=int,default=30); p.add_argument("--patience",type=int,default=5)
    a=p.parse_args(); c,h=train_one(a.seed,a.max_epochs,a.patience)
    print(f"Saved checkpoint: {c}")