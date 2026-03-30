from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.ocr.cnn import OCRCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OCR CNN (classes: 0..9).")
    parser.add_argument("--data-dir", default="data/ocr_cnn", help="Dataset root with folders 0..9.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--out", default="models/ocr_cnn.pt", help="Checkpoint output path.")
    return parser.parse_args()


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return (correct / total) if total > 0 else 0.0


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.data_dir, transform=tfm)
    if len(dataset) < 100:
        raise ValueError("Dataset too small. Build more OCR samples first.")

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = OCRCNN(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item())

        val_acc = _evaluate(model, val_loader, device)
        avg_loss = running / max(1, len(train_loader))
        print(f"epoch={epoch}/{args.epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "val_acc": best_acc,
            "class_to_idx": dataset.class_to_idx,
        },
        out_path,
    )
    print(f"Saved checkpoint to {out_path} (best_val_acc={best_acc:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
