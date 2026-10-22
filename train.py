"""Train one model, or all three x all seeds.

    python train.py --model resnet1d --seed 0
    python train.py --all                  # 3 models x 3 seeds (Kaggle GPU run)
    python train.py --all --epochs 2       # smoke test

Each run saves checkpoints/<model>_s<seed>.pt (best val loss) and
results/logs/<model>_s<seed>.csv with per-epoch train/val loss, val accuracy,
learning rate, time and the gradient norm reaching the stem conv (used to compare
gradient flow between CNN-Deep-Plain and ResNet1D).
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.torch_dataset import load_split
from models import MODEL_NAMES, build_model, count_params
from utils.config import ROOT, load_config
from utils.seed import set_seed


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def stem_conv(model) -> nn.Conv1d:
    """First conv layer: the gradient reaching it shows how well depth is trained."""
    return next(m for m in model.modules() if isinstance(m, nn.Conv1d))


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += loss_fn(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def train_one(cfg, model_name, seed, data_dir, epochs=None, out_root=ROOT):
    t = cfg["train"]
    set_seed(seed)
    device = pick_device(t["device"])
    train_ds, _, _ = load_split(data_dir, "train")
    val_ds, _, _ = load_split(data_dir, "val")
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=t["batch_size"], num_workers=t.get("num_workers", 0), pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, generator=g, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)

    model = build_model(model_name, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    tag = f"{model_name}_s{seed}"
    ckpt_dir = Path(out_root) / t["ckpt_dir"]
    log_dir = Path(out_root) / cfg["eval"]["results_dir"] / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] {count_params(model):,} params on {device}")

    best, wait, rows = float("inf"), 0, []
    for ep in range(1, (epochs or t["epochs"]) + 1):
        model.train()
        t0, loss_sum, n, gsum, steps = time.time(), 0.0, 0, 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            gsum += stem_conv(model).weight.grad.norm().item()
            steps += 1
            opt.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
        val_loss, val_acc = evaluate(model, val_dl, device, loss_fn)
        sched.step(val_loss)
        row = dict(epoch=ep, train_loss=loss_sum / n, val_loss=val_loss, val_acc=val_acc,
                   lr=opt.param_groups[0]["lr"], stem_grad_norm=gsum / steps, sec=time.time() - t0)
        rows.append(row)
        flag = ""
        if val_loss < best - 1e-4:
            best, wait, flag = val_loss, 0, " *"
            torch.save({"model": model_name, "seed": seed, "epoch": ep, "val_loss": val_loss, "val_acc": val_acc,
                        "params": count_params(model), "state_dict": model.state_dict()}, ckpt_dir / f"{tag}.pt")
        else:
            wait += 1
        print(f"[{tag}] ep {ep:2d}  train {row['train_loss']:.4f}  val {val_loss:.4f}  "
              f"acc {val_acc:.4f}  {row['sec']:.1f}s{flag}")
        if wait >= t["early_stop_patience"]:
            print(f"[{tag}] early stop at epoch {ep}")
            break

    with open(log_dir / f"{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODEL_NAMES, default="resnet1d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="all models x config train.model_seeds")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--data", default=None, help="folder with train/val/test.npz (default: config data.out_dir)")
    args = ap.parse_args()
    cfg = load_config()
    data_dir = Path(args.data) if args.data else ROOT / cfg["data"]["out_dir"]
    runs = [(m, s) for m in MODEL_NAMES for s in cfg["train"]["model_seeds"]] if args.all else [(args.model, args.seed)]
    for m, s in runs:
        train_one(cfg, m, s, data_dir, args.epochs)


if __name__ == "__main__":
    main()
