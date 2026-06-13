"""
v3 — Train the CheXNet (medical encoder) + GPT-2 model.

Same training recipe as the baseline (src/train.py) but with the medical
encoder + its preprocessing. Encoder frozen; projector + GPT-2 trained.

  python src/train_chexnet.py --max-samples 64 --epochs 1   # smoke test
  python src/train_chexnet.py --epochs 5 --batch-size 8     # real run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from dataset import CXRDataset, make_collate_fn
from model_chexnet import CXRReportModelMedical, MEDICAL_TRANSFORM

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "outputs"


def build_loader(jsonl, tokenizer, batch_size, max_len, max_samples, shuffle):
    ds = CXRDataset(ROOT / "data" / jsonl, tokenizer, max_len=max_len,
                    max_samples=max_samples, transform=MEDICAL_TRANSFORM)
    collate = make_collate_fn(tokenizer.pad_token_id)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=0, pin_memory=True), len(ds)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, _ = model(**batch)
        total += loss.item() * batch["input_ids"].size(0)
        n += batch["input_ids"].size(0)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_DIR.mkdir(exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, n_train = build_loader("train.jsonl", tokenizer, args.batch_size,
                                         args.max_len, args.max_samples, shuffle=True)
    val_loader, n_val = build_loader("val.jsonl", tokenizer, args.batch_size,
                                     args.max_len, args.max_samples, shuffle=False)
    print(f"train: {n_train} | val: {n_val} | device: {device}")

    model = CXRReportModelMedical().to(device)
    print(f"trainable params: {sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M  (medical encoder frozen)")

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); model.encoder.eval()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        running = 0.0
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            optim.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, _ = model(**batch)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/step:.3f}")

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.3f}  val_loss={val_loss:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       CKPT_DIR / "chexnet_best.pt")
            print(f"  saved best -> outputs/chexnet_best.pt (val {val_loss:.3f})")


if __name__ == "__main__":
    main()
