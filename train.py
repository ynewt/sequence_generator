"""
Train a tiny character-level GPT.

Usage:
  python train.py --dataset protein
  python train.py --dataset dna --max_iters 3000 --n_layer 6 --n_embd 192
  python train.py --dataset protein --max_iters 200 --eval_interval 50   # quick smoke test
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="protein", help="folder name under data/")
    p.add_argument("--out_dir", default=None, help="defaults to checkpoints/<dataset>")
    # model size
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimization
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # logging
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true", help="continue from out_dir/ckpt.pt")
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    data_dir = os.path.join("data", args.dataset)
    out_dir = args.out_dir or os.path.join("checkpoints", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Data ----
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    train_data = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)
    print(f"device={device}  vocab={len(meta['chars'])}  train={len(train_data):,}  val={len(val_data):,} tokens")

    def get_batch(split):
        """Random windows of block_size chars; target is the same window shifted by one."""
        data = train_data if split == "train" else val_data
        ix = np.random.randint(0, len(data) - args.block_size - 1, size=args.batch_size)
        x = np.stack([data[i:i + args.block_size] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + args.block_size] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    # ---- Model ----
    cfg = GPTConfig(
        vocab_size=len(meta["chars"]), block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
    )
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    start_iter, best_val = 0, float("inf")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = GPTConfig(**ckpt["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_iter, best_val = ckpt["iter"], ckpt["best_val"]
        print(f"Resumed from {ckpt_path} at iter {start_iter}")
    else:
        model = GPT(cfg).to(device)
    print(f"Model parameters: {model.num_params() / 1e6:.2f}M")

    # Weight decay on matrices only, not on biases / LayerNorm / embeddings' 1-D params
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.99),
    )
    if args.resume and os.path.exists(ckpt_path):
        optimizer.load_state_dict(ckpt["optimizer"])

    def lr_at(it):
        """Linear warmup, then cosine decay to min_lr."""
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        progress = min(1.0, (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters))
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = [model(*get_batch(split))[1].item() for _ in range(args.eval_iters)]
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    def save(it):
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": cfg.__dict__, "meta": meta, "iter": it, "best_val": best_val,
        }, ckpt_path)

    # ---- Training loop ----
    log_path = os.path.join(out_dir, "log.csv")
    if not args.resume or not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iter,train_loss,val_loss\n")
    stoi = {c: i for i, c in enumerate(meta["chars"])}
    t0 = time.time()
    model.train()
    for it in range(start_iter, args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss()
            # Perplexity-style intuition: exp(loss) ~ number of chars the model is "choosing between"
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| ~{math.exp(losses['val']):.1f} choices/char | {time.time() - t0:.0f}s")
            with open(log_path, "a") as f:
                f.write(f"{it},{losses['train']:.4f},{losses['val']:.4f}\n")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(it)
            # Show a short sample so you can watch the model learn
            start = torch.tensor([[stoi.get("\n", 0)]], device=device)
            model.eval()
            sample = model.generate(start, 120, temperature=0.8, top_k=None)[0].tolist()
            model.train()
            print("  sample:", "".join(meta["chars"][i] for i in sample[1:]).replace("\n", " / "))
        if it == args.max_iters:
            break

        for g in optimizer.param_groups:
            g["lr"] = lr_at(it)
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    print(f"Done. Best val loss {best_val:.4f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
