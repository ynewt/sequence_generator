"""
Interactive "chat" with a trained tiny GPT.

The model is not an assistant: it continues whatever you type, one character at a time.
Type the start of a sequence (e.g. "MKT" for a protein) and it completes it.

Usage:
  python chat.py --dataset protein
  python chat.py --ckpt checkpoints/dna/ckpt.pt --temperature 0.7

Commands inside the chat:
  /temp 0.8      set sampling temperature (lower = safer, higher = more random)
  /topk 5        only sample from the k most likely next characters (0 = off)
  /len 300       max characters to generate
  /n 3           number of completions per prompt
  /line on|off   stop at end of sequence (newline) or keep going
  /probs MKT     show the model's next-character probabilities after a prompt
  /help, /quit
"""
import argparse
import os

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["meta"], ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="protein")
    p.add_argument("--ckpt", default=None, help="defaults to checkpoints/<dataset>/ckpt.pt")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--max_len", type=int, default=300)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt_path = args.ckpt or os.path.join("checkpoints", args.dataset, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"No checkpoint at {ckpt_path}. Train first: python train.py --dataset {args.dataset}")
    model, meta, ckpt = load(ckpt_path, args.device)
    chars = meta["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    nl = stoi.get("\n")
    is_bio = meta["dataset"] in ("protein", "dna")

    s = {"temp": args.temperature, "topk": args.top_k, "len": args.max_len, "n": 1, "line": True}
    print(f"Loaded {ckpt_path}  (iter {ckpt['iter']}, val loss {ckpt['best_val']:.3f}, "
          f"{model.num_params() / 1e6:.2f}M params)")
    print(f"Vocabulary: {''.join(c for c in chars if c != chr(10))}")
    print("Type a prompt and press Enter (empty prompt = start a fresh sequence). /help for commands.\n")

    def encode(text):
        unknown = sorted(set(c for c in text if c not in stoi))
        if unknown:
            print(f"  (ignoring characters not in vocabulary: {''.join(unknown)})")
        return [stoi[c] for c in text if c in stoi]

    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        prompt = line.strip()
        if is_bio:
            prompt = prompt.upper()

        if prompt.startswith("/"):
            cmd, _, val = line.strip().partition(" ")
            try:
                if cmd in ("/quit", "/exit"):
                    break
                elif cmd == "/help":
                    print(__doc__)
                elif cmd == "/temp":
                    s["temp"] = float(val)
                elif cmd == "/topk":
                    s["topk"] = int(val)
                elif cmd == "/len":
                    s["len"] = int(val)
                elif cmd == "/n":
                    s["n"] = int(val)
                elif cmd == "/line":
                    s["line"] = val.strip().lower() != "off"
                elif cmd == "/probs":
                    text = val.strip().upper() if is_bio else val
                    ids = ([nl] if nl is not None else []) + encode(text)
                    x = torch.tensor([ids[-model.cfg.block_size:]], device=args.device)
                    with torch.no_grad():
                        probs = F.softmax(model(x)[0][0, -1], dim=-1)
                    top = torch.topk(probs, min(10, len(chars)))
                    for pr, i in zip(top.values.tolist(), top.indices.tolist()):
                        label = "<end>" if chars[i] == "\n" else repr(chars[i])
                        print(f"  {label:>7} {pr:6.1%} " + "#" * int(pr * 50))
                    continue
                else:
                    print("  unknown command, try /help")
                    continue
                print(f"  settings: {s}")
            except ValueError:
                print("  bad value")
            continue

        # Prefix with newline so the model knows a new sequence is starting
        ids = ([nl] if nl is not None else []) + encode(prompt)
        x = torch.tensor([ids], device=args.device)
        for k in range(s["n"]):
            out = model.generate(
                x, s["len"], temperature=s["temp"], top_k=s["topk"] or None,
                stop_id=nl if s["line"] else None,
            )[0].tolist()
            new = "".join(chars[i] for i in out[len(ids):])
            ended = new.endswith("\n")
            new = new.rstrip("\n")
            tag = f"[{k + 1}] " if s["n"] > 1 else ""
            print(f"gpt> {tag}{prompt}\033[1m{new}\033[0m" + ("  <end>" if ended else ""))
            if is_bio:
                print(f"     length {len(prompt) + len(new)}")
        print()


if __name__ == "__main__":
    main()
