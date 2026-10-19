"""
Download a corpus and turn it into token IDs for a character-level GPT.

Datasets:
  protein      Human reviewed (Swiss-Prot) protein sequences from UniProt, one per line
  dna          E. coli K-12 MG1655 complete genome from NCBI, split into fixed-length lines
  shakespeare  Tiny Shakespeare (the classic nanoGPT / char-rnn corpus)
  custom       Any local FASTA or text file you pass with --input

Output (in data/<name>/):
  train.bin, val.bin   uint16 arrays of token IDs
  meta.json            vocabulary: the character <-> integer mappings

Usage:
  python prepare_data.py --dataset protein
  python prepare_data.py --dataset dna
  python prepare_data.py --dataset custom --input my_seqs.fasta --name myseqs
"""
import argparse
import json
import os
import urllib.request

import numpy as np

URLS = {
    "protein": "https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true+AND+organism_id:9606&format=fasta",
    "dna": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text",
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def download(url, path):
    if os.path.exists(path):
        print(f"Using cached download: {path}")
        return
    print(f"Downloading {url}\n  -> {path}")
    req = urllib.request.Request(url, headers={"User-Agent": "tiny-gpt-tutorial"})
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())


def parse_fasta(text):
    """Return a list of sequences (headers dropped, line wraps joined)."""
    seqs, cur = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return seqs


def build_text(dataset, raw, dna_line_len):
    """Convert the raw download into the exact text the model will learn from."""
    if dataset == "shakespeare":
        return raw
    if not raw.lstrip().startswith(">"):
        return raw  # custom plain-text file
    seqs = parse_fasta(raw)
    if dataset == "dna":
        # One long genome: chop into fixed-length "lines" so newlines are meaningful
        # boundaries and the model sees many independent windows.
        genome = "".join(seqs)
        seqs = [genome[i:i + dna_line_len] for i in range(0, len(genome), dna_line_len)]
    # One sequence per line; "\n" acts as the end-of-sequence token.
    return "\n".join(seqs) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["protein", "dna", "shakespeare", "custom"], default="protein")
    p.add_argument("--input", help="path to a local FASTA/text file (for --dataset custom)")
    p.add_argument("--name", help="output folder name under data/ (defaults to --dataset)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--dna_line_len", type=int, default=256)
    args = p.parse_args()

    name = args.name or args.dataset
    out_dir = os.path.join("data", name)
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == "custom":
        if not args.input:
            p.error("--dataset custom requires --input")
        raw_path = args.input
    else:
        raw_path = os.path.join(out_dir, "raw.txt")
        download(URLS[args.dataset], raw_path)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw = f.read()
    text = build_text(args.dataset, raw, args.dna_line_len)

    # ---- Tokenization: every unique character becomes one token ----
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)

    # Split on a line boundary so no sequence is cut between train and val
    split = int(len(ids) * (1 - args.val_frac))
    nl = stoi.get("\n")
    if nl is not None:
        while split < len(ids) and ids[split] != nl:
            split += 1
        split += 1
    ids[:split].tofile(os.path.join(out_dir, "train.bin"))
    ids[split:].tofile(os.path.join(out_dir, "val.bin"))

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"dataset": args.dataset, "chars": chars}, f, indent=1)

    print(f"Characters in corpus: {len(text):,}")
    print(f"Vocab size: {len(chars)}  ->  {''.join(chars)!r}")
    print(f"Train tokens: {split:,}   Val tokens: {len(ids) - split:,}")
    print(f"Wrote {out_dir}/train.bin, val.bin, meta.json")


if __name__ == "__main__":
    main()
