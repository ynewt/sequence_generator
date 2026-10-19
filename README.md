# Tiny GPT From Scratch — on Proteins and DNA

A small character-level GPT (decoder-only transformer) written from scratch in PyTorch, in the style of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video. Instead of Shakespeare it can train on Shakespeare text as well as **biological sequences**: human proteins from UniProt or the *E. coli* genome, or any custom FASTA sequences. A small interactive chat program lets you type the start of a sequence and watch the model complete it.

The goal is to understand LLMs, not to get a useful protein model. Everything here is about 350 lines of readable code. It contains the same parts as GPT-2, ESM, and the other models you hear about; it is just smaller.

---

## 1. Concepts demonstrated in this project

| Concept | Where it lives | What to look for |
|---|---|---|
| **Tokenization** | [prepare_data.py](prepare_data.py) | Every unique character becomes one integer token. For proteins the vocabulary is the 20 amino acids plus `U` (selenocysteine) and `\n` (end of sequence), 22 tokens in total. Real LLMs use subword tokenizers (BPE); the idea is the same. |
| **Embeddings** | `GPT.__init__` in [model.py](model.py) | Each token ID is looked up as a learned vector (`tok_emb`). A second learned vector encodes its *position* (`pos_emb`), because attention by itself has no sense of order. |
| **Self-attention** | `CausalSelfAttention` in [model.py](model.py) | Each position builds a *query* ("what am I looking for?"), a *key* ("what do I contain?") and a *value* ("what I pass along"). `softmax(QKᵀ/√d)` decides how much each position reads from every earlier one. |
| **Causal masking** | the `mask` buffer | A lower-triangular matrix blocks attention to *future* positions, so the model can only use the past to predict the next character. |
| **Multi-head attention** | `n_head` | The embedding is split into several heads that attend independently, for example one tracking local motifs and another tracking long-range contacts. |
| **Transformer block** | `Block` | Attention (tokens exchange information), then an MLP (each token processes it alone). Both are wrapped in residual connections with pre-LayerNorm. |
| **Language-model objective** | `GPT.forward` | Cross-entropy between the predicted distribution and the actual next character, at *every* position at once. |
| **Training loop** | [train.py](train.py) | Random batches, forward pass, loss, `backward()`, gradient clipping, AdamW step, and a learning rate that warms up and then follows a cosine decay. Periodic validation shows whether the model is overfitting. |
| **Sampling** | `GPT.generate`, [chat.py](chat.py) | Autoregressive generation: predict a distribution, sample one character, append it, repeat. *Temperature* and *top-k* control how random the samples are. |

### The architecture in one picture

```
token ids (B,T)
   │
   ├─ token embedding ─┐
   └─ position embed. ─┴─(+)─► x  (B,T,C)
                                │
               ┌────────────────┴───────────────┐
               │  Block × n_layer                │
               │   x = x + Attention(LN(x))      │   ◄─ tokens communicate
               │   x = x + MLP(LN(x))            │   ◄─ tokens compute
               └────────────────┬───────────────┘
                                │
                         LayerNorm → Linear
                                │
                      logits (B,T,vocab) → softmax → next-char probabilities
```

With the default settings (4 layers, 4 heads, 128-dim embeddings, 256-character context) the model has about **0.83M parameters**. GPT-2 small has 124M, and the ESM-2 protein models range from 8M to 15B.

---

## 2. Project layout

```
sequence_generator/
├── README.md           ← this file
├── requirements.txt    torch, numpy
├── prepare_data.py     download corpus → tokenize → data/<name>/{train,val}.bin + meta.json
├── model.py            the GPT itself (attention, MLP, blocks, generate)
├── train.py            the training loop; writes checkpoints/<name>/ckpt.pt and log.csv
├── chat.py             interactive prompt → completion REPL
├── data/               (created) downloaded and tokenized datasets
└── checkpoints/        (created) trained models and loss logs
```

---

## 3. Datasets

| `--dataset` | Source | Size | Vocab | How it is formatted |
|---|---|---|---|---|
| `protein` (default) | UniProt Swiss-Prot, reviewed human proteins (organism 9606) | about 20k proteins, 11.4M residues | 22 | one protein per line; FASTA headers removed |
| `dna` | NCBI NC_000913.3, *E. coli* K-12 MG1655 genome | 4.6M bases | 5 (`ACGT\n`) | the genome cut into 256-base lines |
| `shakespeare` | Karpathy's tinyshakespeare | 1.1M chars | 65 | plain text; a good sanity check because you can *read* the output |
| `custom` | any FASTA or text file you supply | any | any | FASTA becomes one sequence per line; plain text is used as-is |

In the bio datasets the newline character acts as the **end-of-sequence token**. That lets the model learn where proteins *start* (almost always with `M`, methionine) and when to *stop*.

---

## 4. Setup

```bash
git clone https://github.com/ynewt/sequence_generator.git
cd sequence_generator
pip install -r requirements.txt        # torch and numpy
```

It runs on CPU, an NVIDIA GPU (`cuda`), or Apple Silicon (`mps`). `train.py` picks the device automatically.

The repo includes one small trained checkpoint, `checkpoints/protein/ckpt.pt` (2,000 steps, val loss 2.79), so you can try `python chat.py` right away. The `data/` folder is not included; `prepare_data.py` downloads and tokenizes it.

---

## 5. How to use it

### Step 1: Prepare data

```bash
python prepare_data.py --dataset protein
# or
python prepare_data.py --dataset dna
python prepare_data.py --dataset shakespeare
python prepare_data.py --dataset custom --input path/to/my.fasta --name mydata
```

The script prints the vocabulary and the train/validation token counts. It splits 90/10 on a line boundary, so no protein ends up half in train and half in validation.

### Step 2: Train

```bash
python train.py --dataset protein
```

Every `eval_interval` steps you will see something like:

```
iter   500 | train 2.7801 | val 2.7905 | ~16.3 choices/char | 1210s
  sample: MASPRLLLEGAEQRSLSPEGS / MDLEKLRQAL...
```

* **train / val**: average cross-entropy loss in nats per character. Lower is better.
* **~N choices/char**: `exp(val loss)`, the number of characters the model is effectively choosing between. A model with no knowledge would score 22 for proteins and 4 for DNA.
* **sample**: a short generation so you can watch it improve. ` / ` marks the end of a sequence.

The best checkpoint (lowest validation loss) is saved to `checkpoints/<dataset>/ckpt.pt`, and the loss curve goes to `checkpoints/<dataset>/log.csv`.

**Useful flags**

| Flag | Default | Notes |
|---|---|---|
| `--max_iters` | 2000 | number of optimizer steps |
| `--batch_size` | 32 | sequences per step |
| `--block_size` | 256 | context length in characters |
| `--n_layer`, `--n_head`, `--n_embd` | 4, 4, 128 | model size (`n_embd` must be divisible by `n_head`) |
| `--lr`, `--min_lr`, `--warmup_iters` | 1e-3, 1e-4, 100 | learning-rate schedule |
| `--dropout` | 0.1 | regularization; set it to 0 for tiny experiments |
| `--resume` | off | continue from the saved checkpoint |
| `--device` | auto | `cpu` / `cuda` / `mps` |

**Timing on this machine (CPU only, measured):** about 1.4 s per training step at the default settings, or roughly 1.5–3 s once evaluation and other work on the machine are included. 2000 steps therefore take about 1 hour or more. For faster experiments, shrink the work per step:

```bash
# about 0.3 s/step (roughly 4.5x faster): 1500 steps in about 10 minutes, and it still learns the basics
python train.py --dataset protein --block_size 128 --batch_size 16 --max_iters 1500

# quick check that everything runs
python train.py --dataset shakespeare --max_iters 200 --eval_interval 50 --eval_iters 10
```

You can stop training at any time with Ctrl-C. The best checkpoint so far is already on disk, and `--resume` continues from it.

### Step 3: Chat with it

```bash
python chat.py --dataset protein
```

```
Loaded checkpoints/protein/ckpt.pt  (iter 200, val loss 2.856, 0.83M params)
Vocabulary: ACDEFGHIKLMNPQRSTUVWY
Type a prompt and press Enter (empty prompt = start a fresh sequence). /help for commands.

you> /probs
      'M'  78.1% #######################################
      'Y'   2.0% #
      'F'   1.5%
      ...
you> MKT
gpt> MKTLIELPGLLRSPPPEDAAAAPDAMEGTASSSPSAAPATSRASAASGNSVALTTTAAPPTAPGPFISPADGLPSSLAGPRIK
     length 83
```

*(Real output from a model trained for only 200 steps. It has already learned that proteins start with `M`. Your completions will differ, since sampling is random.)*

What the model does: it is **not** an assistant that answers questions. It is a *completion* model. Whatever you type is treated as the beginning of a sequence, and the model continues it one character at a time until it predicts end-of-sequence or reaches the length limit. (A raw GPT behaves this way before instruction tuning turns it into ChatGPT.)

**Prompt ideas**

| Dataset | Try | What it shows |
|---|---|---|
| protein | *(empty prompt)* | a sequence from scratch; it should almost always start with `M` |
| protein | `MKKLLL` | a signal-peptide-like start (hydrophobic stretch) |
| protein | `/probs` *(nothing after it)* | the probability of each first residue, where `M` should dominate |
| protein | `/probs MAAAAA` | whether it has learned low-complexity repeats |
| protein | `PPGPPGPP` | collagen-like Gly-X-Y repeats |
| dna | `ATGAAA` | a start codon; does it keep a codon-like rhythm? |
| dna | `/probs TATA` | what it expects after a promoter-like motif |
| shakespeare | `ROMEO:` | the classic demo |

**Chat commands**

| Command | Effect |
|---|---|
| `/temp 0.5` | temperature; lower gives safer, more repetitive output and higher gives wilder output (default 0.8) |
| `/topk 5` | sample only from the 5 most likely next characters (`0` turns it off) |
| `/len 500` | maximum characters to generate |
| `/n 3` | generate 3 different completions per prompt |
| `/line off` | keep generating past end-of-sequence (several proteins in a row) |
| `/probs <prompt>` | show a bar chart of the model's next-character probabilities instead of sampling |
| `/help`, `/quit` | |

Other options: `python chat.py --ckpt path/to/ckpt.pt --temperature 0.6 --top_k 10 --max_len 400`.

---

## 6. What results to expect (and why proteins are hard)

| Dataset | Random guessing | Letter frequencies only | Tiny model, realistic result |
|---|---|---|---|
| protein | 3.09 (ln 22) | about 2.90 | about 2.65–2.80 |
| dna (*E. coli*) | 1.39 (ln 4) | about 1.386 | about 1.30–1.35 |
| shakespeare | 4.17 (ln 65) | about 3.3 | about 1.5 (readable text) |

Proteins and DNA are **close to incompressible** at the character level, which is a lesson in itself. An amino acid's identity is determined largely by 3D structure and evolution, not by the few residues next to it. Small models therefore pick up only the easy patterns:

* proteins start with `M` and end at a plausible length;
* the right amino-acid frequencies (lots of `L`, `S`, `A`, `E`; few `W`, `C`);
* low-complexity regions and repeats (poly-Q, poly-A, Pro-rich, collagen `G-X-Y`);
* hydrophobic signal peptides at the N-terminus;
* in DNA, codon periodicity and GC skew.

Generated proteins will look *plausible* to the eye but will not fold. Real protein language models such as ESM-2 and ProtGPT2 reach better scores with tens to thousands of times more parameters and data.

**Tip:** train on Shakespeare first. Because you can read the output, you can watch the model go from random characters, to real words, to plausible play dialogue. That gives you intuition for what a falling loss *means*, which is hard to see in amino-acid strings.

---

## 7. Experiments to try next

1. **Ablations.** Turn off the causal mask and watch training loss collapse while generation breaks (the model cheats by looking ahead). Remove the position embedding, or set `n_layer 1`, and compare validation loss.
2. **Scaling.** Train `n_embd` 64 / 128 / 256 for the same number of steps and plot `log.csv`. Is bigger always better at a fixed compute budget?
3. **Context length.** Try `block_size` 32 vs 256 on proteins. Does long-range context help?
4. **Overfitting.** Train on a tiny custom FASTA (one protein family) with `--dropout 0` and watch train and val loss diverge.
5. **Look at attention.** Return `att` from `CausalSelfAttention.forward` and plot heatmaps for a real protein. Do some heads attend to the previous residue? To `i-3`/`i-4` (α-helix periodicity)?
6. **Better tokenization.** Use codons (3-mers) as tokens for DNA, or train a BPE tokenizer, and compare bits per base.
7. **Conditioning.** Prepend a tag such as `<kinase>` or the organism to each sequence, so you can steer generation from the prompt. This is a first step toward a "chat" model.
8. **Use PyTorch's fused attention.** Replace the manual attention with `F.scaled_dot_product_attention` and measure the speedup.

---

## 8. References

* Karpathy, *Let's build GPT: from scratch, in code, spelled out* — https://www.youtube.com/watch?v=kCc8FmEb1nY
* nanoGPT — https://github.com/karpathy/nanoGPT
* Vaswani et al., *Attention Is All You Need* (2017)
* Ferruz et al., *ProtGPT2 is a deep unsupervised language model for protein design* (2022)
* Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2, 2023)
