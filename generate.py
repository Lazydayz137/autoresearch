"""
Generate MTG text from a checkpoint saved by train.py.

Usage:
    uv run generate.py [--ckpt /workspace/checkpoints/model.pt] [--n 300] [--temp 0.8] [--topk 50]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = self.n_kv_head
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        if self.ve_gate is not None and ve is not None:
            gate = torch.sigmoid(self.ve_gate(ve.mean(-1, keepdim=True).expand(-1, -1, self.n_kv_head)))
            v = v + gate.unsqueeze(-1) * ve.view(B, T, self.n_kv_head, self.head_dim)
        out = fa3.flash_attn_func(q, k, v, causal=True, window_size=(window_size - 1, 0))
        out = out.view(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)) ** 2)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, config.n_kv_head * (config.n_embd // config.n_head))
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        head_dim = config.n_embd // config.n_head
        max_seq = config.sequence_len
        freq = 1.0 / (1000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq, dtype=torch.float32)
        f = torch.outer(t, freq)
        cos = f.cos().repeat(1, 2).view(1, max_seq, 1, head_dim).to(torch.bfloat16)
        sin = f.sin().repeat(1, 2).view(1, max_seq, 1, head_dim).to(torch.bfloat16)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _compute_window_sizes(config):
        sizes = []
        for i in range(config.n_layer):
            c = config.window_pattern[i % len(config.window_pattern)]
            sizes.append(min(256, config.sequence_len) if c == "S" else config.sequence_len)
        return sizes

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)
            return loss
        return logits


MTG_PROMPTS = [
    "Card: Lightning Bolt\nMana: {R} | CMC: 1 | Rarity: common | Type: Instant",
    "Card: Black Lotus\n",
    "Price History: Jace, the Mind Sculptor — 2024-03",
    "Archetype: Modern Burn (Modern)\n",
    "Metagame Snapshot: Rakdos Scam in modern on 2024-07-15",
    "Trading Signal (2024-08-10): BUY on ",
    "Ban List Entry: ",
    "Tournament: Pro Tour Thunder Junction\n",
]


def load_tokenizer(mtg_root: Path):
    pkl = mtg_root / "tokenizer" / "tokenizer.pkl"
    with open(pkl, "rb") as f:
        enc = pickle.load(f)
    return enc


@torch.no_grad()
def generate_text(model, enc, prompt, max_new_tokens, temperature, top_k, device):
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = list(ids)
    for _ in range(max_new_tokens):
        idx_cond = x[:, -model.config.sequence_len:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        out.append(next_id.item())
        try:
            tail = enc.decode(out[-6:])
            if "\n\n" in tail and len(out) > len(ids) + 20:
                break
        except Exception:
            pass
    return enc.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/checkpoints/model.pt")
    ap.add_argument("--mtg-root", default=os.path.expanduser("~/.cache/autoresearch/mtg"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--prompts", nargs="*", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    print(f"loading checkpoint: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    print(f"model config: {asdict(cfg)}", flush=True)

    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    sd = ckpt["model_state_dict"]
    sd = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(dtype=torch.bfloat16)
    model.train(False)

    enc = load_tokenizer(Path(args.mtg_root))
    print(f"tokenizer vocab: {enc.n_vocab}", flush=True)
    print(f"checkpoint meta: val_bpb={ckpt.get('val_bpb')}, params_M={ckpt.get('num_params_M'):.1f}, tokens_M={ckpt.get('total_tokens_M')}, steps={ckpt.get('num_steps')}", flush=True)

    prompts = args.prompts or MTG_PROMPTS
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*78}\nPROMPT {i}/{len(prompts)}: {prompt!r}\n{'-'*78}", flush=True)
        t0 = time.time()
        with autocast:
            text = generate_text(model, enc, prompt, args.n, args.temp, args.topk, device)
        dt = time.time() - t0
        print(text, flush=True)
        print(f"\n[{dt:.1f}s, {args.n} max toks, temp={args.temp}, top_k={args.topk}]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
