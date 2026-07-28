"""Binary/ternary sequence models: Encoder, Decoder, Seq2Seq, AutoEncoder.

Thesis-aligned: attention softmax + LayerNorm stay FP; FFN uses BinaryLinear /
TernaryLinear (STE train). Packed inference via ``bnn encode`` / wrap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers import BinaryLinear, TernaryLinear
from ..ste import clip_weights_


def _ffn(dim: int, ff: int, *, kind: str, bias: bool = False) -> nn.Sequential:
    if kind == "binary":
        return nn.Sequential(BinaryLinear(dim, ff, bias=bias), nn.ReLU(), BinaryLinear(ff, dim, bias=bias))
    if kind == "ternary":
        return nn.Sequential(TernaryLinear(dim, ff, bias=bias), nn.ReLU(), TernaryLinear(ff, dim, bias=bias))
    if kind == "fp":
        return nn.Sequential(nn.Linear(dim, ff), nn.ReLU(), nn.Linear(ff, dim))
    raise ValueError(f"ffn kind must be binary|ternary|fp, got {kind!r}")


class MultiHeadAttention(nn.Module):
    """FP multi-head attention (Q/K/V/proj). Softmax stays higher precision."""

    def __init__(self, dim: int, n_heads: int = 4, *, causal: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        # (3, B, H, T, Hd); unbind gives contiguous-friendly views for SDPA.
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Fused attention: avoids materialising the (B, H, T, T) score matrix and
        # needs no explicit causal mask, so nothing is allocated per forward.
        # Default scale is head_dim**-0.5, matching the previous manual scale.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    """Dedicated FP cross-attention (separate Q vs KV projections)."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        B, T, D = x.shape
        S = memory.size(1)
        q = self.q(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(memory).reshape(B, S, 2, self.n_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        # Cross-attention is never causal: the decoder may see all of memory.
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, ff: int = 256, *, ffn_kind: str = "binary"):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, causal=False)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = _ffn(dim, ff, kind=ffn_kind)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        ff: int = 256,
        *,
        ffn_kind: str = "binary",
        cross_attn: bool = True,
    ):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, n_heads, causal=True)
        self.cross_attn_enabled = cross_attn
        if cross_attn:
            self.n_cross = nn.LayerNorm(dim)
            self.cross = CrossAttention(dim, n_heads)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = _ffn(dim, ff, kind=ffn_kind)

    def forward(self, x: Tensor, memory: Tensor | None = None) -> Tensor:
        x = x + self.self_attn(self.n1(x))
        if self.cross_attn_enabled and memory is not None:
            x = x + self.cross(self.n_cross(x), memory)
        x = x + self.ffn(self.n2(x))
        return x


class BinaryTransformerEncoder(nn.Module):
    """Stack of self-attn (FP) + binary/ternary FFN encoder layers."""

    def __init__(
        self,
        vocab: int,
        dim: int = 64,
        depth: int = 2,
        n_heads: int = 4,
        ff: int = 128,
        max_len: int = 64,
        *,
        ffn_kind: str = "binary",
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(dim, n_heads, ff, ffn_kind=ffn_kind) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.max_len = max_len

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (B, T) long
        B, T = tokens.shape
        if self.max_len < T:
            raise ValueError(f"seq len {T} > max_len {self.max_len}")
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.embed(tokens) + self.pos(pos)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def clip_weights(self) -> None:
        clip_weights_(self)


class BinaryTransformerDecoder(nn.Module):
    """Causal decoder with optional cross-attention + binary/ternary FFN."""

    def __init__(
        self,
        vocab: int,
        dim: int = 64,
        depth: int = 2,
        n_heads: int = 4,
        ff: int = 128,
        max_len: int = 64,
        *,
        ffn_kind: str = "binary",
        cross_attn: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(dim, n_heads, ff, ffn_kind=ffn_kind, cross_attn=cross_attn)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.max_len = max_len
        self.cross_attn = cross_attn

    def forward(self, tokens: Tensor, memory: Tensor | None = None) -> Tensor:
        B, T = tokens.shape
        if self.max_len < T:
            raise ValueError(f"seq len {T} > max_len {self.max_len}")
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.embed(tokens) + self.pos(pos)
        for layer in self.layers:
            x = layer(x, memory)
        return self.head(self.norm(x))

    def clip_weights(self) -> None:
        clip_weights_(self)


class BinarySeq2Seq(nn.Module):
    """Encoder–Decoder for toy copy / reverse tasks."""

    def __init__(
        self,
        vocab: int,
        dim: int = 64,
        depth: int = 2,
        n_heads: int = 4,
        ff: int = 128,
        max_len: int = 32,
        *,
        ffn_kind: str = "binary",
    ):
        super().__init__()
        self.encoder = BinaryTransformerEncoder(
            vocab, dim, depth, n_heads, ff, max_len, ffn_kind=ffn_kind
        )
        self.decoder = BinaryTransformerDecoder(
            vocab, dim, depth, n_heads, ff, max_len, ffn_kind=ffn_kind, cross_attn=True
        )
        self.vocab = vocab
        self.max_len = max_len

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        memory = self.encoder(src)
        return self.decoder(tgt_in, memory)

    def clip_weights(self) -> None:
        self.encoder.clip_weights()
        self.decoder.clip_weights()


class BinaryAutoEncoder(nn.Module):
    """MLP autoencoder with binary bottleneck — encode/decode compression story.

    Input → FP stem → binary latent encode → binary decode → FP recon head.
    Demonstrates weight + activation binary path for reconstruction demos.
    """

    def __init__(self, n_in: int = 64, latent: int = 16, hidden: int = 64, *, ffn_kind: str = "binary"):
        super().__init__()
        self.stem = nn.Linear(n_in, hidden)
        # enc/dec vary with ffn_kind; declare the union.
        self.enc: BinaryLinear | TernaryLinear | nn.Linear
        self.dec: BinaryLinear | TernaryLinear | nn.Linear
        if ffn_kind == "binary":
            self.enc = BinaryLinear(hidden, latent)
            self.dec = BinaryLinear(latent, hidden)
        elif ffn_kind == "ternary":
            self.enc = TernaryLinear(hidden, latent)
            self.dec = TernaryLinear(latent, hidden)
        else:
            self.enc = nn.Linear(hidden, latent)
            self.dec = nn.Linear(latent, hidden)
        self.head = nn.Linear(hidden, n_in)
        self.n_in = n_in
        self.latent = latent

    def encode(self, x: Tensor) -> Tensor:
        return self.enc(F.relu(self.stem(x)))

    def decode(self, z: Tensor) -> Tensor:
        return self.head(F.relu(self.dec(z)))

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def clip_weights(self) -> None:
        clip_weights_(self)


def make_reverse_batch(
    batch: int,
    seq_len: int,
    vocab: int,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """Synthetic reverse task: tgt = reverse(src); teacher-forced tgt_in = BOS+tgt[:-1].

    Token 0 = PAD/BOS, tokens 1..vocab-1 are content. Returns src, tgt_in, tgt_out.
    """
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        src = torch.randint(1, vocab, (batch, seq_len), generator=g)
    else:
        src = torch.randint(1, vocab, (batch, seq_len))
    tgt = torch.flip(src, dims=[1])
    bos = torch.zeros(batch, 1, dtype=torch.long)
    tgt_in = torch.cat([bos, tgt[:, :-1]], dim=1)
    return src.to(device), tgt_in.to(device), tgt.to(device)


def seq2seq_token_accuracy(logits: Tensor, tgt: Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == tgt).float().mean().item())
