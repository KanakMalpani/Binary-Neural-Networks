"""Encoder / Decoder / Seq2Seq / AutoEncoder smoke tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bnn.cli import main as cli_main
from bnn.seq import (
    BinaryAutoEncoder,
    BinarySeq2Seq,
    BinaryTransformerDecoder,
    BinaryTransformerEncoder,
    make_reverse_batch,
    seq2seq_token_accuracy,
)


def test_encoder_forward_shape():
    enc = BinaryTransformerEncoder(vocab=16, dim=32, depth=1, n_heads=4, ff=64, max_len=16)
    tok = torch.randint(1, 16, (4, 8))
    out = enc(tok)
    assert out.shape == (4, 8, 32)


def test_decoder_causal_forward():
    dec = BinaryTransformerDecoder(
        vocab=16, dim=32, depth=1, n_heads=4, ff=64, max_len=16, cross_attn=False
    )
    tok = torch.randint(0, 16, (4, 8))
    logits = dec(tok)
    assert logits.shape == (4, 8, 16)


def test_seq2seq_train_step_and_acc():
    torch.manual_seed(0)
    model = BinarySeq2Seq(vocab=12, dim=32, depth=1, n_heads=4, ff=64, max_len=16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(30):
        src, tgt_in, tgt = make_reverse_batch(16, 6, 12, seed=step)
        opt.zero_grad(set_to_none=True)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, 12), tgt.reshape(-1))
        loss.backward()
        opt.step()
        model.clip_weights()
    src, tgt_in, tgt = make_reverse_batch(32, 6, 12, seed=999)
    with torch.no_grad():
        acc = seq2seq_token_accuracy(model(src, tgt_in), tgt)
    # Toy reverse should learn something above chance (~1/12) quickly
    assert acc > 0.25, f"token acc {acc} too low"


def test_autoencoder_recon_smoke():
    torch.manual_seed(0)
    ae = BinaryAutoEncoder(n_in=32, latent=8, hidden=32, ffn_kind="binary")
    opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
    for _ in range(40):
        x = torch.randn(16, 32)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(ae(x), x)
        loss.backward()
        opt.step()
        ae.clip_weights()
    with torch.no_grad():
        x = torch.randn(16, 32)
        mse = float(F.mse_loss(ae(x), x).item())
    assert mse < 2.0  # loose smoke — learns a bit


def test_cli_train_seq2seq_smoke(tmp_path):
    out = tmp_path / "s2s.json"
    code = cli_main(
        [
            "train-seq2seq",
            "--task",
            "both",
            "--steps",
            "25",
            "--batch",
            "16",
            "--seq-len",
            "6",
            "--dim",
            "32",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert out.is_file()
