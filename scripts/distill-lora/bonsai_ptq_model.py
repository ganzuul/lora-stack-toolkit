#!/usr/bin/env python3
"""Exact PyTorch + Triton PTQ1_0 + prism.hadamard Bonsai-2 (qwen35) student runtime & LoRA adapter.

Loads packed PTQ1_0 ternary GGUF weights directly (~5.38 GB blocks + 0.28 GB output head),
keeping token_embd on CPU and dequantizing one Linear weight at a time into a shared FP16
scratch buffer inside gradient-checkpointed blocks. Supports primal-basis LoRA compatible
with llama.cpp / optimize_lora.py and State-Carrying Tiled-BPTT over long sequences.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure GPU 0 maps to RTX 2070 SUPER and expandable segments prevent fragmentation before CUDA initializes.
os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPTS = SCRIPT_DIR.parent
DEFAULT_MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/home/nos/models"))
DEFAULT_BASE_GGUF = Path(
    os.environ.get("BASE_GGUF", str(DEFAULT_MODEL_DIR / "Ternary-Bonsai-2-27B-PTQ1_0.gguf"))
)
if str(PARENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPTS))

import lora_gguf_common as lgc  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    l2norm as q35_l2norm,
    rotate_half,
    torch_chunk_gated_delta_rule,
)

QK_PTQ1_0 = 128
BYTES_PTQ1_0 = 28
GGML_TYPE_PTQ1_0 = 143
GGML_TYPE_BF16 = 30
GGML_TYPE_F32 = 0

# Precomputed element -> (byte_offset_in_block, pow3_multiplier) lookup for QK_PTQ1_0=128.
_PTQ_BYTE_OFF: list[int] = []
_PTQ_POW3: list[int] = []
_POW3_TABLE = (1, 3, 9, 27, 81)
for _e in range(QK_PTQ1_0):
    if _e < 80:
        _PTQ_BYTE_OFF.append(_e & 15)
        _PTQ_POW3.append(_POW3_TABLE[_e >> 4])
    elif _e < 120:
        _t = _e - 80
        _PTQ_BYTE_OFF.append(16 + (_t & 7))
        _PTQ_POW3.append(_POW3_TABLE[_t >> 3])
    else:
        _t = _e - 120
        _PTQ_BYTE_OFF.append(24 + (_t & 1))
        _PTQ_POW3.append(_POW3_TABLE[_t >> 1])

_PTQ_LUT_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_ptq_luts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached (byte_off[128], pow3[128]) tensors on target device."""
    if device not in _PTQ_LUT_CACHE:
        bo = torch.tensor(_PTQ_BYTE_OFF, dtype=torch.int32, device=device)
        p3 = torch.tensor(_PTQ_POW3, dtype=torch.int32, device=device)
        _PTQ_LUT_CACHE[device] = (bo, p3)
    return _PTQ_LUT_CACHE[device]


@triton.jit
def _dequant_ptq1_0_kernel(
    in_ptr,
    out_ptr,
    bo_ptr,
    p3_ptr,
    n_blocks,
    BLOCK_N: tl.constexpr,
):
    """Dequantize packed 28-byte PTQ1_0 blocks into 128 FP16 values per block."""
    pid = tl.program_id(0)
    b = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    b_mask = b < n_blocks
    e = tl.arange(0, 128)

    bo = tl.load(bo_ptr + e)
    p3 = tl.load(p3_ptr + e)

    base = b * 28
    d_lo = tl.load(in_ptr + base + 26, mask=b_mask, other=0).to(tl.uint16)
    d_hi = tl.load(in_ptr + base + 27, mask=b_mask, other=0).to(tl.uint16)
    d = (d_lo | (d_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

    raw = tl.load(in_ptr + base[:, None] + bo[None, :], mask=b_mask[:, None], other=0).to(tl.uint32)
    q = (raw * p3[None, :].to(tl.uint32)) & 0xFF
    trit = ((q * 3) >> 8).to(tl.int32) - 1
    vals = (trit.to(tl.float32) * d[:, None]).to(tl.float16)

    tl.store(out_ptr + b[:, None] * 128 + e[None, :], vals, mask=b_mask[:, None])


def dequantize_ptq1_0_triton(
    packed: torch.Tensor,
    out_buf: torch.Tensor | None = None,
    out_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Dequantize a GPU uint8 PTQ1_0 buffer [out_dim, (in_dim//128)*28] into FP16 [out_dim, in_dim]."""
    assert packed.is_cuda and packed.dtype == torch.uint8
    if out_shape is None:
        out_dim = packed.shape[0]
        in_dim = (packed.shape[1] // BYTES_PTQ1_0) * QK_PTQ1_0
    else:
        out_dim, in_dim = out_shape
    n_elem = out_dim * in_dim
    assert n_elem % QK_PTQ1_0 == 0
    n_blocks = n_elem // QK_PTQ1_0

    if out_buf is None:
        out = torch.empty((out_dim, in_dim), dtype=torch.float16, device=packed.device)
    else:
        out = out_buf.view(-1)[:n_elem].view(out_dim, in_dim)

    bo, p3 = _get_ptq_luts(packed.device)
    BLOCK_N = 16
    grid = (triton.cdiv(n_blocks, BLOCK_N),)
    _dequant_ptq1_0_kernel[grid](packed.reshape(-1), out.reshape(-1), bo, p3, n_blocks, BLOCK_N=BLOCK_N)
    return out


def dequantize_ptq1_0_numpy(packed_u8: np.ndarray) -> np.ndarray:
    """Exact CPU numpy reference decoder matching dequantize_row_ptq1_0 in ggml-quants.c."""
    flat = np.ascontiguousarray(packed_u8, dtype=np.uint8).reshape(-1, BYTES_PTQ1_0)
    nb = flat.shape[0]
    d = flat[:, 26:28].view(np.float16).reshape(nb, 1).astype(np.float32)
    bo = np.array(_PTQ_BYTE_OFF, dtype=np.int64)
    p3 = np.array(_PTQ_POW3, dtype=np.uint32)
    raw = flat[:, bo].astype(np.uint32)
    q = (raw * p3[None, :]) & 0xFF
    trit = ((q * 3) >> 8).astype(np.int32) - 1
    out = trit.astype(np.float32) * d
    if packed_u8.ndim >= 2:
        in_dim = (packed_u8.shape[1] // BYTES_PTQ1_0) * QK_PTQ1_0
        return out.reshape(packed_u8.shape[0], in_dim)
    return out.reshape(-1)


def make_walsh_hadamard(block_size: int, device: torch.device, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Build normalized Sylvester-Walsh-Hadamard matrix matching llama-model.cpp."""
    idx = torch.arange(block_size, dtype=torch.int64)
    and_mat = idx[:, None] & idx[None, :]
    # Compute bit parity of 10-bit integers
    parity = torch.zeros_like(and_mat)
    tmp = and_mat
    for _ in range(int(math.log2(max(2, block_size))) + 1):
        parity ^= tmp & 1
        tmp >>= 1
    scale = 1.0 / math.sqrt(float(block_size))
    mat = torch.where(parity == 1, -scale, scale).to(dtype=dtype, device=device)
    return mat


def apply_hadamard(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Apply block-diagonal normalized Walsh-Hadamard transform along last dimension."""
    bs = H.shape[0]
    lead = x.shape[:-1]
    d = x.shape[-1]
    return (x.reshape(*lead, d // bs, bs) @ H).reshape(*lead, d)


def apply_gdn_v_perm(x: torch.Tensor, nk: int, rep: int, hd: int, inverse: bool = False) -> torch.Tensor:
    """Permute between tiled [rep, nk, hd] and grouped [nk, rep, hd] GDN V-head ordering."""
    lead = x.shape[:-1]
    if not inverse:
        # tiled [rep, nk, hd] -> grouped [nk, rep, hd]
        return x.reshape(*lead, rep, nk, hd).transpose(-3, -2).contiguous().reshape(*lead, rep * nk * hd)
    # grouped [nk, rep, hd] -> tiled [rep, nk, hd]
    return x.reshape(*lead, nk, rep, hd).transpose(-3, -2).contiguous().reshape(*lead, rep * nk * hd)


class _PTQ1_0LinearFn(torch.autograd.Function):
    """Autograd op: Hadamard-rotate input x -> dequantize W into scratch -> FP16 GEMM."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        packed_w: torch.Tensor,
        scratch_buf: torch.Tensor,
        in_features: int,
        out_features: int,
        H: torch.Tensor | None,
        signs: torch.Tensor | None,
        gdn_geom: tuple[int, int, int] | None,
    ) -> torch.Tensor:
        x_rot = x
        if gdn_geom is not None:
            nk, rep, hd = gdn_geom
            x_rot = apply_gdn_v_perm(x_rot, nk, rep, hd, inverse=False)
        if signs is not None:
            x_rot = x_rot * signs
        if H is not None:
            x_rot = apply_hadamard(x_rot, H)

        w_fp16 = dequantize_ptq1_0_triton(packed_w, out_buf=scratch_buf, out_shape=(out_features, in_features))
        out = F.linear(x_rot, w_fp16)

        ctx.needs_x_grad = ctx.needs_input_grad[0]
        ctx.save_for_backward(packed_w, H, signs)
        ctx.scratch_buf = scratch_buf
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.gdn_geom = gdn_geom
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        if not ctx.needs_x_grad:
            return (None,) * 8
        packed_w, H, signs = ctx.saved_tensors
        w_fp16 = dequantize_ptq1_0_triton(
            packed_w, out_buf=ctx.scratch_buf, out_shape=(ctx.out_features, ctx.in_features)
        )
        grad_x_rot = grad_out @ w_fp16  # [..., out] @ [out, in] -> [..., in]
        grad_x = grad_x_rot
        if H is not None:
            # H is symmetric orthogonal (H^T = H)
            grad_x = apply_hadamard(grad_x, H)
        if signs is not None:
            grad_x = grad_x * signs
        if ctx.gdn_geom is not None:
            nk, rep, hd = ctx.gdn_geom
            grad_x = apply_gdn_v_perm(grad_x, nk, rep, hd, inverse=True)
        return grad_x, None, None, None, None, None, None, None


class PTQLinearWithLoRA(nn.Module):
    """Packed PTQ1_0 Linear layer with optional prism.hadamard input rotation & primal LoRA."""

    def __init__(
        self,
        gguf_name: str,
        packed_u8: torch.Tensor,
        in_features: int,
        out_features: int,
        scratch_buf: torch.Tensor,
        H: torch.Tensor | None = None,
        signs: torch.Tensor | None = None,
        gdn_geom: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.gguf_name = gguf_name
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("packed_w", packed_u8, persistent=False)
        self.scratch_buf = scratch_buf
        self.H = H
        self.signs = signs
        self.gdn_geom = gdn_geom
        self.lora_a: nn.Parameter | None = None
        self.lora_b: nn.Parameter | None = None
        self.lora_scale: float = 0.0
        self.adapter_enabled: bool = True

    def attach_lora(self, rank: int, alpha: float) -> None:
        dev = self.packed_w.device
        # Float32 trainable LoRA adapters (tiny ~150MB, avoids FP16 underflow) cast cleanly in forward
        a = torch.empty((rank, self.in_features), dtype=torch.float32, device=dev)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5.0))
        b = torch.zeros((self.out_features, rank), dtype=torch.float32, device=dev)
        self.lora_a = nn.Parameter(a)
        self.lora_b = nn.Parameter(b)
        self.lora_scale = float(alpha) / float(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = _PTQ1_0LinearFn.apply(
            x, self.packed_w, self.scratch_buf, self.in_features, self.out_features, self.H, self.signs, self.gdn_geom
        )
        if self.adapter_enabled and self.lora_a is not None and self.lora_b is not None:
            # Primal basis LoRA (llama-graph.cpp#L1643-L1650): runs on unrotated x
            delta = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
            base = base + (self.lora_scale * delta)
        return base


class BF16Linear(nn.Module):
    """Frozen BF16 tensor stored as FP16 on device (used for ssm_alpha / ssm_beta)."""

    def __init__(self, weight_fp16: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight_fp16, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


def rms_norm_f32(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GGUF RMSNorm (weights already include +1.0 offset)."""
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


class BonsaiGDNBlock(nn.Module):
    """Qwen3.5 Gated DeltaNet + SwiGLU MLP block matching qwen35.cpp."""

    def __init__(
        self,
        layer_idx: int,
        hparams: dict[str, Any],
        attn_norm: torch.Tensor,
        post_attn_norm: torch.Tensor,
        attn_qkv: PTQLinearWithLoRA,
        attn_gate: PTQLinearWithLoRA,
        ssm_out: PTQLinearWithLoRA,
        ssm_alpha: BF16Linear,
        ssm_beta: BF16Linear,
        ssm_conv1d: torch.Tensor,
        ssm_a: torch.Tensor,
        ssm_dt_bias: torch.Tensor,
        ssm_norm: torch.Tensor,
        ffn_gate: PTQLinearWithLoRA,
        ffn_up: PTQLinearWithLoRA,
        ffn_down: PTQLinearWithLoRA,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.eps = float(hparams["rms_norm_eps"])
        self.nk = int(hparams["ssm_n_group"])
        self.nv = int(hparams["ssm_dt_rank"])
        self.dk = int(hparams["ssm_d_state"])
        self.dv = int(hparams["ssm_d_inner"]) // self.nv
        self.d_conv = int(hparams["ssm_d_conv"])

        self.register_buffer("attn_norm", attn_norm, persistent=False)
        self.register_buffer("post_attn_norm", post_attn_norm, persistent=False)
        self.register_buffer("ssm_conv1d", ssm_conv1d, persistent=False)  # [C, 1, K]
        self.register_buffer("ssm_a", ssm_a, persistent=False)
        self.register_buffer("ssm_dt_bias", ssm_dt_bias, persistent=False)
        self.register_buffer("ssm_norm", ssm_norm, persistent=False)

        self.attn_qkv = attn_qkv
        self.attn_gate = attn_gate
        self.ssm_out = ssm_out
        self.ssm_alpha = ssm_alpha
        self.ssm_beta = ssm_beta
        self.ffn_gate = ffn_gate
        self.ffn_up = ffn_up
        self.ffn_down = ffn_down

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        rec_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        bsz, seqlen, _ = x.shape
        h = rms_norm_f32(x, self.attn_norm, self.eps)

        qkv = self.attn_qkv(h)
        z = self.attn_gate(h).reshape(bsz, seqlen, self.nv, self.dv)
        beta = torch.sigmoid(self.ssm_beta(h).float())
        alpha = self.ssm_alpha(h).float()
        g = F.softplus(alpha + self.ssm_dt_bias.float()) * self.ssm_a.float()

        # Causal depthwise conv1d
        qkv_t = qkv.transpose(1, 2).float()  # [B, C, L]
        pad_k = self.d_conv - 1
        if conv_state is not None:
            conv_in = torch.cat([conv_state.float(), qkv_t], dim=-1)
        else:
            conv_in = F.pad(qkv_t, (pad_k, 0))
        new_conv_state = conv_in[:, :, -pad_k:].detach().to(torch.float16) if pad_k > 0 else None
        conv_out = F.conv1d(conv_in, self.ssm_conv1d.float(), bias=None, groups=qkv.shape[-1])
        conv_out = F.silu(conv_out).transpose(1, 2)  # [B, L, C]

        k_dim = self.nk * self.dk
        v_dim = self.nv * self.dv
        q_c, k_c, v_c = torch.split(conv_out, [k_dim, k_dim, v_dim], dim=-1)
        q_c = q35_l2norm(q_c.reshape(bsz, seqlen, self.nk, self.dk), dim=-1, eps=self.eps)
        k_c = q35_l2norm(k_c.reshape(bsz, seqlen, self.nk, self.dk), dim=-1, eps=self.eps)
        v_c = v_c.reshape(bsz, seqlen, self.nv, self.dv)

        rep = self.nv // self.nk
        if rep > 1:
            # llama.cpp ggml_repeat_4d tiles Q/K along heads matching tiled GGUF V-head order
            q_c = q_c.repeat(1, 1, rep, 1)
            k_c = k_c.repeat(1, 1, rep, 1)

        out_rec, new_rec_state = torch_chunk_gated_delta_rule(
            q_c,
            k_c,
            v_c,
            g=g,
            beta=beta,
            chunk_size=64,
            initial_state=rec_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        new_rec_state = new_rec_state.detach() if new_rec_state is not None else None

        # Gated RMSNorm matching build_norm_gated: rms_norm(out, ssm_norm) * silu(z)
        out_norm = rms_norm_f32(out_rec.float(), self.ssm_norm, self.eps) * F.silu(z.float())
        out_flat = out_norm.reshape(bsz, seqlen, v_dim).to(x.dtype)
        x = x + self.ssm_out(out_flat)

        h2 = rms_norm_f32(x, self.post_attn_norm, self.eps)
        mlp_out = self.ffn_down(F.silu(self.ffn_gate(h2)) * self.ffn_up(h2))
        x = x + mlp_out
        return x, new_conv_state, new_rec_state


def _apply_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_offset: int,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3.5 NeoX partial RoPE on the leading n_rot=64 dims of [B, H, L, head_dim]."""
    _, _, seqlen, head_dim = q.shape
    rot_dim = inv_freq.shape[0] * 2
    pos = torch.arange(pos_offset, pos_offset + seqlen, device=q.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [L, rot_dim//2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [L, rot_dim]
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    q_rot, q_pass = q[..., :rot_dim].float(), q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim].float(), k[..., rot_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_out = torch.cat([q_rot.to(q.dtype), q_pass], dim=-1)
    k_out = torch.cat([k_rot.to(k.dtype), k_pass], dim=-1)
    return q_out, k_out


def _group_chunked_causal_sdpa(
    q: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    prefix_len: int,
    q_chunk: int = 256,
) -> torch.Tensor:
    """Group+query-tiled causal SDPA preventing Turing sm_75 multi-GB Math SDPA OOM."""
    bsz, n_q, l_q, head_dim = q.shape
    _, n_kv, l_kv, _ = k_all.shape
    gsize = n_q // n_kv
    q_grouped = q.view(bsz, n_kv, gsize, l_q, head_dim)
    scale = 1.0 / math.sqrt(float(head_dim))

    group_outs = []
    for g_idx in range(n_kv):
        kg = k_all[:, g_idx : g_idx + 1].expand(bsz, gsize, l_kv, head_dim)
        vg = v_all[:, g_idx : g_idx + 1].expand(bsz, gsize, l_kv, head_dim)
        q_pieces = []
        for s in range(0, l_q, q_chunk):
            e = min(l_q, s + q_chunk)
            q_sl = q_grouped[:, g_idx, :, s:e, :]
            # Build causal mask for rows [prefix_len + s .. prefix_len + e - 1] over keys [0 .. prefix_len + e - 1]
            k_slice = kg[:, :, : prefix_len + e, :]
            v_slice = vg[:, :, : prefix_len + e, :]
            q_pos = torch.arange(prefix_len + s, prefix_len + e, device=q.device)[:, None]
            k_pos = torch.arange(0, prefix_len + e, device=q.device)[None, :]
            mask = q_pos >= k_pos
            q_pieces.append(F.scaled_dot_product_attention(q_sl, k_slice, v_slice, attn_mask=mask, scale=scale))
        group_outs.append(torch.cat(q_pieces, dim=2))  # [B, gsize, l_q, head_dim]
    return torch.cat(group_outs, dim=1)  # [B, n_q, l_q, head_dim]


class BonsaiFullAttnBlock(nn.Module):
    """Qwen3.5 Full Gated-Attention + SwiGLU MLP block matching qwen35.cpp."""

    def __init__(
        self,
        layer_idx: int,
        hparams: dict[str, Any],
        inv_freq: torch.Tensor,
        attn_norm: torch.Tensor,
        post_attn_norm: torch.Tensor,
        attn_q_norm: torch.Tensor,
        attn_k_norm: torch.Tensor,
        attn_q: PTQLinearWithLoRA,
        attn_k: PTQLinearWithLoRA,
        attn_v: PTQLinearWithLoRA,
        attn_output: PTQLinearWithLoRA,
        ffn_gate: PTQLinearWithLoRA,
        ffn_up: PTQLinearWithLoRA,
        ffn_down: PTQLinearWithLoRA,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.eps = float(hparams["rms_norm_eps"])
        self.n_head = int(hparams["n_head"])
        self.n_kv = int(hparams["n_head_kv"])
        self.head_dim = int(hparams["head_dim"])

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("attn_norm", attn_norm, persistent=False)
        self.register_buffer("post_attn_norm", post_attn_norm, persistent=False)
        self.register_buffer("attn_q_norm", attn_q_norm, persistent=False)
        self.register_buffer("attn_k_norm", attn_k_norm, persistent=False)

        self.attn_q = attn_q
        self.attn_k = attn_k
        self.attn_v = attn_v
        self.attn_output = attn_output
        self.ffn_gate = ffn_gate
        self.ffn_up = ffn_up
        self.ffn_down = ffn_down

    def forward(
        self,
        x: torch.Tensor,
        pos_offset: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape
        h = rms_norm_f32(x, self.attn_norm, self.eps)

        # attn_q outputs interleaved [query, gate] per head: [B, L, n_head, 2, head_dim]
        qg = self.attn_q(h).view(bsz, seqlen, self.n_head, 2, self.head_dim)
        q = qg[:, :, :, 0, :]
        gate = qg[:, :, :, 1, :].reshape(bsz, seqlen, self.n_head * self.head_dim)

        k = self.attn_k(h).view(bsz, seqlen, self.n_kv, self.head_dim)
        v = self.attn_v(h).view(bsz, seqlen, self.n_kv, self.head_dim)

        q = rms_norm_f32(q, self.attn_q_norm, self.eps).transpose(1, 2)  # [B, Hq, L, D]
        k = rms_norm_f32(k, self.attn_k_norm, self.eps).transpose(1, 2)  # [B, Hkv, L, D]
        v = v.transpose(1, 2)  # [B, Hkv, L, D]

        q, k = _apply_partial_rope(q, k, pos_offset, self.inv_freq)

        prefix_len = k_cache.shape[2] if k_cache is not None else 0
        k_all = torch.cat([k_cache, k], dim=2) if k_cache is not None else k
        v_all = torch.cat([v_cache, v], dim=2) if v_cache is not None else v

        attn_out = _group_chunked_causal_sdpa(q, k_all, v_all, prefix_len=prefix_len)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, seqlen, self.n_head * self.head_dim)
        attn_out = attn_out * torch.sigmoid(gate.float()).to(x.dtype)
        x = x + self.attn_output(attn_out)

        h2 = rms_norm_f32(x, self.post_attn_norm, self.eps)
        x = x + self.ffn_down(F.silu(self.ffn_gate(h2)) * self.ffn_up(h2))
        return x, k_all.detach(), v_all.detach()


class ChunkedTokenLogProbsFn(torch.autograd.Function):
    """Differentiable per-token log P(target | hidden) across full vocab in 16k-row slices (~180 MB peak)."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,  # [N, D]
        targets: torch.Tensor,  # [N] int64
        packed_head: torch.Tensor,  # [V, ...] uint8 PTQ1_0 on GPU
        scratch_buf: torch.Tensor,
        H: torch.Tensor | None,
        signs: torch.Tensor | None,
        vocab_size: int,
        hidden_dim: int,
        vocab_chunk: int = 16384,
    ) -> torch.Tensor:
        with torch.no_grad():
            h_rot = hidden
            if signs is not None:
                h_rot = h_rot * signs
            if H is not None:
                h_rot = apply_hadamard(h_rot, H)

            n_tok = h_rot.shape[0]
            max_l = torch.full((n_tok,), -1e9, dtype=torch.float32, device=hidden.device)
            sum_exp = torch.zeros((n_tok,), dtype=torch.float32, device=hidden.device)
            target_logits = torch.zeros((n_tok,), dtype=torch.float32, device=hidden.device)

            for v0 in range(0, vocab_size, vocab_chunk):
                v1 = min(vocab_size, v0 + vocab_chunk)
                w_sub = dequantize_ptq1_0_triton(
                    packed_head[v0:v1], out_buf=scratch_buf, out_shape=(v1 - v0, hidden_dim)
                )
                logits_sub = F.linear(h_rot, w_sub).float()
                chunk_max = logits_sub.max(dim=-1).values
                new_max = torch.maximum(max_l, chunk_max)
                sum_exp = sum_exp * torch.exp(max_l - new_max) + torch.exp(logits_sub - new_max[:, None]).sum(dim=-1)
                max_l = new_max

                in_chunk = (targets >= v0) & (targets < v1)
                if in_chunk.any():
                    idx = torch.nonzero(in_chunk, as_tuple=False).squeeze(-1)
                    target_logits[idx] = logits_sub[idx, targets[idx] - v0]

            lse = max_l + torch.log(sum_exp.clamp_min(1e-30))
            logps = target_logits - lse

        ctx.save_for_backward(h_rot, targets, lse, packed_head, H, signs)
        ctx.scratch_buf = scratch_buf
        ctx.vocab_size = vocab_size
        ctx.hidden_dim = hidden_dim
        ctx.vocab_chunk = vocab_chunk
        return logps

    @staticmethod
    def backward(ctx: Any, grad_logps: torch.Tensor) -> tuple[Any, ...]:
        h_rot, targets, lse, packed_head, H, signs = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return (None,) * 9
        # d(z_t - lse) / d(z_v) = 1{v=t} - p_v  =>  grad_z = -grad_logps * (p_v - 1{v=t})
        neg_g = -grad_logps.float()
        grad_h_rot = torch.zeros_like(h_rot, dtype=torch.float32)

        with torch.no_grad():
            for v0 in range(0, ctx.vocab_size, ctx.vocab_chunk):
                v1 = min(ctx.vocab_size, v0 + ctx.vocab_chunk)
                w_sub = dequantize_ptq1_0_triton(
                    packed_head[v0:v1], out_buf=ctx.scratch_buf, out_shape=(v1 - v0, ctx.hidden_dim)
                )
                logits_sub = F.linear(h_rot, w_sub).float()
                probs_sub = torch.exp(logits_sub - lse[:, None])
                in_chunk = (targets >= v0) & (targets < v1)
                if in_chunk.any():
                    idx = torch.nonzero(in_chunk, as_tuple=False).squeeze(-1)
                    probs_sub[idx, targets[idx] - v0] -= 1.0
                grad_logits = (probs_sub * neg_g[:, None]).to(torch.float16)
                grad_h_rot += (grad_logits @ w_sub).float()

            grad_h = grad_h_rot.to(h_rot.dtype)
            if H is not None:
                grad_h = apply_hadamard(grad_h, H)
            if signs is not None:
                grad_h = grad_h * signs
        return grad_h, None, None, None, None, None, None, None, None


class BonsaiPTQStudent(nn.Module):
    """Complete Ternary-Bonsai-2-27B PTQ1_0 runtime + primal LoRA trainer."""

    def __init__(
        self,
        gguf_path: str | Path = DEFAULT_BASE_GGUF,
        device: torch.device | None = None,
        start_grad_layer: int = 32,
        target_modules: tuple[str, ...] = (
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
        ),
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
    ):
        super().__init__()
        self.gguf_path = Path(gguf_path)
        self.device = device or lgc.select_device("auto", min_free_mb=5000)
        if self.device.type != "cuda":
            raise RuntimeError(f"Expected CUDA GPU (RTX 2070 SUPER), got {self.device}")
        self.start_grad_layer = int(start_grad_layer)
        self.target_modules = set(target_modules)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)

        gguf_mod = lgc.ensure_gguf()
        reader = gguf_mod.GGUFReader(str(self.gguf_path), mode="r")
        self.hparams = self._read_hparams(reader)
        self.blocks_n = int(self.hparams["n_layer"])

        # Shared single FP16 scratch matrix large enough for 17408x5120 or 16384x5120 (~178 MB)
        max_scratch_elems = max(
            int(self.hparams["n_ff"]) * int(self.hparams["n_embd"]),
            16384 * int(self.hparams["n_embd"]),
        )
        self.scratch_buf = torch.empty(max_scratch_elems, dtype=torch.float16, device=self.device)

        # Hadamard rotations & signs
        bs = int(self.hparams["hadamard_block_size"])
        self.H = make_walsh_hadamard(bs, self.device, dtype=torch.float16)
        self.signs: dict[int, torch.Tensor] = {}
        for width, vec in self.hparams["hadamard_signs"].items():
            self.signs[width] = torch.tensor(vec, dtype=torch.float16, device=self.device)
        self.hadamard_weights = set(self.hparams["hadamard_weight_names"])
        self.hadamard_inverses = set(self.hparams["hadamard_inverse_weight_names"])
        self.gdn_v_grouped = bool(self.hparams["hadamard_gdn_v_grouped"])

        tmap = {t.name: t for t in reader.tensors}

        # Keep token_embd on CPU (~278 MB saved on VRAM)
        tok_t = tmap["token_embd.weight"]
        self.token_embd_cpu = np.array(tok_t.data, copy=True)

        self.register_buffer(
            "output_norm",
            torch.from_numpy(np.array(tmap["output_norm.weight"].data, dtype=np.float32, copy=True)).to(self.device),
            persistent=False,
        )
        out_t = tmap["output.weight"]
        self.register_buffer(
            "output_packed",
            torch.from_numpy(np.array(out_t.data, dtype=np.uint8, copy=True)).to(self.device),
            persistent=False,
        )

        # Partial RoPE inverse frequencies (rot_dim = 64, base = 1e7)
        rot_dim = int(self.hparams["rope_n_rot"])
        freq_base = float(self.hparams["rope_freq_base"])
        inv_freq = 1.0 / (
            freq_base ** (torch.arange(0, rot_dim, 2, dtype=torch.int64, device=self.device).float() / rot_dim)
        )

        def make_ptq_lin(tname: str, in_dim: int, out_dim: int) -> PTQLinearWithLoRA:
            t = tmap[tname]
            packed = torch.from_numpy(np.array(t.data, dtype=np.uint8, copy=True)).to(self.device)
            use_h = tname in self.hadamard_weights
            H = self.H if use_h else None
            signs = self.signs.get(in_dim) if use_h else None
            gdn_geom = None
            if use_h and self.gdn_v_grouped and ".ssm_out." in tname:
                nk = int(self.hparams["ssm_n_group"])
                nv = int(self.hparams["ssm_dt_rank"])
                hd = in_dim // nv
                gdn_geom = (nk, nv // nk, hd)
            return PTQLinearWithLoRA(tname, packed, in_dim, out_dim, self.scratch_buf, H=H, signs=signs, gdn_geom=gdn_geom)

        def get_f32(tname: str) -> torch.Tensor:
            return torch.from_numpy(np.array(tmap[tname].data, dtype=np.float32, copy=True)).to(self.device)

        def get_bf16_as_fp16(tname: str) -> BF16Linear:
            raw = np.array(tmap[tname].data, dtype=np.uint8, copy=True)
            rows = raw.shape[0]
            t_bf16 = torch.from_numpy(raw).view(torch.bfloat16).reshape(rows, -1).to(device=self.device, dtype=torch.float16)
            return BF16Linear(t_bf16)

        n_embd = int(self.hparams["n_embd"])
        n_ff = int(self.hparams["n_ff"])
        full_interval = int(self.hparams["full_attn_interval"])
        gdn_k_dim = int(self.hparams["ssm_n_group"]) * int(self.hparams["ssm_d_state"])
        gdn_v_dim = int(self.hparams["ssm_d_inner"])
        gdn_qkv_dim = gdn_k_dim * 2 + gdn_v_dim
        full_q_out = int(self.hparams["n_head"]) * 2 * int(self.hparams["head_dim"])
        full_kv_out = int(self.hparams["n_head_kv"]) * int(self.hparams["head_dim"])

        self.layers = nn.ModuleList()
        self.lora_modules: list[PTQLinearWithLoRA] = []
        for il in range(self.blocks_n):
            is_full = ((il + 1) % full_interval) == 0
            a_norm = get_f32(f"blk.{il}.attn_norm.weight")
            p_norm = get_f32(f"blk.{il}.post_attention_norm.weight")
            fg = make_ptq_lin(f"blk.{il}.ffn_gate.weight", n_embd, n_ff)
            fu = make_ptq_lin(f"blk.{il}.ffn_up.weight", n_embd, n_ff)
            fd = make_ptq_lin(f"blk.{il}.ffn_down.weight", n_ff, n_embd)

            if not is_full:
                qkv = make_ptq_lin(f"blk.{il}.attn_qkv.weight", n_embd, gdn_qkv_dim)
                zgate = make_ptq_lin(f"blk.{il}.attn_gate.weight", n_embd, gdn_v_dim)
                sout = make_ptq_lin(f"blk.{il}.ssm_out.weight", gdn_v_dim, n_embd)
                conv = get_f32(f"blk.{il}.ssm_conv1d.weight").unsqueeze(1)  # [10240, 1, 4]
                blk: nn.Module = BonsaiGDNBlock(
                    il,
                    self.hparams,
                    a_norm,
                    p_norm,
                    qkv,
                    zgate,
                    sout,
                    get_bf16_as_fp16(f"blk.{il}.ssm_alpha.weight"),
                    get_bf16_as_fp16(f"blk.{il}.ssm_beta.weight"),
                    conv,
                    get_f32(f"blk.{il}.ssm_a"),
                    get_f32(f"blk.{il}.ssm_dt.bias"),
                    get_f32(f"blk.{il}.ssm_norm.weight"),
                    fg,
                    fu,
                    fd,
                )
                candidates = {"attn_qkv": qkv, "attn_gate": zgate, "ssm_out": sout, "ffn_gate": fg, "ffn_up": fu, "ffn_down": fd}
            else:
                aq = make_ptq_lin(f"blk.{il}.attn_q.weight", n_embd, full_q_out)
                ak = make_ptq_lin(f"blk.{il}.attn_k.weight", n_embd, full_kv_out)
                av = make_ptq_lin(f"blk.{il}.attn_v.weight", n_embd, full_kv_out)
                ao = make_ptq_lin(f"blk.{il}.attn_output.weight", gdn_v_dim, n_embd)
                blk = BonsaiFullAttnBlock(
                    il,
                    self.hparams,
                    inv_freq,
                    a_norm,
                    p_norm,
                    get_f32(f"blk.{il}.attn_q_norm.weight"),
                    get_f32(f"blk.{il}.attn_k_norm.weight"),
                    aq,
                    ak,
                    av,
                    ao,
                    fg,
                    fu,
                    fd,
                )
                candidates = {
                    "attn_q": aq,
                    "attn_k": ak,
                    "attn_v": av,
                    "attn_output": ao,
                    "ffn_gate": fg,
                    "ffn_up": fu,
                    "ffn_down": fd,
                }

            self.layers.append(blk)
            if lora_rank > 0 and il >= self.start_grad_layer:
                for short_name, mod in candidates.items():
                    if short_name in self.target_modules:
                        mod.attach_lora(self.lora_rank, self.lora_alpha)
                        self.lora_modules.append(mod)

    @staticmethod
    def _read_hparams(reader: Any) -> dict[str, Any]:
        def gf(k: str, default: Any = None) -> Any:
            return reader.fields[k].contents() if k in reader.fields else default

        widths = [int(x) for x in gf("prism.hadamard.sign_widths", [])]
        vals = [int(x) for x in gf("prism.hadamard.sign_values", [])]
        signs_map: dict[int, list[int]] = {}
        off = 0
        for w in widths:
            signs_map[w] = vals[off : off + w]
            off += w

        return {
            "arch": str(gf("general.architecture")),
            "vocab_size": len(gf("tokenizer.ggml.tokens", [])),
            "n_layer": int(gf("qwen35.block_count")),
            "n_embd": int(gf("qwen35.embedding_length")),
            "n_ff": int(gf("qwen35.feed_forward_length")),
            "n_head": int(gf("qwen35.attention.head_count")),
            "n_head_kv": int(gf("qwen35.attention.head_count_kv")),
            "head_dim": int(gf("qwen35.attention.key_length")),
            "full_attn_interval": int(gf("qwen35.full_attention_interval", 4)),
            "rms_norm_eps": float(gf("qwen35.attention.layer_norm_rms_epsilon", 1e-6)),
            "rope_n_rot": int(gf("qwen35.rope.dimension_count", 64)),
            "rope_freq_base": float(gf("qwen35.rope.freq_base", 10_000_000.0)),
            "ssm_d_conv": int(gf("qwen35.ssm.conv_kernel")),
            "ssm_d_inner": int(gf("qwen35.ssm.inner_size")),
            "ssm_d_state": int(gf("qwen35.ssm.state_size")),
            "ssm_dt_rank": int(gf("qwen35.ssm.time_step_rank")),
            "ssm_n_group": int(gf("qwen35.ssm.group_count")),
            "hadamard_block_size": int(gf("prism.hadamard.block_size", 1024)),
            "hadamard_signs": signs_map,
            "hadamard_weight_names": [str(x) for x in gf("prism.hadamard.weight_names", [])],
            "hadamard_inverse_weight_names": [str(x) for x in gf("prism.hadamard.inverse_weight_names", [])],
            "hadamard_gdn_v_grouped": bool(gf("prism.hadamard.gdn_v_grouped", False)),
        }

    def set_adapters_enabled(self, enabled: bool) -> None:
        for m in self.lora_modules:
            m.adapter_enabled = enabled

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Lookup rows from CPU token_embd.weight, decode PTQ1_0, and invert Hadamard rotation."""
        bsz, seqlen = input_ids.shape
        ids_np = input_ids.detach().cpu().numpy().reshape(-1)
        rows_u8 = self.token_embd_cpu[ids_np]
        emb_np = dequantize_ptq1_0_numpy(rows_u8)
        e = torch.from_numpy(emb_np).to(device=self.device, dtype=torch.float16).reshape(bsz, seqlen, -1)
        if "token_embd.weight" in self.hadamard_inverses:
            # llama-model / qwen35.cpp#L743: first H, then signs
            e = apply_hadamard(e, self.H)
            n_embd = int(self.hparams["n_embd"])
            if n_embd in self.signs:
                e = e * self.signs[n_embd]
        return e

    def init_states(self) -> list[Any]:
        return [None] * self.blocks_n

    def forward_hidden_window(
        self,
        input_ids: torch.Tensor,
        states: list[Any],
        pos_offset: int,
        use_checkpoint: bool = True,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Run all 64 blocks on a token window [B, L_win], carrying detached hybrid states."""
        x = self.embed_tokens(input_ids)
        next_states: list[Any] = [None] * self.blocks_n

        for il, blk in enumerate(self.layers):
            st = states[il]
            states[il] = None
            in_grad = self.training and (il >= self.start_grad_layer) and (len(self.lora_modules) > 0)
            if isinstance(blk, BonsaiGDNBlock):
                cs, rs = (None, None) if st is None else st
                if in_grad and use_checkpoint:
                    x, ncs, nrs = torch.utils.checkpoint.checkpoint(
                        blk, x, cs, rs, use_reentrant=False
                    )
                elif in_grad:
                    x, ncs, nrs = blk(x, cs, rs)
                else:
                    with torch.no_grad():
                        x, ncs, nrs = blk(x, cs, rs)
                next_states[il] = (ncs, nrs)
            else:
                kc, vc = (None, None) if st is None else st
                if in_grad and use_checkpoint:
                    x, nkc, nvc = torch.utils.checkpoint.checkpoint(
                        blk, x, pos_offset, kc, vc, use_reentrant=False
                    )
                elif in_grad:
                    x, nkc, nvc = blk(x, pos_offset, kc, vc)
                else:
                    with torch.no_grad():
                        x, nkc, nvc = blk(x, pos_offset, kc, vc)
                next_states[il] = (nkc, nvc)

        normed = rms_norm_f32(x, self.output_norm, float(self.hparams["rms_norm_eps"]))
        return normed, next_states

    @torch.no_grad()
    def forward_logits_last(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return full FP32 logits [B, vocab_size] for the final position of input_ids."""
        states = self.init_states()
        normed, _ = self.forward_hidden_window(input_ids, states, pos_offset=0, use_checkpoint=False)
        h_last = normed[:, -1, :]  # [B, D]
        n_embd = int(self.hparams["n_embd"])
        if "output.weight" in self.hadamard_weights:
            h_last = apply_hadamard(h_last * self.signs[n_embd], self.H)
        vocab = int(self.hparams["vocab_size"])
        chunks = []
        for v0 in range(0, vocab, 16384):
            v1 = min(vocab, v0 + 16384)
            w_sub = dequantize_ptq1_0_triton(
                self.output_packed[v0:v1], out_buf=self.scratch_buf, out_shape=(v1 - v0, n_embd)
            )
            chunks.append(F.linear(h_last, w_sub).float())
        return torch.cat(chunks, dim=-1)

    def chunked_token_logps(
        self,
        hidden_active: torch.Tensor,  # [N_act, D]
        targets_active: torch.Tensor,  # [N_act] int64
        sub_chunk: int = 256,
    ) -> torch.Tensor:
        """Compute differentiable target log-probabilities in small token & vocab slices."""
        n_embd = int(self.hparams["n_embd"])
        vocab = int(self.hparams["vocab_size"])
        use_h = "output.weight" in self.hadamard_weights
        H = self.H if use_h else None
        signs = self.signs.get(n_embd) if use_h else None
        parts = []
        for s in range(0, hidden_active.shape[0], sub_chunk):
            e = min(hidden_active.shape[0], s + sub_chunk)
            parts.append(
                ChunkedTokenLogProbsFn.apply(
                    hidden_active[s:e], targets_active[s:e], self.output_packed, self.scratch_buf, H, signs, vocab, n_embd
                )
            )
        return torch.cat(parts, dim=0)

    def chunked_ce_loss(
        self,
        hidden: torch.Tensor,  # [B, L, D]
        targets: torch.Tensor,  # [B, L]
        loss_weights: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """Compute full-vocab weighted cross-entropy over active positions in low VRAM."""
        n_embd = int(self.hparams["n_embd"])
        h_flat = hidden.reshape(-1, n_embd)
        t_flat = targets.reshape(-1)
        w_flat = loss_weights.reshape(-1).float()
        active = w_flat != 0.0
        if not active.any():
            return hidden.sum() * 0.0
        idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        logps = self.chunked_token_logps(h_flat[idx], t_flat[idx])
        return -(logps * w_flat[idx]).sum()

    def export_peft_dir(self, out_dir: str | Path, base_ref: str = "Qwen/Qwen3.8-27B") -> Path:
        """Save trained primal LoRA weights as PEFT adapter_model.safetensors + adapter_config.json."""
        from safetensors.torch import save_file

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        target_hf: set[str] = set()
        for mod in self.lora_modules:
            assert mod.lora_a is not None and mod.lora_b is not None
            a_t = mod.lora_a.detach().cpu().to(torch.float32).contiguous()
            b_t = mod.lora_b.detach().cpu().to(torch.float32).contiguous()
            # Match HF PEFT convention (convert GGUF tiled GDN ssm_out -> HF grouped if needed)
            if self.gdn_v_grouped and ".ssm_out." in mod.gguf_name:
                nk = int(self.hparams["ssm_n_group"])
                nv = int(self.hparams["ssm_dt_rank"])
                hd = a_t.shape[1] // nv
                a_t = apply_gdn_v_perm(a_t, nk, nv // nk, hd, inverse=False)
            hf_stem = lgc.gguf_base_to_hf_stem(mod.gguf_name)
            target_hf.add(hf_stem.rsplit(".", 1)[-1])
            tensors[f"base_model.model.{hf_stem}.lora_A.weight"] = a_t
            tensors[f"base_model.model.{hf_stem}.lora_B.weight"] = b_t

        save_file(tensors, str(out_path / "adapter_model.safetensors"))
        cfg = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": base_ref,
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": sorted(target_hf),
        }
        (out_path / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return out_path

    def load_peft_dir(self, peft_dir: str | Path) -> None:
        """Load trained LoRA A/B weights back from a PEFT safetensors directory."""
        from safetensors.torch import load_file

        tensors = load_file(str(Path(peft_dir) / "adapter_model.safetensors"), device="cpu")
        mod_map = {m.gguf_name: m for m in self.lora_modules}
        loaded = 0
        for gname, mod in mod_map.items():
            hf_stem = lgc.gguf_base_to_hf_stem(gname)
            ka = f"base_model.model.{hf_stem}.lora_A.weight"
            kb = f"base_model.model.{hf_stem}.lora_B.weight"
            if ka in tensors and kb in tensors and mod.lora_a is not None and mod.lora_b is not None:
                a_t = tensors[ka].to(dtype=torch.float32, device=self.device)
                b_t = tensors[kb].to(dtype=torch.float32, device=self.device)
                if self.gdn_v_grouped and ".ssm_out." in gname:
                    nk = int(self.hparams["ssm_n_group"])
                    nv = int(self.hparams["ssm_dt_rank"])
                    hd = a_t.shape[1] // nv
                    a_t = apply_gdn_v_perm(a_t, nk, nv // nk, hd, inverse=True)
                with torch.no_grad():
                    mod.lora_a.copy_(a_t)
                    mod.lora_b.copy_(b_t)
                loaded += 1
        if loaded == 0:
            raise RuntimeError(f"Loaded 0 LoRA pairs from {peft_dir}")


def test_triton_ptq_decoder(gguf_path: Path) -> None:
    """Verify Triton PTQ1_0 dequantizer matches NumPy C reference bit-for-bit."""
    dev = lgc.select_device("auto", min_free_mb=1000)
    rng = np.random.default_rng(42)
    # 1) Synthetic random blocks (random trits + scales)
    rand_u8 = rng.integers(0, 256, size=(128, 8 * BYTES_PTQ1_0), dtype=np.uint8)
    # Keep scales well-formed finite FP16 in normal range
    scales = rng.uniform(-0.2, 0.2, size=(128 * 8,)).astype(np.float16).view(np.uint8).reshape(128, 8, 2)
    rand_u8.reshape(128, 8, BYTES_PTQ1_0)[:, :, 26:28] = scales
    ref = dequantize_ptq1_0_numpy(rand_u8).astype(np.float16)
    got = dequantize_ptq1_0_triton(torch.from_numpy(rand_u8).to(dev)).cpu().numpy()
    max_diff = float(np.max(np.abs(ref.astype(np.float32) - got.astype(np.float32))))
    assert max_diff == 0.0, f"Synthetic PTQ1_0 mismatch: max_diff={max_diff}"

    # 2) Real GGUF tensor blk.0.ffn_gate.weight
    gguf_mod = lgc.ensure_gguf()
    r = gguf_mod.GGUFReader(str(gguf_path), "r")
    t = next(x for x in r.tensors if x.name == "blk.0.ffn_gate.weight")
    real_u8 = np.array(t.data, dtype=np.uint8, copy=True)
    ref_real = dequantize_ptq1_0_numpy(real_u8).astype(np.float16)
    got_real = dequantize_ptq1_0_triton(torch.from_numpy(real_u8).to(dev)).cpu().numpy()
    max_diff_real = float(np.max(np.abs(ref_real.astype(np.float32) - got_real.astype(np.float32))))
    assert max_diff_real == 0.0, f"Real GGUF PTQ1_0 mismatch: max_diff={max_diff_real}"
    print(f"TRITON_PTQ1_0_PASS synthetic_diff={max_diff} real_ffn_gate_diff={max_diff_real} shape={got_real.shape}")


def _check_llama_swap_free() -> None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/running", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("running"):
            raise RuntimeError(f"llama-swap currently holds VRAM ({data['running']}); unload first.")
    except urllib.error.URLError:
        pass


def verify_equivalence(
    gguf_path: Path,
    prompt: str = "<|im_start|>user\nWrite a Blender Python script to build a wooden chair.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\nimport bpy\n",
) -> None:
    """Compare logits of BonsaiPTQStudent directly against live Bonsai-demo llama-server."""
    _check_llama_swap_free()
    llama_bin = Path(os.environ.get(
        "LLAMA_SERVER_BIN",
        "/run/media/nos/games/inference_engines/Bonsai-demo/bin/cuda/llama-server",
    ))
    port = 18089
    cmd = [
        str(llama_bin),
        "-m", str(gguf_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", "999",
        "-ot", "token_embd.weight=CPU",
        "-c", "2048",
        "-np", "1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"llama-server exited early ({proc.returncode}): {err[-1000:]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise TimeoutError("llama-server health timeout")

        def post_json(ep: str, body: dict[str, Any]) -> dict[str, Any]:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/{ep}",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))

        toks = post_json("tokenize", {"content": prompt, "add_special": False})["tokens"]
        comp = post_json(
            "completion",
            {
                "prompt": toks,
                "n_predict": 1,
                "n_probs": 20,
                "post_sampling_probs": False,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
                "cache_prompt": False,
            },
        )
        top_probs = comp["completion_probabilities"][0]["top_logprobs"]
        ref_ids = [int(x["id"]) for x in top_probs]
        ref_logps = np.array([float(x["logprob"]) for x in top_probs], dtype=np.float32)
        ref_toks = [x["token"] for x in top_probs]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model = BonsaiPTQStudent(gguf_path=gguf_path, lora_rank=0)
    model.eval()
    t_load = time.perf_counter() - t0

    ids_t = torch.tensor([toks], dtype=torch.long, device=model.device)
    logits = model.forward_logits_last(ids_t)[0]  # [V]
    logps_all = F.log_softmax(logits.float(), dim=-1)
    pt_ref_logps = logps_all[torch.tensor(ref_ids, device=model.device)].cpu().numpy()

    top1_pt = int(torch.argmax(logps_all).item())
    top5_pt = torch.topk(logps_all, 5).indices.tolist()
    cos = float(np.dot(ref_logps, pt_ref_logps) / (np.linalg.norm(ref_logps) * np.linalg.norm(pt_ref_logps)))
    pearson = float(np.corrcoef(ref_logps, pt_ref_logps)[0, 1])
    max_abs = float(np.max(np.abs(ref_logps - pt_ref_logps)))
    vram_mb = torch.cuda.max_memory_allocated(model.device) / (1024 * 1024)

    print(f"EQUIVALENCE load={t_load:.1f}s peak_vram={vram_mb:.0f}MiB prompt_tokens={len(toks)}")
    print(f"  llama.cpp top1 id={ref_ids[0]} ({ref_toks[0]!r}) logp={ref_logps[0]:.4f}")
    print(f"  pytorch   top1 id={top1_pt} top5={top5_pt} logp(ref_top1)={pt_ref_logps[0]:.4f}")
    print(f"  top20 cos={cos:.6f} pearson={pearson:.6f} max_abs_logp_diff={max_abs:.4f}")
    assert top1_pt == ref_ids[0] or ref_ids[0] in top5_pt, f"Top-1 mismatch: {top1_pt} vs {ref_ids[0]}"
    assert cos >= 0.995, f"Cosine too low: {cos}"
    print("EQUIVALENCE VERIFICATION PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_BASE_GGUF)
    parser.add_argument("--test-triton", action="store_true")
    parser.add_argument("--verify-equivalence", action="store_true")
    args = parser.parse_args()

    if args.test_triton:
        test_triton_ptq_decoder(args.gguf)
    if args.verify_equivalence:
        verify_equivalence(args.gguf)
    if not (args.test_triton or args.verify_equivalence):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
