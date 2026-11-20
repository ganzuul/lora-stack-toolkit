#!/usr/bin/env python3
"""Local Dataset SFT & GRPO LoRA Trainer for Ternary-Bonsai-2-27B-PTQ1_0 on 8 GB VRAM.

Uses State-Carrying Tiled-BPTT (window_len=1024) + Chunked Full-Vocab LogProbs +
bitsandbytes AdamW8bit, exporting PEFT adapters and Q8_0 GGUFs via optimize_lora.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import bitsandbytes as bnb
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPTS = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PARENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPTS))

import bonsai_ptq_model as bpm  # noqa: E402
import lora_gguf_common as lgc  # noqa: E402

OPTIMIZE_LORA_PY = PARENT_SCRIPTS / "optimize_lora.py"
DEFAULT_BASE_GGUF = Path("/home/nos/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf")


def load_qwen35_tokenizer() -> Tokenizer:
    """Load the exact Qwen3.5 BPE tokenizer aligned with Ternary-Bonsai-2-27B."""
    try:
        tok_path = hf_hub_download("ccore/Qwen3.5-4B-3D-Coder", "tokenizer.json", local_files_only=True)
    except Exception:
        tok_path = hf_hub_download("ccore/Qwen3.5-4B-3D-Coder", "tokenizer.json")
    return Tokenizer.from_file(tok_path)


def render_chat_prefix(system: str, prompt: str, enable_thinking: bool) -> str:
    """Render ChatML prompt prefix matching qwen_chat_template.jinja."""
    base = (
        f"<|im_start|>system\n{system.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    return base if enable_thinking else f"{base}\n</think>\n\n"


def format_conversation(
    row: dict[str, Any],
    tok: Tokenizer,
    max_seq_len: int = 3072,
    truncate: bool = True,
) -> tuple[list[int], int]:
    """Render ChatML sequence matching qwen_chat_template.jinja and return (token_ids, prefix_len)."""
    if "prefix_text" in row and "completion_text" in row:
        p_ids = tok.encode(str(row["prefix_text"]), add_special_tokens=False).ids
        c_ids = tok.encode(str(row["completion_text"]), add_special_tokens=False).ids
        full_ids = p_ids + c_ids
        if truncate and len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
        return full_ids, len(p_ids)

    default_sys = (
        "You are an expert Blender 4.2 Python programmer. Generate clean, self-contained, executable `bpy` code "
        "compatible with Blender 4.2.10."
    )
    system = str(row.get("system") or default_sys).strip()
    prompt = str(row.get("prompt") or "").strip()
    if not row.get("preserve_prompt") and "/workspace/exports/final.blend" not in prompt:
        prompt += (
            "\nWrite a complete, standalone Blender 4.2.10 Python (`bpy`) script that builds this entire scene from scratch: "
            "clear existing objects via `for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)` "
            "(do not call `bpy.ops.mesh.delete` or `mode_set` on an empty scene), define `make_material(...)` and "
            "`assign_material(obj, mat)` using Blender 4.2 Principled BSDF names (`'Specular IOR Level'`, `'Coat Weight'`, `'Transmission Weight'`) "
            "and `obj.data.materials.append(mat)`, configure lighting, active camera, and Cycles (`64` samples), "
            "and save `/workspace/exports/final.blend`. Keep the script concise (under 130 lines) in one ```python block."
        )
    reasoning = str(row.get("reasoning") or "").strip()
    response = str(row.get("response") or "").strip()

    if reasoning:
        prefix_txt = render_chat_prefix(system, prompt, enable_thinking=True)
        comp_txt = f"{reasoning}\n</think>\n\n{response}<|im_end|>\n"
    else:
        prefix_txt = render_chat_prefix(system, prompt, enable_thinking=False)
        comp_txt = f"{response}<|im_end|>\n"

    p_ids = tok.encode(prefix_txt, add_special_tokens=False).ids
    c_ids = tok.encode(comp_txt, add_special_tokens=False).ids
    full_ids = p_ids + c_ids
    if truncate and len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
    return full_ids, len(p_ids)


@torch.no_grad()
def compute_sequence_logps(
    model: bpm.BonsaiPTQStudent,
    ids: list[int],
    prefix_len: int,
    window_len: int = 1024,
) -> torch.Tensor:
    """Compute per-completion-token log-probabilities [N_comp] without tracking gradients."""
    was_training = model.training
    model.eval()
    try:
        seq = torch.tensor(ids, dtype=torch.long, device=model.device)
        total_len = seq.shape[0]
        if total_len <= prefix_len:
            return torch.empty(0, dtype=torch.float32, device=model.device)

        states = model.init_states()
        out_logps: list[torch.Tensor] = []
        for w0 in range(0, total_len - 1, window_len):
            w1 = min(total_len - 1, w0 + window_len)
            win_in = seq[w0:w1].unsqueeze(0)
            win_tgt = seq[w0 + 1 : w1 + 1]
            hidden, states = model.forward_hidden_window(win_in, states, pos_offset=w0, use_checkpoint=False)
            tgt_pos = torch.arange(w0 + 1, w1 + 1, device=model.device)
            mask = tgt_pos >= prefix_len
            if mask.any():
                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                h_act = hidden[0, idx, :]
                t_act = win_tgt[idx]
                out_logps.append(model.chunked_token_logps(h_act, t_act))
        return torch.cat(out_logps, dim=0) if out_logps else torch.empty(0, dtype=torch.float32, device=model.device)
    finally:
        model.train(was_training)


def backward_sft_sequence(
    model: bpm.BonsaiPTQStudent,
    ids: list[int],
    prefix_len: int,
    weight: float = 1.0,
    window_len: int = 1024,
    grad_scale: float = 1.0,
) -> tuple[float, int]:
    """Run State-Carrying Tiled-BPTT SFT backward over one tokenized example."""
    seq = torch.tensor(ids, dtype=torch.long, device=model.device)
    total_len = seq.shape[0]
    n_comp = max(0, total_len - prefix_len)
    if n_comp == 0:
        return 0.0, 0

    states = model.init_states()
    total_loss_val = 0.0
    denom = float(n_comp) / float(grad_scale)

    for w0 in range(0, total_len - 1, window_len):
        w1 = min(total_len - 1, w0 + window_len)
        win_in = seq[w0:w1].unsqueeze(0)
        win_tgt = seq[w0 + 1 : w1 + 1].unsqueeze(0)
        tgt_pos = torch.arange(w0 + 1, w1 + 1, device=model.device)
        w_mask = ((tgt_pos >= prefix_len).float() * float(weight)).unsqueeze(0)

        hidden, states = model.forward_hidden_window(win_in, states, pos_offset=w0, use_checkpoint=True)
        if (w_mask != 0).any():
            loss_w = model.chunked_ce_loss(hidden, win_tgt, w_mask) / denom
            loss_w.backward()
            total_loss_val += float(loss_w.detach().item())
        torch.cuda.empty_cache()

    return total_loss_val, n_comp


def backward_grpo_sequence(
    model: bpm.BonsaiPTQStudent,
    ids: list[int],
    prefix_len: int,
    advantage: float,
    ref_logps: torch.Tensor,
    old_logps: torch.Tensor,
    kl_beta: float = 0.02,
    clip_eps: float = 0.2,
    window_len: int = 1024,
    grad_scale: float = 1.0,
) -> tuple[float, float, int]:
    """Run State-Carrying Tiled-BPTT GRPO clipped surrogate + k3 KL backward over one rollout."""
    seq = torch.tensor(ids, dtype=torch.long, device=model.device)
    total_len = seq.shape[0]
    n_comp = max(0, total_len - prefix_len)
    if n_comp == 0 or ref_logps.numel() != n_comp:
        return 0.0, 0.0, 0

    states = model.init_states()
    total_pg = 0.0
    total_kl = 0.0
    offset = 0
    adv_t = float(advantage)

    for w0 in range(0, total_len - 1, window_len):
        w1 = min(total_len - 1, w0 + window_len)
        win_in = seq[w0:w1].unsqueeze(0)
        win_tgt = seq[w0 + 1 : w1 + 1]
        tgt_pos = torch.arange(w0 + 1, w1 + 1, device=model.device)
        mask = tgt_pos >= prefix_len

        hidden, states = model.forward_hidden_window(win_in, states, pos_offset=w0, use_checkpoint=True)
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            k_len = int(idx.shape[0])
            ref_sl = ref_logps[offset : offset + k_len]
            old_sl = old_logps[offset : offset + k_len]
            offset += k_len

            logp_pi = model.chunked_token_logps(hidden[0, idx, :], win_tgt[idx])
            log_ratio = torch.clamp(logp_pi - old_sl, -5.0, 5.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            pg_term = -torch.minimum(surr1, surr2)

            diff_ref = torch.clamp(ref_sl - logp_pi, -10.0, 10.0)
            kl_term = torch.exp(diff_ref) - diff_ref - 1.0
            loss_w = ((pg_term + kl_beta * kl_term).sum() / float(n_comp)) * float(grad_scale)
            loss_w.backward()

            total_pg += float(pg_term.sum().detach().item()) / float(n_comp)
            total_kl += float(kl_term.sum().detach().item()) / float(n_comp)
        torch.cuda.empty_cache()

    return total_pg, total_kl, n_comp


def backward_dppo_sequence(
    model: bpm.BonsaiPTQStudent,
    ids: list[int],
    prefix_len: int,
    advantage: float,
    ref_logps: torch.Tensor,
    rollout_logps: torch.Tensor,
    kl_beta: float = 0.015,
    dppo_variant: str = "binary_tv",
    clip_low: float = 0.15,
    clip_high: float = 0.15,
    is_clip_c: float = 3.0,
    norm_tokens: float = 2048.0,
    window_len: int = 512,
    grad_scale: float = 1.0,
) -> tuple[float, float, float, int]:
    """Run State-Carrying Tiled-BPTT Stable-RL DPPO (Binary-TV / Binary-KL) + Dr.GRPO backward over one rollout."""
    if dppo_variant not in ("binary_tv", "binary_kl"):
        raise ValueError(f"Unsupported dppo_variant={dppo_variant!r}; expected 'binary_tv' or 'binary_kl'")

    seq = torch.tensor(ids, dtype=torch.long, device=model.device)
    total_len = seq.shape[0]
    n_comp = max(0, total_len - prefix_len)
    if n_comp == 0 or ref_logps.numel() != n_comp or rollout_logps.numel() != n_comp:
        return 0.0, 0.0, 0.0, 0

    states = model.init_states()
    total_pg = 0.0
    total_kl = 0.0
    masked_count = 0.0
    offset = 0
    adv_t = float(advantage)
    denom = float(max(1.0, norm_tokens))

    for w0 in range(0, total_len - 1, window_len):
        w1 = min(total_len - 1, w0 + window_len)
        win_in = seq[w0:w1].unsqueeze(0)
        win_tgt = seq[w0 + 1 : w1 + 1]
        tgt_pos = torch.arange(w0 + 1, w1 + 1, device=model.device)
        mask = tgt_pos >= prefix_len

        hidden, states = model.forward_hidden_window(win_in, states, pos_offset=w0, use_checkpoint=True)
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            k_len = int(idx.shape[0])
            ref_sl = ref_logps[offset : offset + k_len].to(device=model.device, dtype=torch.float32)
            roll_sl = rollout_logps[offset : offset + k_len].to(device=model.device, dtype=torch.float32)
            offset += k_len

            logp_pi = model.chunked_token_logps(hidden[0, idx, :], win_tgt[idx]).float()
            with torch.no_grad():
                p_pi = torch.exp(logp_pi.detach().clamp(min=-30.0, max=0.0))
                p_roll = torch.exp(roll_sl.clamp(min=-30.0, max=0.0))
                is_ratio = torch.exp(torch.clamp(logp_pi.detach() - roll_sl, -5.0, 5.0)).clamp(max=float(is_clip_c))
                if dppo_variant == "binary_tv":
                    diff_p = p_pi - p_roll
                    invalid = (diff_p > float(clip_high)) if adv_t > 0.0 else (diff_p < -float(clip_low))
                else:
                    p_r = p_roll.clamp(1e-7, 1.0 - 1e-7)
                    p_t = p_pi.clamp(1e-7, 1.0 - 1e-7)
                    kl_bin = p_r * (torch.log(p_r) - torch.log(p_t)) + (1.0 - p_r) * (
                        torch.log(1.0 - p_r) - torch.log(1.0 - p_t)
                    )
                    invalid = (
                        ((p_t > p_r) & (kl_bin > float(clip_high)))
                        if adv_t > 0.0
                        else ((p_t < p_r) & (kl_bin > float(clip_low)))
                    )
                valid_mask = (~invalid).float()
                masked_count += float(invalid.sum().item())

            pg_term = -(adv_t * is_ratio * valid_mask) * logp_pi
            if float(kl_beta) > 0.0:
                diff_ref = torch.clamp(ref_sl - logp_pi, -10.0, 10.0)
                kl_term = torch.exp(diff_ref) - diff_ref - 1.0
            else:
                kl_term = torch.zeros_like(logp_pi)

            loss_w = ((pg_term + float(kl_beta) * kl_term).sum() / denom) * float(grad_scale)
            if loss_w.requires_grad:
                loss_w.backward()

            total_pg += float(pg_term.sum().detach().item()) / denom
            total_kl += float(kl_term.sum().detach().item()) / denom
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_pg, total_kl, masked_count / float(n_comp), n_comp


def export_peft_and_gguf(
    model: bpm.BonsaiPTQStudent,
    peft_dir: Path,
    gguf_out: Path,
    base_hf_ref: str = "Qwen/Qwen3.8-27B",
    outtype: str = "q8_0",
) -> None:
    """Export trained PEFT weights and convert/verify to llama.cpp GGUF via optimize_lora.py."""
    model.export_peft_dir(peft_dir, base_ref=base_hf_ref)
    gguf_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(OPTIMIZE_LORA_PY),
        str(peft_dir),
        "--base",
        base_hf_ref,
        "-o",
        str(gguf_out),
        "--outtype",
        outtype,
        "--no-balance",
        "--device",
        "cpu",
        "--verify",
        "--local-files-only",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        cmd.remove("--local-files-only")
        subprocess.run(cmd, check=True)
    else:
        print(res.stdout.strip())


def _ensure_llama_swap_free() -> None:
    """Unload any resident llama-swap model so PyTorch training has full VRAM."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/running", timeout=2) as resp:
            running = json.loads(resp.read().decode("utf-8")).get("running", [])
        if running:
            req = urllib.request.Request("http://127.0.0.1:8080/api/models/unload", method="POST", data=b"")
            with urllib.request.urlopen(req, timeout=10):
                pass
            time.sleep(2.0)
    except Exception:
        pass


def run_sft(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty training dataset: {args.data}")

    tok = load_qwen35_tokenizer()
    samples: list[tuple[list[int], int, float, str]] = []
    skipped_long = 0
    for r in rows:
        ids, plen = format_conversation(r, tok, max_seq_len=args.max_seq_len, truncate=False)
        if len(ids) > args.max_seq_len:
            skipped_long += 1
            continue
        if len(ids) > plen + 8:
            samples.append((ids, plen, float(r.get("weight", 1.0)), str(r.get("id", "sample"))))
    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(
        f"[sft] loaded {len(samples)} complete samples (skipped_too_long={skipped_long}, max_seq_len={args.max_seq_len}, shuffle={args.shuffle})",
        flush=True,
    )

    _ensure_llama_swap_free()
    model = bpm.BonsaiPTQStudent(
        gguf_path=args.gguf,
        start_grad_layer=args.start_layer,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
    )
    if args.init_peft:
        model.load_peft_dir(args.init_peft)

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    print(f"[sft] trainable LoRA params: {n_trainable / 1e6:.2f}M across layers {args.start_layer}..63", flush=True)

    opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
    model.train()
    torch.cuda.reset_peak_memory_stats(model.device)

    total_updates = max(1, (len(samples) * args.epochs + args.grad_accum - 1) // args.grad_accum)
    if args.max_steps:
        total_updates = min(total_updates, args.max_steps)

    step_idx = 0
    opt.zero_grad(set_to_none=True)
    accum_count = 0
    accum_loss = 0.0
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        for s_idx, (ids, plen, weight, sid) in enumerate(samples):
            if step_idx >= total_updates:
                break
            t0 = time.perf_counter()
            loss_val, n_tok = backward_sft_sequence(
                model,
                ids,
                plen,
                weight=weight,
                window_len=args.window_len,
                grad_scale=1.0 / float(args.grad_accum),
            )
            dt = time.perf_counter() - t0
            accum_loss += loss_val
            accum_count += 1
            vram_mb = torch.cuda.max_memory_allocated(model.device) / (1024 * 1024)
            print(
                f"  ep={epoch} sample={s_idx + 1}/{len(samples)} id={sid} toks={len(ids)} (comp={n_tok}) "
                f"loss={loss_val * args.grad_accum:.4f} dt={dt:.1f}s peak_vram={vram_mb:.0f}MiB",
                flush=True,
            )

            if accum_count >= args.grad_accum or (epoch == args.epochs - 1 and s_idx == len(samples) - 1):
                # Linear warmup + cosine decay
                frac = min(1.0, float(step_idx + 1) / float(max(1, total_updates)))
                lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * lr_mult
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step_idx += 1
                print(
                    f"[step {step_idx}/{total_updates}] avg_loss={accum_loss:.4f} lr={args.lr * lr_mult:.2e} "
                    f"elapsed={time.perf_counter() - t_start:.1f}s",
                    flush=True,
                )
                accum_loss = 0.0
                accum_count = 0

    export_peft_and_gguf(model, args.peft_out, args.gguf_out, base_hf_ref=args.base_hf_ref, outtype="q8_0")
    print(f"[sft] DONE -> {args.gguf_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL training records.")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_BASE_GGUF)
    parser.add_argument("--base-hf-ref", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--peft-out", type=Path, default=SCRIPT_DIR / "work" / "peft_warmup")
    parser.add_argument("--gguf-out", type=Path, default=Path("/home/nos/models/Bonsai-2-27B-blender-warmup-Q8_0.gguf"))
    parser.add_argument("--init-peft", type=Path, default=None)
    parser.add_argument("--start-layer", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=3072)
    parser.add_argument("--window-len", type=int, default=512)
    parser.add_argument("--shuffle", action="store_true", help="Deterministically shuffle samples before slicing/training.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for --shuffle.")
    args = parser.parse_args()
    return run_sft(args)


if __name__ == "__main__":
    raise SystemExit(main())
