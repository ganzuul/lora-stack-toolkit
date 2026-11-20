#!/usr/bin/env python3
"""Optimize LoRA adapters (PEFT safetensors/bin or GGUF) for llama.cpp / Bonsai-demo.

Supports:
  - Direct GGUF -> Q8_0 / F16 / BF16 / F32 optimization (no HF base weights needed)
  - PEFT dir -> GGUF with Qwen3.5 linear_attn V-head grouped->tiled reordering
    (including linear_attn.out_proj -> ssm_out.lora_a, which upstream convert_lora_to_gguf.py rejects)
  - Exact QR + r×r SVD rank reduction (--rank or adaptive --energy), singular-value balancing,
    near-zero tensor pruning (--min-norm), and post-quantization error verification (--verify)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lora_gguf_common as lgc  # noqa: E402


def _read_peft_dir(
    peft_dir: Path,
    base_override: str | None,
    hf_cache_dir: str | None,
    local_files_only: bool,
    include_visual: bool,
) -> tuple[str, float, list[tuple[str, torch.Tensor, torch.Tensor]], dict[str, Any], dict[str, Any]]:
    """Load PEFT LoRA tensors and map keys to canonical GGUF names."""
    cfg_path = peft_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in {peft_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        lparams = json.load(f)

    base_ref = base_override or lparams.get("base_model_name_or_path") or lparams.get("base_model")
    if not base_ref:
        raise ValueError("Neither --base nor base_model_name_or_path/base_model in adapter_config.json was found.")

    raw_cfg = lgc.load_hf_json_file(base_ref, "config.json", hf_cache_dir=hf_cache_dir, local_files_only=local_files_only)
    cfg = lgc.normalize_hf_config(raw_cfg)
    gguf_arch = lgc.infer_gguf_arch(cfg)
    n_layers = int(cfg["num_hidden_layers"])

    st_path = peft_dir / "adapter_model.safetensors"
    bin_path = peft_dir / "adapter_model.bin"
    if st_path.is_file():
        from safetensors.torch import load_file

        raw_tensors = load_file(str(st_path), device="cpu")
    elif bin_path.is_file():
        raw_tensors = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {peft_dir}")

    default_r = int(lparams.get("r", 1))
    default_alpha = float(lparams.get("lora_alpha", default_r))
    rank_pattern = lparams.get("rank_pattern") or {}
    alpha_pattern = lparams.get("alpha_pattern") or {}

    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    skipped: list[str] = []
    for key, tensor in raw_tensors.items():
        if ".lora_A." in key or key.endswith(".lora_embedding_A"):
            role = "a"
            stem = key.replace(".lora_A.weight", "").replace(".lora_A", "").replace(".lora_embedding_A", "")
        elif ".lora_B." in key or key.endswith(".lora_embedding_B"):
            role = "b"
            stem = key.replace(".lora_B.weight", "").replace(".lora_B", "").replace(".lora_embedding_B", "")
        elif ".base_layer." in key or "_norm." in key or key.endswith(".norm.weight"):
            skipped.append(key)
            continue
        else:
            raise ValueError(f"Unexpected PEFT tensor '{key}' (not lora_A/lora_B)")

        gguf_base = lgc.hf_weight_to_gguf_name(stem + ".weight", gguf_arch, n_layers, include_visual=include_visual)
        if gguf_base is None:
            skipped.append(key)
            continue
        grouped[gguf_base][role] = tensor
        grouped[gguf_base]["hf_stem"] = stem

    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    per_tensor_scale: dict[str, float] = {}
    for gguf_base in sorted(grouped.keys()):
        entry = grouped[gguf_base]
        if "a" not in entry or "b" not in entry:
            raise ValueError(f"Incomplete LoRA pair for '{gguf_base}'")
        A = entry["a"].to(torch.float32)
        B = entry["b"].to(torch.float32)
        r_i = int(A.shape[0])
        stem = entry["hf_stem"]
        # Match PEFT scaling resolution (exact or suffix-boundary match on pattern keys).
        stripped = stem.removeprefix("base_model.model.")
        alpha_i = next(
            (float(v) for k, v in alpha_pattern.items() if stripped == k or stripped.endswith(f".{k}")),
            default_alpha,
        )
        rank_pat = next(
            (int(v) for k, v in rank_pattern.items() if stripped == k or stripped.endswith(f".{k}")),
            r_i,
        )
        eff_scale = alpha_i / float(rank_pat)
        per_tensor_scale[gguf_base] = eff_scale
        pairs.append((gguf_base, A, B))

    meta = {
        "input_kind": "peft",
        "general_name": peft_dir.name,
        "base_ref": str(base_ref),
        "cfg": cfg,
        "lparams": lparams,
        "skipped": skipped,
        "per_tensor_scale": per_tensor_scale,
        "alora_tokens": lparams.get("alora_invocation_tokens"),
    }
    return gguf_arch, default_alpha, pairs, cfg, meta


def _read_gguf_lora(
    gguf_path: Path,
    base_override: str | None,
    hf_cache_dir: str | None,
    local_files_only: bool,
) -> tuple[str, float, list[tuple[str, torch.Tensor, torch.Tensor]], dict[str, Any] | None, dict[str, Any]]:
    """Load an existing GGUF LoRA file and convert tensors back to logical (A, B)."""
    gguf_mod = lgc.ensure_gguf()
    reader = gguf_mod.GGUFReader(str(gguf_path), mode="r")

    def get_str(k: str, default: str = "") -> str:
        return str(reader.fields[k].contents()) if k in reader.fields else default

    if get_str("general.type") != "adapter" or get_str("adapter.type") != "lora":
        raise ValueError(f"{gguf_path} is not a LoRA GGUF adapter.")

    arch = get_str("general.architecture")
    if not arch:
        raise ValueError(f"{gguf_path} is missing general.architecture.")
    alpha = 0.0
    for ak in ("adapter.lora.alpha", "adapter.lora_alpha"):
        if ak in reader.fields:
            alpha = float(reader.fields[ak].contents())
            break
    general_name = get_str("general.name") or gguf_path.stem
    alora_tokens = (
        [int(x) for x in reader.fields["adapter.alora.invocation_tokens"].contents()]
        if "adapter.alora.invocation_tokens" in reader.fields
        else None
    )

    by_base: dict[str, dict[str, Any]] = defaultdict(dict)
    for t in reader.tensors:
        if t.name.endswith(".lora_a"):
            by_base[t.name[: -len(".lora_a")]]["a"] = t
        elif t.name.endswith(".lora_b"):
            by_base[t.name[: -len(".lora_b")]]["b"] = t
        else:
            raise ValueError(f"Unsupported non-AB tensor '{t.name}' in GGUF adapter")

    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    per_tensor_scale: dict[str, float] = {}
    for gguf_base in sorted(by_base.keys()):
        entry = by_base[gguf_base]
        if "a" not in entry or "b" not in entry:
            raise ValueError(f"Incomplete GGUF LoRA pair for '{gguf_base}'")
        ta, tb = entry["a"], entry["b"]
        arr_a = lgc.dequantize_gguf_array(gguf_mod, np.array(ta.data, copy=False), ta.tensor_type)
        arr_b = lgc.dequantize_gguf_array(gguf_mod, np.array(tb.data, copy=False), tb.tensor_type)
        A, B = lgc.gguf_arrays_to_ab(gguf_base, arr_a, arr_b)
        rank_i = int(A.shape[0])
        per_tensor_scale[gguf_base] = (alpha / float(rank_i)) if alpha != 0.0 else 1.0
        pairs.append((gguf_base, A, B))

    cfg = None
    if base_override:
        raw_cfg = lgc.load_hf_json_file(base_override, "config.json", hf_cache_dir=hf_cache_dir, local_files_only=local_files_only)
        cfg = lgc.normalize_hf_config(raw_cfg)

    meta = {
        "input_kind": "gguf",
        "input_bytes": gguf_path.stat().st_size,
        "general_name": general_name,
        "per_tensor_scale": per_tensor_scale,
        "alora_tokens": alora_tokens,
    }
    return arch, alpha, pairs, cfg, meta


def _save_peft_dir(
    peft_out: Path,
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]],
    out_alpha: float,
    base_ref: str,
) -> None:
    """Save optimized logical (A, B) pairs as a PEFT adapter directory."""
    from safetensors.torch import save_file

    peft_out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, float] = {}
    ranks = sorted({int(A.shape[0]) for _, A, _ in pairs})
    base_r = ranks[-1]

    for gguf_base, A, B in pairs:
        hf_stem = lgc.gguf_base_to_hf_stem(gguf_base)
        r_i = int(A.shape[0])
        tensors[f"base_model.model.{hf_stem}.lora_A.weight"] = A.cpu().to(torch.float32).contiguous()
        tensors[f"base_model.model.{hf_stem}.lora_B.weight"] = B.cpu().to(torch.float32).contiguous()
        if len(ranks) > 1:
            short_key = hf_stem.removeprefix("model.language_model.")
            rank_pattern[short_key] = r_i
            alpha_pattern[short_key] = float(out_alpha)

    save_file(tensors, str(peft_out / "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_ref,
        "r": base_r,
        "lora_alpha": out_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
    }
    with open(peft_out / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _verify_written_gguf(
    out_path: Path,
    ref_pairs: list[tuple[str, torch.Tensor, torch.Tensor, float]],
) -> dict[str, float]:
    """Read back quantized tensors from disk and verify against unquantized reference pairs."""
    gguf_mod = lgc.ensure_gguf()
    reader = gguf_mod.GGUFReader(str(out_path), mode="r")
    alpha = float(reader.fields["adapter.lora.alpha"].contents())
    gmap = {t.name: t for t in reader.tensors}

    rel_errors: list[float] = []
    cos_sims: list[float] = []
    for gguf_base, A_ref, B_ref, scale_ref in ref_pairs:
        ta = gmap[f"{gguf_base}.lora_a"]
        tb = gmap[f"{gguf_base}.lora_b"]
        arr_a = lgc.dequantize_gguf_array(gguf_mod, np.array(ta.data, copy=False), ta.tensor_type)
        arr_b = lgc.dequantize_gguf_array(gguf_mod, np.array(tb.data, copy=False), tb.tensor_type)
        A_q, B_q = lgc.gguf_arrays_to_ab(gguf_base, arr_a, arr_b)
        rank_q = int(A_q.shape[0])
        scale_q = (alpha / float(rank_q)) if alpha != 0.0 else 1.0
        rel_err, cos = lgc.low_rank_pair_metrics(A_ref, B_ref, scale_ref, A_q, B_q, scale_q)
        rel_errors.append(rel_err)
        cos_sims.append(cos)

    return {
        "mean_rel_error": float(np.mean(rel_errors)) if rel_errors else 0.0,
        "max_rel_error": float(np.max(rel_errors)) if rel_errors else 0.0,
        "mean_cosine": float(np.mean(cos_sims)) if cos_sims else 1.0,
        "min_cosine": float(np.min(cos_sims)) if cos_sims else 1.0,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Optimize and quantize LoRA adapters (PEFT or GGUF -> Q8_0/Q5_0/Q4_0/MXFP4/F16/BF16/F32 GGUF)")
    ap.add_argument("input", type=Path, help="Input PEFT LoRA directory or .gguf LoRA file")
    ap.add_argument("-o", "--outfile", type=Path, required=True, help="Output .gguf path")
    ap.add_argument("--outtype", choices=list(lgc.SUPPORTED_OUTTYPES), default="q8_0", help="Target GGUF tensor type (default: q8_0)")
    ap.add_argument("--rank", type=int, default=None, help="Compress all tensors to max rank R via exact QR-SVD")
    ap.add_argument("--energy", type=float, default=None, help="Adaptive per-tensor rank retaining at least E fraction of Frobenius^2 energy (e.g. 0.95)")
    ap.add_argument("--min-rank", type=int, default=1, help="Minimum rank when using --energy (default: 1)")
    ap.add_argument("--max-rank", type=int, default=None, help="Maximum rank cap when using --energy")
    ap.add_argument("--snap-rank-mult", type=int, default=1, help="Round compressed ranks up to a multiple of M (e.g. 32 for full Q8_0 on lora_b)")
    ap.add_argument("--min-norm", type=float, default=0.0, help="Drop tensor pairs whose scaled Frobenius norm is below this threshold")
    ap.add_argument("--scale", type=float, default=1.0, help="Bake an extra multiplicative scale factor into the adapter (default: 1.0)")
    ap.add_argument("--balance", dest="balance", action="store_true", default=True, help="QR-SVD balance singular values between A and B (default: on)")
    ap.add_argument("--no-balance", dest="balance", action="store_false", help="Keep original A/B factorization when not changing rank")
    ap.add_argument("--base", type=str, default=None, help="Base HF model dir or repo ID (auto-read from adapter_config.json for PEFT; needed for --fix-vhead-order)")
    ap.add_argument("--hf-cache-dir", type=str, default=None, help="HuggingFace cache directory for config lookup")
    ap.add_argument("--local-files-only", action="store_true", help="Do not hit HuggingFace network")
    ap.add_argument("--no-vhead-reorder", action="store_true", help="Skip grouped->tiled V-head reorder on PEFT->GGUF conversion")
    ap.add_argument("--fix-vhead-order", action="store_true", help="Apply grouped->tiled V-head reorder to an existing unpermuted GGUF input (requires --base)")
    ap.add_argument("--include-visual", action="store_true", help="Include visual tower LoRA tensors if present in PEFT")
    ap.add_argument("--name", type=str, default=None, help="Override GGUF general.name metadata")
    ap.add_argument("--device", type=str, default="auto", help="Compute device for QR-SVD (auto, cpu, cuda, cuda:0)")
    ap.add_argument("--peft-out", type=Path, default=None, help="Optional directory to also export optimized PEFT weights")
    ap.add_argument("--verify", dest="verify", action="store_true", default=True, help="Verify dequantized output against input (default: on)")
    ap.add_argument("--no-verify", dest="verify", action="store_false", help="Skip verification")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.time()

    if args.rank is not None and args.energy is not None:
        raise SystemExit("Pass either --rank or --energy, not both.")
    if args.energy is not None and not (0.0 < args.energy <= 1.0):
        raise SystemExit("--energy must be in (0, 1].")

    inp = args.input
    if inp.is_dir():
        arch, in_alpha, raw_pairs, cfg, meta = _read_peft_dir(
            inp, args.base, args.hf_cache_dir, args.local_files_only, args.include_visual
        )
        do_vhead_reorder = not args.no_vhead_reorder
    elif inp.is_file():
        if args.fix_vhead_order and not args.base:
            raise SystemExit("--fix-vhead-order on GGUF input requires --base (dir or repo) to read head counts.")
        arch, in_alpha, raw_pairs, cfg, meta = _read_gguf_lora(
            inp, args.base, args.hf_cache_dir, args.local_files_only
        )
        do_vhead_reorder = args.fix_vhead_order
    else:
        raise SystemExit(f"Input path does not exist: {inp}")

    device = lgc.select_device(args.device, min_free_mb=256)
    compressing = (
        args.rank is not None
        or args.energy is not None
        or args.max_rank is not None
        or args.min_norm > 0.0
        or args.balance
    )

    out_pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    ref_pairs: list[tuple[str, torch.Tensor, torch.Tensor, float]] = []
    ranks_out: list[int] = []
    reordered_count = 0
    pruned_count = 0

    for gguf_base, A_in, B_in in raw_pairs:
        eff_scale = meta["per_tensor_scale"][gguf_base] * float(args.scale)
        A_work = A_in.to(device)
        B_work = B_in.to(device)

        if do_vhead_reorder:
            A_work, B_work, reordered = lgc.maybe_reorder_qwen35_v_heads(gguf_base, A_work, B_work, cfg)
            if reordered:
                reordered_count += 1

        # Keep a reference copy in post-reorder layout so verification isolates SVD+quant error cleanly.
        if args.verify:
            ref_pairs.append((gguf_base, A_work.cpu(), B_work.cpu(), eff_scale))

        if compressing:
            A_opt, B_opt, info = lgc.compress_ab_qr_svd(
                A_work,
                B_work,
                scale_in=eff_scale,
                target_rank=args.rank,
                energy=args.energy,
                min_rank=args.min_rank,
                max_rank=args.max_rank,
                snap_mult=args.snap_rank_mult,
                balance=args.balance,
            )
            if info["orig_norm"] < args.min_norm:
                pruned_count += 1
                continue
        else:
            # No QR-SVD: keep input factors, folding eff_scale cleanly when non-unit.
            A_opt, B_opt = A_work, B_work
            if eff_scale != 1.0:
                sign = 1.0 if eff_scale >= 0 else -1.0
                mag = abs(eff_scale) ** 0.5
                B_opt = B_opt * (sign * mag)
                A_opt = A_opt * mag

        ranks_out.append(int(A_opt.shape[0]))
        out_pairs.append((gguf_base, A_opt.cpu(), B_opt.cpu()))

    if not out_pairs:
        raise SystemExit("All tensors were pruned; nothing to write.")

    # When factors represent exact unscaled delta = B_opt @ A_opt, setting alpha = max_rank
    # keeps scale=1.0 for uniform rank and cleanly triggers write_lora_gguf's per-layer compensation otherwise.
    out_alpha = float(max(ranks_out))
    gname = args.name or meta["general_name"]
    write_info = lgc.write_lora_gguf(
        out_path=args.outfile,
        arch=arch,
        lora_alpha=out_alpha,
        pairs=out_pairs,
        outtype=args.outtype,
        general_name=gname,
        alora_tokens=meta.get("alora_tokens"),
    )

    if args.peft_out is not None:
        base_ref = meta.get("base_ref") or args.base or f"unknown-{arch}"
        _save_peft_dir(args.peft_out, out_pairs, out_alpha, str(base_ref))

    elapsed = time.time() - t_start
    print(f"Input      : {inp} ({meta['input_kind']}, {len(raw_pairs)} pairs, alpha={in_alpha})")
    print(f"Output     : {write_info['out_path']} ({write_info['file_bytes']/1024**2:.2f} MiB, {write_info['n_pairs']} pairs, alpha={out_alpha:g})")
    print(f"Ranks      : min={min(ranks_out)} max={max(ranks_out)} | pruned={pruned_count} | vhead_reordered={reordered_count}")
    print(f"Quant types: {write_info['qtype_counts']} (F16 fallbacks: {write_info['fallback_count']})")
    if write_info["fallback_sample"]:
        print(f"Fallback eg: {write_info['fallback_sample']}")
    if "input_bytes" in meta:
        saved_pct = 100.0 * (1.0 - write_info["file_bytes"] / meta["input_bytes"])
        print(f"Size delta : {meta['input_bytes']/1024**2:.2f} MiB -> {write_info['file_bytes']/1024**2:.2f} MiB ({saved_pct:+.1f}% saved)")

    if args.verify:
        kept_names = {p[0] for p in out_pairs}
        kept_refs = [r for r in ref_pairs if r[0] in kept_names]
        v = _verify_written_gguf(args.outfile, kept_refs)
        print(
            f"Verified   : mean_rel_err={v['mean_rel_error']:.5f} max_rel_err={v['max_rel_error']:.5f} "
            f"| mean_cos={v['mean_cosine']:.6f} min_cos={v['min_cosine']:.6f}"
        )
    print(f"Elapsed    : {elapsed:.2f}s (device={device})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
