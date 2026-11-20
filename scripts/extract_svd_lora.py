#!/usr/bin/env python3
"""Extract a rank-compressed SVD LoRA from ΔW = W_finetune - W_base (ThinkingCap-style).

Designed for 24 GB host RAM / 8 GB VRAM workstations:
  - Streams safetensors shards one pair at a time via safe_open (low host RAM)
  - Optional ref-counted cache eviction (--evict-shards) when disk space is tight
  - Default GPU Gram-eigh SVD (~16x faster than GPU full SVD, ~87x faster than CPU,
    exact full singular spectrum + ~99.5% subspace fidelity) with exact & lowrank modes
  - Supports uniform --rank or adaptive --energy per-tensor rank selection
  - Emits PEFT artifacts (adapter_model.safetensors, adapter_config.json,
    extraction_stats.json, validation_report.json) and optional quantized GGUF
    with proper Qwen3.5 V-head reordering (including linear_attn.out_proj -> ssm_out)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lora_gguf_common as lgc  # noqa: E402

SKIP_SUFFIXES = (
    ".bias",
    ".A_log",
    ".dt_bias",
    ".conv1d.weight",
    "norm.weight",
    "layernorm.weight",
    "rotary_emb.inv_freq",
)


def _resolve_model_index(
    source: str,
    hf_cache_dir: str | None,
    local_files_only: bool,
) -> tuple[bool, Path | None, dict[str, str], dict[str, Path]]:
    """Return (is_local, local_dir, weight_map[key->shard], local_shard_paths)."""
    p = Path(source)
    if p.is_dir():
        idx = p / "model.safetensors.index.json"
        if idx.is_file():
            with open(idx, "r", encoding="utf-8") as f:
                wmap = json.load(f)["weight_map"]
            return True, p, wmap, {s: p / s for s in set(wmap.values())}
        single = p / "model.safetensors"
        if single.is_file():
            from safetensors import safe_open

            with safe_open(str(single), framework="pt", device="cpu") as f:
                wmap = {k: "model.safetensors" for k in f.keys()}
            return True, p, wmap, {"model.safetensors": single}
        raise FileNotFoundError(f"No model.safetensors(.index.json) in {p}")

    # HF Hub repo
    try:
        idx_json = lgc.load_hf_json_file(
            source, "model.safetensors.index.json", hf_cache_dir=hf_cache_dir, local_files_only=local_files_only
        )
        wmap = idx_json["weight_map"]
        return False, None, wmap, {}
    except Exception:
        # Fallback for single-file HF repos.
        from huggingface_hub import hf_hub_download

        repo_id, rev = lgc.parse_repo_spec(source)
        path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            revision=rev,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}
        return False, None, wmap, {"model.safetensors": Path(path)}


def _fetch_shard(
    source: str,
    shard_name: str,
    is_local: bool,
    local_dir: Path | None,
    resolved_cache: dict[str, Path],
    hf_cache_dir: str | None,
    local_files_only: bool,
) -> Path:
    """Resolve or download one safetensors shard file."""
    if shard_name in resolved_cache and resolved_cache[shard_name].exists():
        return resolved_cache[shard_name]
    if is_local:
        assert local_dir is not None
        p = local_dir / shard_name
        resolved_cache[shard_name] = p
        return p

    from huggingface_hub import hf_hub_download

    repo_id, rev = lgc.parse_repo_spec(source)
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=shard_name,
            revision=rev,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
    )
    resolved_cache[shard_name] = downloaded
    return downloaded


def _evict_hf_shard(shard_path: Path, hf_cache_dir: Path, protected_real_paths: set[Path]) -> None:
    """Safely unlink an HF cached shard symlink and blob once refcount reaches 0."""
    try:
        cache_root = hf_cache_dir.resolve()
        real = shard_path.resolve()
        if not real.is_relative_to(cache_root):
            return
        if shard_path.is_symlink() or shard_path.is_file():
            shard_path.unlink(missing_ok=True)
        if real != shard_path and real.is_file() and real not in protected_real_paths:
            real.unlink(missing_ok=True)
    except OSError:
        pass


def _is_eligible_key(
    key: str,
    include_visual: bool,
    include_embeddings: bool,
    include_re: re.Pattern[str] | None,
    exclude_re: re.Pattern[str] | None,
) -> bool:
    """Pre-filter weight keys cleanly before opening shards."""
    if not key.endswith(".weight"):
        return False
    if key.endswith(SKIP_SUFFIXES):
        return False
    if not include_visual and (".visual." in f".{key}" or "vision_tower" in key or "vision_model" in key):
        return False
    if not include_embeddings and (key.endswith("embed_tokens.weight") or key == "lm_head.weight" or key.endswith(".lm_head.weight")):
        return False
    if include_re and not include_re.search(key):
        return False
    if exclude_re and exclude_re.search(key):
        return False
    return True


def _pick_rank(
    s2: torch.Tensor,
    total_e: float,
    rank: int | None,
    energy: float | None,
    min_rank: int,
    max_rank: int | None,
    snap_mult: int,
) -> int:
    """Select rank k from squared singular values."""
    full_k = int(s2.shape[0])
    eff_max = min(full_k, max_rank) if max_rank is not None else full_k
    if rank is not None:
        k = min(max(1, rank), eff_max)
    elif energy is not None:
        cum = torch.cumsum(s2, dim=0) / max(total_e, 1e-30)
        idx = torch.nonzero(cum >= energy, as_tuple=False)
        k = int(idx[0].item()) + 1 if idx.numel() > 0 else eff_max
        k = max(min_rank, min(k, eff_max))
    else:
        k = min(32, eff_max)
    if snap_mult > 1 and k < full_k:
        k = min(eff_max, ((k + snap_mult - 1) // snap_mult) * snap_mult)
    return max(1, min(k, full_k))


def _svd_decompose(
    delta: torch.Tensor,
    mode: str,
    rank: int | None,
    energy: float | None,
    min_rank: int,
    max_rank: int | None,
    snap_mult: int,
    oversample: int,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    """Compute top-k (U, S, Vh, k, energy_retained) for float32 delta."""
    m, n = int(delta.shape[0]), int(delta.shape[1])
    total_e = float((delta * delta).sum().item())
    if total_e <= 0.0:
        raise ValueError("Zero delta matrix reached SVD")

    if mode == "exact":
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        s2 = S * S
        k = _pick_rank(s2, total_e, rank, energy, min_rank, max_rank, snap_mult)
        kept = float(s2[:k].sum().item())
        return U[:, :k], S[:k], Vh[:k, :], k, kept / total_e

    if mode == "gram":
        if m <= n:
            G = delta @ delta.T
            evals, evecs = torch.linalg.eigh(G)
            evals = evals.flip(0).clamp_min(0.0)
            evecs = evecs.flip(1)
            k = _pick_rank(evals, total_e, rank, energy, min_rank, max_rank, snap_mult)
            Uk = evecs[:, :k]
            proj = Uk.T @ delta
            Up, S, Vh = torch.linalg.svd(proj, full_matrices=False)
            U = Uk @ Up
        else:
            G = delta.T @ delta
            evals, evecs = torch.linalg.eigh(G)
            evals = evals.flip(0).clamp_min(0.0)
            evecs = evecs.flip(1)
            k = _pick_rank(evals, total_e, rank, energy, min_rank, max_rank, snap_mult)
            Vk = evecs[:, :k]
            proj = delta @ Vk
            U, S, Wh = torch.linalg.svd(proj, full_matrices=False)
            Vh = Wh @ Vk.T
        s2 = S * S
        kept = float(s2[:k].sum().item())
        return U[:, :k], S[:k], Vh[:k, :], k, kept / total_e

    if mode == "lowrank":
        if rank is None:
            raise ValueError("--svd-mode lowrank requires explicit --rank (cannot observe full spectrum for --energy).")
        target_k = min(rank, min(m, n))
        q = min(min(m, n), target_k + oversample)
        U, S, V = torch.svd_lowrank(delta, q=q, niter=niter)
        s2 = S * S
        kept = float(s2[:target_k].sum().item())
        return U[:, :target_k], S[:target_k], V[:, :target_k].T, target_k, kept / total_e

    raise ValueError(f"Unknown svd mode: {mode}")


def _estimate_svd_vram_mb(m: int, n: int, mode: str) -> float:
    """Conservative VRAM estimate in MiB so huge tensors auto-fallback to CPU cleanly."""
    mn_mb = (m * n * 4) / (1024.0 * 1024.0)
    if mode == "exact":
        return mn_mb * 5.8 + 128.0
    if mode == "gram":
        d = min(m, n)
        gram_mb = (d * d * 4) / (1024.0 * 1024.0)
        return mn_mb * 2.0 + gram_mb * 3.5 + 128.0
    return mn_mb * 2.0 + 128.0


def _cast_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _peft_key_prefix(weight_key: str) -> str:
    """Normalize weight key into PEFT 'base_model.model.<stem>' format."""
    stem = weight_key[: -len(".weight")]
    return f"base_model.model.{stem}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract SVD LoRA from (finetune - base) safetensors checkpoints")
    ap.add_argument("--base", required=True, help="Base model local dir or HF repo (e.g. Qwen/Qwen3.8-27B)")
    ap.add_argument("--finetune", required=True, help="Fine-tuned model local dir or HF repo")
    ap.add_argument("--out", type=Path, required=True, help="Output PEFT adapter directory (stats + validation JSON written here too)")
    ap.add_argument("--gguf-out", type=Path, default=None, help="Optional GGUF adapter output path")
    ap.add_argument("--outtype", choices=list(lgc.SUPPORTED_OUTTYPES), default="q8_0", help="GGUF tensor quant type when --gguf-out is set (default: q8_0)")
    ap.add_argument("--svd-mode", choices=["gram", "exact", "lowrank"], default="gram", help="SVD solver: gram (fast default, exact energy), exact, or lowrank")
    ap.add_argument("--rank", type=int, default=None, help="Uniform truncated rank (default: 32 if --energy is not set)")
    ap.add_argument("--energy", type=float, default=None, help="Adaptive rank per tensor capturing at least E fraction of delta energy")
    ap.add_argument("--min-rank", type=int, default=8, help="Min rank under --energy (default: 8)")
    ap.add_argument("--max-rank", type=int, default=64, help="Max rank cap under --energy (default: 64)")
    ap.add_argument("--snap-rank-mult", type=int, default=1, help="Round adaptive ranks up to a multiple of M (e.g. 32 for Q8_0)")
    ap.add_argument("--norm-threshold", type=float, default=0.05, help="Skip tensors with ||W_ft - W_base||_F < threshold (default: 0.05)")
    ap.add_argument("--oversample", type=int, default=32, help="Oversampling columns for --svd-mode lowrank (default: 32)")
    ap.add_argument("--niter", type=int, default=6, help="Subspace iterations for --svd-mode lowrank (default: 6)")
    ap.add_argument("--peft-dtype", choices=["float32", "bfloat16", "float16"], default="float32", help="Saved safetensors dtype (default: float32)")
    ap.add_argument("--include-embeddings", action="store_true", help="Also scan embed_tokens and lm_head")
    ap.add_argument("--include-visual", action="store_true", help="Also scan visual/vision_tower weights")
    ap.add_argument("--include-regex", type=str, default=None, help="Only process tensor keys matching regex")
    ap.add_argument("--exclude-regex", type=str, default=None, help="Skip tensor keys matching regex")
    ap.add_argument("--scan-only", action="store_true", help="Compute norms and write extraction_stats.json without running SVD")
    ap.add_argument("--hf-cache-dir", type=str, default=None, help="HuggingFace cache directory (e.g. /run/media/nos/disk/hf-cache)")
    ap.add_argument("--local-files-only", action="store_true", help="Do not download from HuggingFace")
    ap.add_argument("--evict-shards", action="store_true", help="Delete HF-cached shard files as soon as all tensors in them finish (never deletes local input dirs)")
    ap.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda, cuda:0)")
    ap.add_argument("--name", type=str, default=None, help="Optional general.name for GGUF metadata")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t_total = time.time()

    if args.rank is not None and args.energy is not None:
        raise SystemExit("Pass either --rank or --energy, not both.")
    target_rank = 32 if (args.rank is None and args.energy is None) else args.rank
    if args.svd_mode == "lowrank" and target_rank is None:
        raise SystemExit("--svd-mode lowrank requires --rank (use --svd-mode gram or exact with --energy).")
    if args.evict_shards and not args.hf_cache_dir:
        raise SystemExit("--evict-shards requires an explicit --hf-cache-dir so shared default cache isn't pruned.")

    raw_base_cfg = lgc.load_hf_json_file(args.base, "config.json", hf_cache_dir=args.hf_cache_dir, local_files_only=args.local_files_only)
    raw_ft_cfg = lgc.load_hf_json_file(args.finetune, "config.json", hf_cache_dir=args.hf_cache_dir, local_files_only=args.local_files_only)
    base_cfg = lgc.normalize_hf_config(raw_base_cfg)
    ft_cfg = lgc.normalize_hf_config(raw_ft_cfg)

    gguf_arch = lgc.infer_gguf_arch(base_cfg)
    ft_arch = lgc.infer_gguf_arch(ft_cfg)
    if gguf_arch != ft_arch:
        raise SystemExit(f"Architecture mismatch: base={gguf_arch} vs finetune={ft_arch}")
    n_layers = int(base_cfg["num_hidden_layers"])

    base_local, base_dir, base_wmap, base_shards = _resolve_model_index(args.base, args.hf_cache_dir, args.local_files_only)
    ft_local, ft_dir, ft_wmap, ft_shards = _resolve_model_index(args.finetune, args.hf_cache_dir, args.local_files_only)

    common_keys = sorted(set(base_wmap.keys()) & set(ft_wmap.keys()))
    if not common_keys:
        raise SystemExit("Base and finetune checkpoints share zero tensor keys.")

    inc_re = re.compile(args.include_regex) if args.include_regex else None
    exc_re = re.compile(args.exclude_regex) if args.exclude_regex else None
    eligible = [
        k for k in common_keys
        if _is_eligible_key(k, args.include_visual, args.include_embeddings, inc_re, exc_re)
    ]
    if not eligible:
        raise SystemExit("No eligible 2D weight keys left after filtering.")

    # Order tasks by (ft_shard, base_shard) and ref-count shards for safe eviction.
    tasks = sorted([(k, ft_wmap[k], base_wmap[k]) for k in eligible], key=lambda x: (x[1], x[2], x[0]))
    ft_ref = Counter(x[1] for x in tasks)
    base_ref = Counter(x[2] for x in tasks)

    device = lgc.select_device(args.device, min_free_mb=1024)
    out_dtype = _cast_dtype(args.peft_dtype)
    print(f"Eligible keys: {len(tasks)} | arch={gguf_arch} | device={device} | svd_mode={args.svd_mode}")

    from safetensors import safe_open
    from safetensors.torch import save_file

    norms: dict[str, float] = {}
    extracted_stats: list[dict[str, Any]] = []
    validation_checked: list[dict[str, Any]] = []
    peft_tensors: dict[str, torch.Tensor] = {}
    gguf_pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, float] = {}
    target_modules: set[str] = set()
    ranks_used: list[int] = []

    # Group by (ft_shard, base_shard) to open each safetensors shard cleanly once per batch.
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for k, fs, bs in tasks:
        grouped[(fs, bs)].append(k)

    cache_path = Path(args.hf_cache_dir) if args.hf_cache_dir else None
    current_ft_name: str | None = None
    current_ft_handle: Any = None

    for (fs, bs), keys in grouped.items():
        if fs != current_ft_name:
            if current_ft_handle is not None:
                current_ft_handle.__exit__(None, None, None)
            ft_file = _fetch_shard(args.finetune, fs, ft_local, ft_dir, ft_shards, args.hf_cache_dir, args.local_files_only)
            current_ft_handle = safe_open(str(ft_file), framework="pt", device="cpu")
            current_ft_handle.__enter__()
            current_ft_name = fs

        bs_file = _fetch_shard(args.base, bs, base_local, base_dir, base_shards, args.hf_cache_dir, args.local_files_only)
        with safe_open(str(bs_file), framework="pt", device="cpu") as base_handle:
            for key in keys:
                w_ft = current_ft_handle.get_tensor(key)
                w_base = base_handle.get_tensor(key)
                if w_ft.ndim != 2 or w_ft.shape != w_base.shape:
                    continue

                m, n = int(w_ft.shape[0]), int(w_ft.shape[1])
                run_dev = device
                if run_dev.type == "cuda":
                    need_mb = _estimate_svd_vram_mb(m, n, args.svd_mode)
                    if lgc.gpu_free_mb(run_dev) < need_mb:
                        run_dev = torch.device("cpu")

                t0 = time.time()
                try:
                    delta = w_ft.to(device=run_dev, dtype=torch.float32) - w_base.to(device=run_dev, dtype=torch.float32)
                    del w_ft, w_base
                    frob = float(torch.linalg.norm(delta).item())
                    norms[key] = frob

                    if args.scan_only or frob < args.norm_threshold:
                        del delta
                        continue

                    try:
                        U, S, Vh, r_used, energy_ret = _svd_decompose(
                            delta,
                            mode=args.svd_mode,
                            rank=target_rank,
                            energy=args.energy,
                            min_rank=args.min_rank,
                            max_rank=args.max_rank,
                            snap_mult=args.snap_rank_mult,
                            oversample=args.oversample,
                            niter=args.niter,
                        )
                    except torch.cuda.OutOfMemoryError:
                        delta = delta.cpu()
                        torch.cuda.empty_cache()
                        U, S, Vh, r_used, energy_ret = _svd_decompose(
                            delta,
                            mode=args.svd_mode,
                            rank=target_rank,
                            energy=args.energy,
                            min_rank=args.min_rank,
                            max_rank=args.max_rank,
                            snap_mult=args.snap_rank_mult,
                            oversample=args.oversample,
                            niter=args.niter,
                        )

                    sqrt_s = torch.sqrt(S)
                    B = (U * sqrt_s.unsqueeze(0)).to(dtype=out_dtype)
                    A = (sqrt_s.unsqueeze(1) * Vh).to(dtype=out_dtype)
                    recon = B.to(torch.float32) @ A.to(torch.float32)
                    rel_err = float(torch.linalg.norm(delta - recon).item() / max(frob, 1e-30))
                    dt = time.time() - t0

                    A_cpu = A.cpu().contiguous()
                    B_cpu = B.cpu().contiguous()
                    del delta, U, S, Vh, sqrt_s, B, A, recon
                    if run_dev.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception:
                    if run_dev.type == "cuda":
                        torch.cuda.empty_cache()
                    raise

                prefix = _peft_key_prefix(key)
                peft_tensors[f"{prefix}.lora_A.weight"] = A_cpu
                peft_tensors[f"{prefix}.lora_B.weight"] = B_cpu

                stem = key[: -len(".weight")]
                target_modules.add(stem.rsplit(".", 1)[-1])
                ranks_used.append(r_used)
                rank_pattern[stem] = r_used

                gguf_base = lgc.hf_weight_to_gguf_name(key, gguf_arch, n_layers, include_visual=args.include_visual)
                if gguf_base is not None:
                    A_g, B_g, _ = lgc.maybe_reorder_qwen35_v_heads(gguf_base, A_cpu.to(torch.float32), B_cpu.to(torch.float32), base_cfg)
                    gguf_pairs.append((gguf_base, A_g, B_g))

                extracted_stats.append({
                    "key": key,
                    "shape": [m, n],
                    "norm": frob,
                    "rank_used": r_used,
                    "energy_retained": energy_ret,
                    "time_s": round(dt, 3),
                })
                validation_checked.append({
                    "key": key,
                    "rel_error": round(rel_err, 5),
                    "energy_retained": energy_ret,
                })
                print(
                    f"[{len(extracted_stats):3d}] {key}: norm={frob:.4f} r={r_used} "
                    f"energy={energy_ret:.3f} rel_err={rel_err:.4f} ({dt:.2f}s)",
                    flush=True,
                )

        # Decrement shard refcounts and evict HF blobs when no longer needed.
        ft_ref[fs] -= len(keys)
        base_ref[bs] -= len(keys)
        if args.evict_shards and cache_path is not None:
            active_real: set[Path] = set()
            for sh_name, cnt in base_ref.items():
                if cnt > 0 and sh_name in base_shards:
                    active_real.add(base_shards[sh_name].resolve())
            for sh_name, cnt in ft_ref.items():
                if cnt > 0 and sh_name in ft_shards:
                    active_real.add(ft_shards[sh_name].resolve())
            if not base_local and base_ref[bs] == 0 and bs in base_shards:
                _evict_hf_shard(base_shards[bs], cache_path, active_real)
            if not ft_local and ft_ref[fs] == 0 and fs in ft_shards:
                if current_ft_handle is not None and fs == current_ft_name:
                    current_ft_handle.__exit__(None, None, None)
                    current_ft_handle = None
                    current_ft_name = None
                _evict_hf_shard(ft_shards[fs], cache_path, active_real)

    if current_ft_handle is not None:
        current_ft_handle.__exit__(None, None, None)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "extraction_stats.json", "w", encoding="utf-8") as f:
        json.dump({"norms": norms, "extracted": extracted_stats}, f, indent=2)

    if args.scan_only:
        above = sum(1 for v in norms.values() if v >= args.norm_threshold)
        print(f"Scan complete: {len(norms)} tensors scanned, {above} >= {args.norm_threshold} norm_threshold.")
        return 0

    if not peft_tensors:
        raise SystemExit(f"No tensors exceeded --norm-threshold {args.norm_threshold}; nothing saved.")

    save_file(peft_tensors, str(args.out / "adapter_model.safetensors"))

    unique_ranks = sorted(set(ranks_used))
    max_r = max(unique_ranks)
    uniform = len(unique_ranks) == 1
    if uniform:
        rank_pattern_out: dict[str, int] = {}
        alpha_pattern_out: dict[str, float] = {}
        rank_policy = f"{args.svd_mode} SVD, uniform rank {max_r}"
    else:
        rank_pattern_out = rank_pattern
        alpha_pattern_out = {k: float(v) for k, v in rank_pattern.items()}
        rank_policy = f"{args.svd_mode} SVD, adaptive energy>={args.energy} (r in {unique_ranks[0]}..{max_r})"

    adapter_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": args.base,
        "base_model": args.base,
        "source_model": args.finetune,
        "r": max_r,
        "lora_alpha": max_r,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(target_modules),
        "rank_pattern": rank_pattern_out,
        "alpha_pattern": alpha_pattern_out,
        "norm_threshold": args.norm_threshold,
        "svd_mode": args.svd_mode,
        "rank_policy": rank_policy,
        "dtype": args.peft_dtype,
    }
    with open(args.out / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_cfg, f, indent=2)

    rel_errs = [x["rel_error"] for x in validation_checked]
    energies = [x["energy_retained"] for x in validation_checked]
    val_report = {
        "summary": {
            "count": len(validation_checked),
            "mean_rel_error": float(np.mean(rel_errs)),
            "max_rel_error": float(np.max(rel_errs)),
            "mean_energy_retained": float(np.mean(energies)),
            "min_energy_retained": float(np.min(energies)),
        },
        "checked": validation_checked,
    }
    with open(args.out / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    if args.gguf_out is not None:
        if not gguf_pairs:
            raise SystemExit("No GGUF-mappable tensor pairs found for --gguf-out.")
        g_info = lgc.write_lora_gguf(
            out_path=args.gguf_out,
            arch=gguf_arch,
            lora_alpha=float(max_r),
            pairs=gguf_pairs,
            outtype=args.outtype,
            general_name=args.name or args.out.name,
            extra_kv={"lora.svd.source_model": str(args.finetune), "lora.svd.base_model": str(args.base)},
        )
        print(
            f"GGUF out   : {g_info['out_path']} ({g_info['file_bytes']/1024**2:.2f} MiB, "
            f"{g_info['n_pairs']} pairs, qtypes={g_info['qtype_counts']})"
        )

    print(
        f"Done in {time.time() - t_total:.1f}s -> {args.out} ({len(extracted_stats)} pairs, "
        f"mean_energy={val_report['summary']['mean_energy_retained']:.3f}, "
        f"mean_rel_err={val_report['summary']['mean_rel_error']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
