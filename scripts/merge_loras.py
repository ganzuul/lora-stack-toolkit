#!/usr/bin/env python3
"""Merge two LoRA adapters (PEFT dirs or GGUFs) in FP32 and export an optimized GGUF adapter.

Supports Qwen3.5 grouped->tiled V-head reordering for PEFT inputs, optional pre-merge SVD
truncation, exact orthogonal sum (lossless), naive truncated SVD, rank-allocated merges
(maximin / norm_greedy), and hybrid snapped-rank merges ({rank, 2*rank}) that stay 100%
block-32 aligned for Q8_0 (0 F16 fallbacks).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lora_gguf_common as lgc  # noqa: E402
import optimize_lora as ol  # noqa: E402


def _factor_qr_svd(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
    pre_rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact thin QR-SVD of scale * (B @ A) -> (U, S, Vh) in float32."""
    A32 = A.to(torch.float32)
    B32 = B.to(torch.float32)
    Qa, Ra = torch.linalg.qr(A32.T, mode="reduced")
    Qb, Rb = torch.linalg.qr(B32, mode="reduced")
    core = (Rb @ Ra.T) * float(scale)
    U_m, S, Vh_m = torch.linalg.svd(core, full_matrices=False)
    U = Qb @ U_m
    Vh = Vh_m @ Qa.T
    if pre_rank is not None and pre_rank < S.shape[0]:
        U = U[:, :pre_rank]
        S = S[:pre_rank]
        Vh = Vh[:pre_rank, :]
    return U, S, Vh


def _factors_to_balanced_ab(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert SVD factors (U, S, Vh) to balanced (A, B) with B @ A = U @ diag(S) @ Vh."""
    sqrt_s = torch.sqrt(torch.clamp(S, min=0.0))
    A_out = (sqrt_s.unsqueeze(1) * Vh).contiguous()
    B_out = (U * sqrt_s.unsqueeze(0)).contiguous()
    return A_out, B_out


def _balance_sum(
    U1: torch.Tensor,
    S1: torch.Tensor,
    Vh1: torch.Tensor,
    U2: torch.Tensor,
    S2: torch.Tensor,
    Vh2: torch.Tensor,
    trunc_r: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact (or top-r truncated) QR-SVD of U1 diag(S1) Vh1 + U2 diag(S2) Vh2."""
    B_cat = torch.cat([U1 * S1.unsqueeze(0), U2 * S2.unsqueeze(0)], dim=1)
    A_cat = torch.cat([Vh1, Vh2], dim=0)
    U_c, S_c, Vh_c = _factor_qr_svd(A_cat, B_cat, scale=1.0, pre_rank=trunc_r)
    return _factors_to_balanced_ab(U_c, S_c, Vh_c)


def _load_adapter_factors(
    path: Path,
    user_scale: float,
    pre_rank: int | None,
    base_override: str | None,
    hf_cache_dir: str | None,
    local_files_only: bool,
    include_visual: bool,
    device: torch.device,
) -> tuple[str, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Load PEFT dir or GGUF adapter, apply V-head reorder if PEFT, and extract QR-SVD factors."""
    if path.is_dir():
        arch, alpha, raw_pairs, cfg, meta = ol._read_peft_dir(
            path, base_override, hf_cache_dir, local_files_only, include_visual
        )
        do_reorder = True
    elif path.is_file():
        arch, alpha, raw_pairs, cfg, meta = ol._read_gguf_lora(
            path, base_override, hf_cache_dir, local_files_only
        )
        do_reorder = False
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")

    factors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    reordered = 0
    for k, A_in, B_in in raw_pairs:
        eff_scale = float(meta["per_tensor_scale"][k]) * float(user_scale)
        A_w = A_in.to(device)
        B_w = B_in.to(device)
        if do_reorder:
            A_w, B_w, flag = lgc.maybe_reorder_qwen35_v_heads(k, A_w, B_w, cfg)
            if flag:
                reordered += 1
        U, S, Vh = _factor_qr_svd(A_w, B_w, eff_scale, pre_rank=pre_rank)
        factors[k] = (U, S, Vh)

    info = {
        "path": str(path),
        "kind": meta["input_kind"],
        "alpha": alpha,
        "general_name": meta.get("general_name") or path.stem,
        "alora_tokens": meta.get("alora_tokens"),
        "n_pairs": len(factors),
        "vhead_reordered": reordered,
    }
    return arch, factors, info


def _gguf_tensor_bytes_est(n_rows: int, n_cols_ne0: int, outtype: str) -> int:
    """Estimate serialized GGUF tensor bytes including block-32 F16 fallback behavior."""
    ot = outtype.lower()
    block_bytes = {
        "q8_0": (32, 34),
        "q5_1": (32, 24),
        "q5_0": (32, 22),
        "q4_1": (32, 20),
        "q4_0": (32, 18),
        "mxfp4": (32, 17),
        "f16": (1, 2),
        "bf16": (1, 2),
        "f32": (1, 4),
    }
    blk, nbytes = block_bytes.get(ot, (32, 34))
    if blk > 1 and (n_cols_ne0 % blk) == 0:
        return n_rows * (n_cols_ne0 // blk) * nbytes
    if blk == 1:
        return n_rows * n_cols_ne0 * nbytes
    # Block-misaligned fallback to F16
    return n_rows * n_cols_ne0 * 2


def _pair_gguf_bytes_est(in_dim: int, out_dim: int, rank: int, outtype: str) -> int:
    return _gguf_tensor_bytes_est(rank, in_dim, outtype) + _gguf_tensor_bytes_est(out_dim, rank, outtype)


def _choose_rank_split(
    Sa: torch.Tensor,
    Sb: torch.Tensor,
    target_rank: int,
    mode: str,
) -> int:
    """Choose k_a in [max(1, target_rank-rb), min(ra, target_rank-1)] for maximin or norm_greedy."""
    ra = int(Sa.shape[0])
    rb = int(Sb.shape[0])
    lo = max(1, target_rank - rb)
    hi = min(ra, target_rank - 1)
    if lo > hi:
        return min(ra, target_rank)

    ea_all = (Sa * Sa).cpu().numpy()
    eb_all = (Sb * Sb).cpu().numpy()
    ea = max(float(ea_all.sum()), 1e-12)
    eb = max(float(eb_all.sum()), 1e-12)
    cum_a = np.cumsum(ea_all) / ea
    cum_b = np.cumsum(eb_all) / eb

    best_score = -1.0
    best_ka = lo
    for ka in range(lo, hi + 1):
        kb = target_rank - ka
        fa = float(cum_a[ka - 1])
        fb = float(cum_b[kb - 1])
        score = min(fa, fb) if mode == "maximin" else (fa + fb)
        if score > best_score:
            best_score = score
            best_ka = ka
    return best_ka


def _low_rank_inner_product(
    A1: torch.Tensor,
    B1: torch.Tensor,
    scale1: float,
    A2: torch.Tensor,
    B2: torch.Tensor,
    scale2: float,
) -> float:
    """Compute exact Frobenius inner product <(s1 B1 A1), (s2 B2 A2)> in O(r^2 (m+n))."""
    b1 = B1.to(torch.float32) * float(scale1)
    a1 = A1.to(torch.float32)
    b2 = B2.to(torch.float32) * float(scale2)
    a2 = A2.to(torch.float32)
    g_b12 = b1.T @ b2
    g_a21 = a2 @ a1.T
    return float(torch.sum(g_b12 * g_a21.T).item())


def _low_rank_norm_sq(A: torch.Tensor, B: torch.Tensor, scale: float) -> float:
    b = B.to(torch.float32) * float(scale)
    a = A.to(torch.float32)
    return float(torch.sum((b.T @ b) * (a @ a.T)).item())


def _verify_merged_gguf(
    out_path: Path,
    merged_refs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    factors_a: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    factors_b: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    shared_keys: list[str],
) -> dict[str, Any]:
    """Read back serialized GGUF from disk and compute quantization + per-adapter fidelity metrics."""
    gguf_mod = lgc.ensure_gguf()
    reader = gguf_mod.GGUFReader(str(out_path), mode="r")
    alpha = float(reader.fields["adapter.lora.alpha"].contents())
    gmap = {t.name: t for t in reader.tensors}

    quant_rel_errs: list[float] = []
    quant_cos: list[float] = []
    exact_cos: list[float] = []
    exact_rel_errs: list[float] = []
    exact_tot_norm_sq = 0.0
    exact_diff_sq_sum = 0.0

    pi_ret_list: list[float] = []
    qw_ret_list: list[float] = []
    pi_num_tot = 0.0
    pi_den_tot = 0.0
    qw_num_tot = 0.0
    qw_den_tot = 0.0

    shared_set = set(shared_keys)
    for k, (A_ref_m, B_ref_m) in merged_refs.items():
        ta = gmap[f"{k}.lora_a"]
        tb = gmap[f"{k}.lora_b"]
        arr_a = lgc.dequantize_gguf_array(gguf_mod, np.array(ta.data, copy=False), ta.tensor_type)
        arr_b = lgc.dequantize_gguf_array(gguf_mod, np.array(tb.data, copy=False), tb.tensor_type)
        A_q, B_q = lgc.gguf_arrays_to_ab(k, arr_a, arr_b)
        rank_q = int(A_q.shape[0])
        scale_q = (alpha / float(rank_q)) if alpha != 0.0 else 1.0

        # 1. Quantization error relative to unquantized merged reference
        rel_m, cos_m = lgc.low_rank_pair_metrics(A_ref_m, B_ref_m, 1.0, A_q, B_q, scale_q)
        quant_rel_errs.append(rel_m)
        quant_cos.append(cos_m)

        # 2. Exact unmerged target (ΔW_a + ΔW_b, or single adapter on disjoint keys)
        if k in shared_set:
            Ua, Sa, Vha = factors_a[k]
            Ub, Sb, Vhb = factors_b[k]
            Aa, Ba = _factors_to_balanced_ab(Ua.cpu(), Sa.cpu(), Vha.cpu())
            Ab, Bb = _factors_to_balanced_ab(Ub.cpu(), Sb.cpu(), Vhb.cpu())
            A_exact = torch.cat([Aa, Ab], dim=0)
            B_exact = torch.cat([Ba, Bb], dim=1)

            na_sq = float((Sa * Sa).sum().item())
            nb_sq = float((Sb * Sb).sum().item())
            ip_a = _low_rank_inner_product(A_q, B_q, scale_q, Aa, Ba, 1.0)
            ip_b = _low_rank_inner_product(A_q, B_q, scale_q, Ab, Bb, 1.0)
            if na_sq > 1e-12:
                pi_ret_list.append(ip_a / na_sq)
                pi_num_tot += ip_a
                pi_den_tot += na_sq
            if nb_sq > 1e-12:
                qw_ret_list.append(ip_b / nb_sq)
                qw_num_tot += ip_b
                qw_den_tot += nb_sq
        elif k in factors_a:
            Ua, Sa, Vha = factors_a[k]
            A_exact, B_exact = _factors_to_balanced_ab(Ua.cpu(), Sa.cpu(), Vha.cpu())
        else:
            Ub, Sb, Vhb = factors_b[k]
            A_exact, B_exact = _factors_to_balanced_ab(Ub.cpu(), Sb.cpu(), Vhb.cpu())

        rel_e, cos_e = lgc.low_rank_pair_metrics(A_exact, B_exact, 1.0, A_q, B_q, scale_q)
        exact_rel_errs.append(rel_e)
        exact_cos.append(cos_e)
        n_exact_sq = _low_rank_norm_sq(A_exact, B_exact, 1.0)
        exact_tot_norm_sq += n_exact_sq
        exact_diff_sq_sum += (rel_e * rel_e) * n_exact_sq

    return {
        "quant_mean_rel_err": float(np.mean(quant_rel_errs)) if quant_rel_errs else 0.0,
        "quant_max_rel_err": float(np.max(quant_rel_errs)) if quant_rel_errs else 0.0,
        "quant_mean_cos": float(np.mean(quant_cos)) if quant_cos else 1.0,
        "quant_min_cos": float(np.min(quant_cos)) if quant_cos else 1.0,
        "exact_mean_rel_err": float(np.mean(exact_rel_errs)) if exact_rel_errs else 0.0,
        "exact_mean_cos": float(np.mean(exact_cos)) if exact_cos else 1.0,
        "exact_min_cos": float(np.min(exact_cos)) if exact_cos else 1.0,
        "exact_global_energy_retained": 1.0 - (exact_diff_sq_sum / max(exact_tot_norm_sq, 1e-12)),
        "shared_a_global_retention": pi_num_tot / max(pi_den_tot, 1e-12),
        "shared_a_mean_retention": float(np.mean(pi_ret_list)) if pi_ret_list else 1.0,
        "shared_a_min_retention": float(np.min(pi_ret_list)) if pi_ret_list else 1.0,
        "shared_a_p10_retention": float(np.percentile(pi_ret_list, 10)) if pi_ret_list else 1.0,
        "shared_b_global_retention": qw_num_tot / max(qw_den_tot, 1e-12),
        "shared_b_mean_retention": float(np.mean(qw_ret_list)) if qw_ret_list else 1.0,
        "shared_b_min_retention": float(np.min(qw_ret_list)) if qw_ret_list else 1.0,
        "shared_b_p10_retention": float(np.percentile(qw_ret_list, 10)) if qw_ret_list else 1.0,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True, help="First adapter (PEFT dir or .gguf)")
    ap.add_argument("--b", type=Path, required=True, help="Second adapter (PEFT dir or .gguf)")
    ap.add_argument("--scale-a", type=float, default=1.0, help="Scale multiplier for adapter A (default: 1.0)")
    ap.add_argument("--scale-b", type=float, default=1.0, help="Scale multiplier for adapter B (default: 1.0)")
    ap.add_argument("--pre-rank-a", type=int, default=None, help="Optional SVD pre-truncation rank for adapter A")
    ap.add_argument("--pre-rank-b", type=int, default=None, help="Optional SVD pre-truncation rank for adapter B")
    ap.add_argument(
        "--method",
        required=True,
        choices=("naive", "maximin", "norm_greedy", "hybrid", "lossless"),
        help="Shared-tensor merge method",
    )
    ap.add_argument("--rank", type=int, default=32, help="Target rank for naive/maximin/norm_greedy or base rank for hybrid")
    ap.add_argument("--upgrade-top", type=int, default=64, help="Number of shared tensors upgraded to lossless in hybrid mode")
    ap.add_argument("--outtype", choices=list(lgc.SUPPORTED_OUTTYPES), default="q8_0", help="Output GGUF tensor type (default: q8_0)")
    ap.add_argument("-o", "--outfile", type=Path, required=True, help="Output .gguf path")
    ap.add_argument("--report-json", type=Path, default=None, help="Optional path to write verification/stats JSON")
    ap.add_argument("--name", type=str, default=None, help="Override GGUF general.name")
    ap.add_argument("--base", type=str, default=None, help="Base HF model override for PEFT loading")
    ap.add_argument("--hf-cache-dir", type=str, default=None, help="HF cache directory")
    ap.add_argument("--local-files-only", action="store_true", help="Do not query HuggingFace Hub")
    ap.add_argument("--include-visual", action="store_true", help="Include visual tower tensors if present")
    ap.add_argument("--device", type=str, default="cpu", help="Compute device (default: cpu; or auto/cuda/cuda:0)")
    ap.add_argument("--verify", dest="verify", action="store_true", default=True, help="Verify written GGUF readback (default: on)")
    ap.add_argument("--no-verify", dest="verify", action="store_false", help="Skip readback verification")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    device = lgc.select_device(args.device, min_free_mb=256)

    arch_a, factors_a, info_a = _load_adapter_factors(
        args.a, args.scale_a, args.pre_rank_a, args.base, args.hf_cache_dir, args.local_files_only, args.include_visual, device
    )
    arch_b, factors_b, info_b = _load_adapter_factors(
        args.b, args.scale_b, args.pre_rank_b, args.base, args.hf_cache_dir, args.local_files_only, args.include_visual, device
    )
    if arch_a != arch_b:
        raise SystemExit(f"Architecture mismatch: {arch_a} vs {arch_b}")

    all_keys = sorted(set(factors_a) | set(factors_b))
    shared_keys = sorted(set(factors_a) & set(factors_b))
    a_only = sorted(set(factors_a) - set(factors_b))
    b_only = sorted(set(factors_b) - set(factors_a))

    # Precompute maximin splits and hybrid upgrades if needed
    split_ka: dict[str, int] = {}
    upgrade_set: set[str] = set()
    if args.method in ("maximin", "norm_greedy", "hybrid"):
        split_mode = "norm_greedy" if args.method == "norm_greedy" else "maximin"
        tot_ea = sum(max(float((factors_a[k][1] ** 2).sum().item()), 0.0) for k in shared_keys)
        tot_eb = sum(max(float((factors_b[k][1] ** 2).sum().item()), 0.0) for k in shared_keys)
        hybrid_candidates: list[tuple[float, str]] = []
        for k in shared_keys:
            Ua, Sa, Vha = factors_a[k]
            Ub, Sb, Vhb = factors_b[k]
            ka = _choose_rank_split(Sa, Sb, args.rank, split_mode)
            split_ka[k] = ka
            if args.method == "hybrid":
                kb = args.rank - ka
                ea = float((Sa * Sa).sum().item())
                eb = float((Sb * Sb).sum().item())
                loss_a = ea - float((Sa[:ka] ** 2).sum().item())
                loss_b = eb - float((Sb[:kb] ** 2).sum().item())
                in_dim = int(Vha.shape[1])
                out_dim = int(Ua.shape[0])
                full_r = int(Sa.shape[0] + Sb.shape[0])
                extra_bytes = (
                    _pair_gguf_bytes_est(in_dim, out_dim, full_r, args.outtype)
                    - _pair_gguf_bytes_est(in_dim, out_dim, args.rank, args.outtype)
                )
                score = ((loss_a / max(tot_ea, 1e-12)) + (loss_b / max(tot_eb, 1e-12))) / (
                    max(extra_bytes, 1) / 1048576.0
                )
                hybrid_candidates.append((score, k))
        if args.method == "hybrid" and args.upgrade_top > 0:
            hybrid_candidates.sort(reverse=True)
            upgrade_set = {k for _, k in hybrid_candidates[: args.upgrade_top]}

    out_pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    merged_refs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    ranks_out: list[int] = []
    shared_ranks: list[int] = []

    for k in all_keys:
        if k in factors_a and k in factors_b:
            Ua, Sa, Vha = factors_a[k]
            Ub, Sb, Vhb = factors_b[k]
            full_r = int(Sa.shape[0] + Sb.shape[0])
            if args.method == "lossless" or k in upgrade_set or full_r <= args.rank:
                A_out, B_out = _balance_sum(Ua, Sa, Vha, Ub, Sb, Vhb, trunc_r=None)
            elif args.method == "naive":
                A_out, B_out = _balance_sum(Ua, Sa, Vha, Ub, Sb, Vhb, trunc_r=args.rank)
            else:
                ka = split_ka[k]
                kb = min(int(Sb.shape[0]), args.rank - ka)
                A_out, B_out = _balance_sum(
                    Ua[:, :ka], Sa[:ka], Vha[:ka, :],
                    Ub[:, :kb], Sb[:kb], Vhb[:kb, :],
                    trunc_r=None,
                )
            shared_ranks.append(int(A_out.shape[0]))
        elif k in factors_a:
            Ua, Sa, Vha = factors_a[k]
            A_out, B_out = _factors_to_balanced_ab(Ua, Sa, Vha)
        else:
            Ub, Sb, Vhb = factors_b[k]
            A_out, B_out = _factors_to_balanced_ab(Ub, Sb, Vhb)

        A_cpu = A_out.cpu()
        B_cpu = B_out.cpu()
        out_pairs.append((k, A_cpu, B_cpu))
        merged_refs[k] = (A_cpu, B_cpu)
        ranks_out.append(int(A_cpu.shape[0]))

    out_alpha = float(max(ranks_out))
    gname = args.name or f"{info_a['general_name']}+{info_b['general_name']}-{args.method}"
    alora_tokens = info_a["alora_tokens"] if info_a["alora_tokens"] is not None else info_b["alora_tokens"]

    write_info = lgc.write_lora_gguf(
        out_path=args.outfile,
        arch=arch_a,
        lora_alpha=out_alpha,
        pairs=out_pairs,
        outtype=args.outtype,
        general_name=gname,
        alora_tokens=alora_tokens,
    )

    verify_info: dict[str, Any] = {}
    if args.verify:
        verify_info = _verify_merged_gguf(args.outfile, merged_refs, factors_a, factors_b, shared_keys)

    elapsed = time.time() - t0
    file_mib = write_info["file_bytes"] / (1024.0 * 1024.0)
    rank_hist = {r: ranks_out.count(r) for r in sorted(set(ranks_out))}
    report = {
        "method": args.method,
        "rank_arg": args.rank,
        "upgrade_top": args.upgrade_top if args.method == "hybrid" else 0,
        "outtype": args.outtype,
        "arch": arch_a,
        "lora_alpha": out_alpha,
        "input_a": info_a,
        "input_b": info_b,
        "n_total": len(all_keys),
        "n_shared": len(shared_keys),
        "n_a_only": len(a_only),
        "n_b_only": len(b_only),
        "rank_histogram": rank_hist,
        "upgraded_count": len(upgrade_set),
        "mean_k_a": float(np.mean(list(split_ka.values()))) if split_ka else None,
        "outfile": str(args.outfile),
        "file_bytes": write_info["file_bytes"],
        "file_mib": round(file_mib, 3),
        "qtype_counts": write_info["qtype_counts"],
        "fallback_count": write_info["fallback_count"],
        "fallback_sample": write_info["fallback_sample"],
        "verify": verify_info,
        "elapsed_sec": round(elapsed, 2),
        "device": str(device),
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Adapter A  : {info_a['path']} ({info_a['kind']}, {info_a['n_pairs']} pairs, scale={args.scale_a}, pre_rank={args.pre_rank_a})")
    print(f"Adapter B  : {info_b['path']} ({info_b['kind']}, {info_b['n_pairs']} pairs, scale={args.scale_b}, pre_rank={args.pre_rank_b})")
    print(f"Merge      : method={args.method} shared={len(shared_keys)} a_only={len(a_only)} b_only={len(b_only)} ranks={rank_hist}")
    print(f"Output     : {args.outfile} ({file_mib:.2f} MiB, alpha={out_alpha:g}, qtypes={write_info['qtype_counts']}, fallbacks={write_info['fallback_count']})")
    if verify_info:
        print(
            f"Quant vs M : mean_cos={verify_info['quant_mean_cos']:.6f} min_cos={verify_info['quant_min_cos']:.6f} "
            f"mean_rel_err={verify_info['quant_mean_rel_err']:.5f}"
        )
        print(
            f"Sum Fidelity: global_E={100*verify_info['exact_global_energy_retained']:.2f}% "
            f"mean_cos={verify_info['exact_mean_cos']:.5f} min_cos={verify_info['exact_min_cos']:.5f}"
        )
        print(
            f"Retained A : global={100*verify_info['shared_a_global_retention']:.2f}% "
            f"(mean={100*verify_info['shared_a_mean_retention']:.2f}%, p10={100*verify_info['shared_a_p10_retention']:.1f}%, min={100*verify_info['shared_a_min_retention']:.1f}%)"
        )
        print(
            f"Retained B : global={100*verify_info['shared_b_global_retention']:.2f}% "
            f"(mean={100*verify_info['shared_b_mean_retention']:.2f}%, p10={100*verify_info['shared_b_p10_retention']:.1f}%, min={100*verify_info['shared_b_min_retention']:.1f}%)"
        )
    print(f"Elapsed    : {elapsed:.2f}s (device={device})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
