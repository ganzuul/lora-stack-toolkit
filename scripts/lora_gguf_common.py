#!/usr/bin/env python3
"""Shared LoRA optimization, SVD compression, Qwen3.5 V-head layout, and GGUF helpers."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_GGUF_PY = Path("/run/media/nos/games/inference_engines/Bonsai-demo/llama.cpp/gguf-py")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


def ensure_gguf():
    """Import Bonsai-demo's gguf-py (supports Q8_0, BF16, PTQ1_0/Qwen3.5 mappings)."""
    gguf_dir = Path(os.environ.get("BONSAI_GGUF_PY", str(DEFAULT_GGUF_PY)))
    if gguf_dir.is_dir() and str(gguf_dir) not in sys.path:
        sys.path.insert(0, str(gguf_dir))
    import gguf  # noqa: E402
    return gguf


def select_device(requested: str = "auto", min_free_mb: int = 512) -> torch.device:
    """Pick a compute device compatible with the active PyTorch CUDA build."""
    req = requested.strip().lower()
    if req == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if req != "auto":
            raise RuntimeError(f"Requested CUDA device '{requested}', but CUDA is unavailable.")
        return torch.device("cpu")

    supported_arches = set(torch.cuda.get_arch_list())
    candidates = [int(req.split(":")[1])] if req.startswith("cuda:") else list(range(torch.cuda.device_count()))

    for idx in candidates:
        major, minor = torch.cuda.get_device_capability(idx)
        sm_tag = f"sm_{major}{minor}"
        if supported_arches and sm_tag not in supported_arches and not any(
            a.startswith(f"sm_{major}") and minor >= int(a[4:]) for a in supported_arches if a.startswith("sm_")
        ):
            continue
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
        except Exception:
            free_bytes = 0
        if free_bytes >= min_free_mb * 1024 * 1024:
            return torch.device(f"cuda:{idx}")

    if req != "auto":
        raise RuntimeError(
            f"Requested device '{requested}' is incompatible or lacks {min_free_mb} MiB free VRAM."
        )
    return torch.device("cpu")


def gpu_free_mb(device: torch.device) -> float:
    """Return free VRAM in MiB for a CUDA device (inf for CPU)."""
    if device.type != "cuda":
        return float("inf")
    free_bytes, _ = torch.cuda.mem_get_info(device.index or 0)
    return free_bytes / (1024.0 * 1024.0)


def parse_repo_spec(spec: str) -> tuple[str, str | None]:
    """Split 'org/repo@rev' into (repo_id, revision)."""
    if "@" in spec and not Path(spec).exists():
        repo_id, rev = spec.rsplit("@", 1)
        return repo_id, rev
    return spec, None


def load_hf_json_file(
    source: str | Path,
    filename: str,
    hf_cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load a JSON file from a local model directory or HF Hub repo."""
    p = Path(source)
    if p.is_dir():
        target = p / filename
        if not target.is_file():
            raise FileNotFoundError(f"Missing {filename} in {p}")
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)

    from huggingface_hub import hf_hub_download

    repo_id, rev = parse_repo_spec(str(source))
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=rev,
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
        local_files_only=local_files_only,
    )
    with open(downloaded, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_hf_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten multimodal wrappers (e.g. Qwen3_5ForConditionalGeneration text_config)."""
    cfg = dict(raw_cfg)
    if isinstance(cfg.get("text_config"), dict):
        merged = dict(cfg)
        merged.update(cfg["text_config"])
        return merged
    return cfg


def infer_gguf_arch(cfg: dict[str, Any]) -> str:
    """Map HF config architectures/model_type to canonical GGUF general.architecture."""
    arch_list = cfg.get("architectures") or []
    arch_str = arch_list[0] if arch_list else ""
    model_type = str(cfg.get("model_type", "")).lower()

    by_arch = {
        "Qwen3_5ForConditionalGeneration": "qwen35",
        "Qwen3_5ForCausalLM": "qwen35",
        "Qwen3_5MoeForConditionalGeneration": "qwen35moe",
        "Qwen3_5MoeForCausalLM": "qwen35moe",
        "Qwen3NextForCausalLM": "qwen3next",
        "Qwen3ForCausalLM": "qwen3",
        "Qwen3MoeForCausalLM": "qwen3moe",
        "Qwen2ForCausalLM": "qwen2",
        "Qwen2MoeForCausalLM": "qwen2moe",
        "LlamaForCausalLM": "llama",
        "MistralForCausalLM": "llama",
        "Gemma2ForCausalLM": "gemma2",
        "Gemma3ForCausalLM": "gemma3",
        "Gemma3ForConditionalGeneration": "gemma3",
    }
    if arch_str in by_arch:
        return by_arch[arch_str]

    by_type = {
        "qwen3_5": "qwen35",
        "qwen3_5_text": "qwen35",
        "qwen3_5_moe": "qwen35moe",
        "qwen3_5_moe_text": "qwen35moe",
        "qwen3_next": "qwen3next",
        "qwen3": "qwen3",
        "qwen3_moe": "qwen3moe",
        "qwen2": "qwen2",
        "llama": "llama",
        "mistral": "llama",
        "gemma2": "gemma2",
        "gemma3": "gemma3",
        "gemma3_text": "gemma3",
    }
    if model_type in by_type:
        return by_type[model_type]
    raise ValueError(f"Unsupported architecture '{arch_str}' / model_type '{model_type}'")


def resolve_gguf_arch_enum(gguf_mod: Any, gguf_arch: str) -> Any:
    """Resolve GGUF architecture string to gguf.MODEL_ARCH enum."""
    for enum_val, name in gguf_mod.MODEL_ARCH_NAMES.items():
        if name == gguf_arch:
            return enum_val
    raise ValueError(f"Unknown GGUF architecture '{gguf_arch}'")


# Canonical Qwen/Llama suffix map used cleanly when mapping PEFT keys -> GGUF names.
_SUFFIX_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def hf_weight_to_gguf_name(
    hf_weight_name: str,
    gguf_arch: str,
    n_layers: int,
    include_visual: bool = False,
) -> str | None:
    """Map a HuggingFace base weight key to its canonical GGUF tensor name."""
    gguf_mod = ensure_gguf()
    name = hf_weight_name
    while name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    if not include_visual and (".visual." in f".{name}" or "vision_tower" in name or "vision_model" in name):
        return None

    # Normalize multimodal wrapper prefix so tensor_mapping finds standard names.
    norm = name
    for prefix in ("model.language_model.", "language_model.model.", "language_model."):
        if norm.startswith(prefix):
            norm = "model." + norm[len(prefix) :]
            break
    if norm.endswith(".weight"):
        stem = norm[: -len(".weight")]
    else:
        stem = norm

    if stem in ("model.embed_tokens", "embed_tokens"):
        return "token_embd.weight"
    if stem in ("lm_head", "model.lm_head"):
        return "output.weight"

    m = LAYER_RE.match(stem)
    if m:
        bid = int(m.group(1))
        sub = m.group(2)
        if sub in _SUFFIX_MAP:
            return f"blk.{bid}.{_SUFFIX_MAP[sub]}.weight"

    arch_enum = resolve_gguf_arch_enum(gguf_mod, gguf_arch)
    tmap = gguf_mod.get_tensor_name_map(arch_enum, n_layers)
    mapped = tmap.get_name(stem)
    if mapped is not None:
        return f"{mapped}.weight"
    return None


def gguf_base_to_hf_stem(gguf_base: str) -> str:
    """Best-effort reverse map from GGUF base tensor name to Qwen3.8/PEFT base stem."""
    if gguf_base == "token_embd.weight":
        return "model.language_model.embed_tokens"
    if gguf_base == "output.weight":
        return "lm_head"
    rev = {v: k for k, v in _SUFFIX_MAP.items()}
    if gguf_base.startswith("blk.") and gguf_base.endswith(".weight"):
        parts = gguf_base[: -len(".weight")].split(".", 2)
        if len(parts) == 3 and parts[2] in rev:
            return f"model.language_model.layers.{parts[1]}.{rev[parts[2]]}"
    raise ValueError(f"Cannot reverse-map GGUF tensor name '{gguf_base}' to HF PEFT key")


def reorder_v_heads(tensor: torch.Tensor, dim: int, nk: int, rep: int, hd: int) -> torch.Tensor:
    """Reorder Gated DeltaNet V heads from HF grouped [nk, rep, hd] to llama.cpp tiled [rep, nk, hd]."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    expected = nk * rep * hd
    if shape[dim] != expected:
        raise ValueError(f"V-head reorder dim {dim} has size {shape[dim]}, expected {expected}")
    new_shape = shape[:dim] + [nk, rep, hd] + shape[dim + 1 :]
    reshaped = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return reshaped.permute(*perm).contiguous().reshape(*shape)


def maybe_reorder_qwen35_v_heads(
    gguf_base: str,
    A: torch.Tensor,
    B: torch.Tensor,
    cfg: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Apply Qwen3.5 grouped->tiled V-head permutation cleanly for linear_attn LoRA pairs."""
    if not cfg:
        return A, B, False
    nk = int(cfg.get("linear_num_key_heads", 0) or 0)
    nv = int(cfg.get("linear_num_value_heads", 0) or 0)
    if nk <= 0 or nv <= 0 or nk == nv:
        return A, B, False

    dk = int(cfg["linear_key_head_dim"])
    dv = int(cfg["linear_value_head_dim"])
    rep = nv // nk

    if gguf_base.endswith(".attn_qkv.weight"):
        q_dim = dk * nk
        k_dim = dk * nk
        if B.shape[0] != q_dim + k_dim + nv * dv:
            raise ValueError(f"Unexpected attn_qkv output dim {B.shape[0]} for nk={nk}, nv={nv}")
        v_reordered = reorder_v_heads(B[q_dim + k_dim :], 0, nk, rep, dv)
        return A, torch.cat([B[: q_dim + k_dim], v_reordered], dim=0), True

    if gguf_base.endswith(".attn_gate.weight"):
        return A, reorder_v_heads(B, 0, nk, rep, dv), True

    if gguf_base.endswith((".ssm_alpha.weight", ".ssm_beta.weight")):
        return A, reorder_v_heads(B, 0, nk, rep, 1), True

    if gguf_base.endswith(".ssm_out.weight"):
        return reorder_v_heads(A, 1, nk, rep, dv), B, True

    return A, B, False


def compress_ab_qr_svd(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_in: float = 1.0,
    target_rank: int | None = None,
    energy: float | None = None,
    min_rank: int = 1,
    max_rank: int | None = None,
    snap_mult: int = 1,
    balance: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Exact QR + r×r SVD of delta = scale_in * (B @ A), returning unscaled delta = B_new @ A_new."""
    orig_rank = int(A.shape[0])
    A32 = A.to(torch.float32)
    B32 = B.to(torch.float32)

    if scale_in != 1.0:
        sign = 1.0 if scale_in >= 0 else -1.0
        mag = abs(float(scale_in)) ** 0.5
        B32 = B32 * (sign * mag)
        A32 = A32 * mag

    # Fast exact decomposition via thin QR on m×r and n×r factors + tiny r×r SVD.
    Qb, Rb = torch.linalg.qr(B32, mode="reduced")
    Qa, Ra = torch.linalg.qr(A32.T, mode="reduced")
    core = Rb @ Ra.T
    U_m, S, Vh_m = torch.linalg.svd(core, full_matrices=False)

    s2 = S * S
    total_e = float(s2.sum().item())
    orig_norm = total_e ** 0.5

    if total_e <= 0.0:
        k = min_rank
        A_new = torch.zeros((k, A.shape[1]), dtype=torch.float32, device=A.device)
        B_new = torch.zeros((B.shape[0], k), dtype=torch.float32, device=B.device)
        return A_new, B_new, {
            "orig_rank": orig_rank,
            "new_rank": k,
            "orig_norm": 0.0,
            "energy_retained": 1.0,
            "rel_error": 0.0,
        }

    eff_max = min(orig_rank, max_rank) if max_rank is not None else orig_rank
    if target_rank is not None:
        k = min(max(1, target_rank), eff_max)
    elif energy is not None:
        cum = torch.cumsum(s2, dim=0) / total_e
        idx = torch.nonzero(cum >= energy, as_tuple=False)
        k = int(idx[0].item()) + 1 if idx.numel() > 0 else orig_rank
        k = max(min_rank, min(k, eff_max))
    else:
        k = eff_max

    if snap_mult > 1 and k < orig_rank:
        k = min(eff_max, ((k + snap_mult - 1) // snap_mult) * snap_mult)
    k = max(1, min(k, S.shape[0]))

    kept_e = float(s2[:k].sum().item())
    energy_retained = kept_e / total_e
    rel_error = max(0.0, 1.0 - energy_retained) ** 0.5

    if not balance and k == orig_rank and snap_mult <= 1 and max_rank is None and target_rank is None:
        return A32, B32, {
            "orig_rank": orig_rank,
            "new_rank": k,
            "orig_norm": orig_norm,
            "energy_retained": energy_retained,
            "rel_error": rel_error,
        }

    sqrt_s = torch.sqrt(S[:k])
    U_k = Qb @ U_m[:, :k]
    Vh_k = Vh_m[:k, :] @ Qa.T
    B_new = U_k * sqrt_s.unsqueeze(0)
    A_new = sqrt_s.unsqueeze(1) * Vh_k

    return A_new, B_new, {
        "orig_rank": orig_rank,
        "new_rank": k,
        "orig_norm": orig_norm,
        "energy_retained": energy_retained,
        "rel_error": rel_error,
    }


def low_rank_pair_metrics(
    A1: torch.Tensor,
    B1: torch.Tensor,
    scale1: float,
    A2: torch.Tensor,
    B2: torch.Tensor,
    scale2: float,
) -> tuple[float, float]:
    """Compute exact (rel_frob_error, cosine_sim) between scale1*(B1@A1) and scale2*(B2@A2) in O(r^2*(m+n))."""
    b1 = B1.to(torch.float32) * float(scale1)
    a1 = A1.to(torch.float32)
    b2 = B2.to(torch.float32) * float(scale2)
    a2 = A2.to(torch.float32)

    g_b11 = b1.T @ b1
    g_a11 = a1 @ a1.T
    n1_sq = float(torch.sum(g_b11 * g_a11).item())

    g_b22 = b2.T @ b2
    g_a22 = a2 @ a2.T
    n2_sq = float(torch.sum(g_b22 * g_a22).item())

    g_b12 = b1.T @ b2
    g_a21 = a2 @ a1.T
    dot12 = float(torch.sum(g_b12 * g_a21.T).item())

    if n1_sq <= 0.0 and n2_sq <= 0.0:
        return 0.0, 1.0
    if n1_sq <= 0.0 or n2_sq <= 0.0:
        return 1.0, 0.0

    diff_sq = max(0.0, n1_sq + n2_sq - 2.0 * dot12)
    rel_err = (diff_sq / n1_sq) ** 0.5
    cos = max(-1.0, min(1.0, dot12 / ((n1_sq * n2_sq) ** 0.5)))
    return rel_err, cos


def ab_to_gguf_arrays(gguf_base: str, A: torch.Tensor, B: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert logical A (r, in_dim), B (out_dim, r) to GGUF numpy layout (handling token_embd)."""
    a_np = A.detach().cpu().to(torch.float32).numpy()
    b_np = B.detach().cpu().to(torch.float32).numpy()
    if gguf_base == "token_embd.weight":
        # llama.cpp expects token_embd.lora_a as [r, vocab] in GGML => (vocab, r) in numpy.
        a_np = np.ascontiguousarray(a_np.T)
    return np.ascontiguousarray(a_np), np.ascontiguousarray(b_np)


def gguf_arrays_to_ab(gguf_base: str, arr_a: np.ndarray, arr_b: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert GGUF numpy arrays back to logical A (r, in_dim), B (out_dim, r) float32 tensors."""
    a_t = torch.from_numpy(np.array(arr_a, dtype=np.float32, copy=True))
    b_t = torch.from_numpy(np.array(arr_b, dtype=np.float32, copy=True))
    if gguf_base == "token_embd.weight":
        a_t = a_t.T.contiguous()
    return a_t, b_t


SUPPORTED_OUTTYPES = ("q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "mxfp4", "f16", "bf16", "f32")


def outtype_to_qtype(gguf_mod: Any, outtype: str) -> tuple[Any, Any]:
    """Map CLI outtype string to (GGMLQuantizationType, LlamaFileType)."""
    mapping = {
        "f32": (gguf_mod.GGMLQuantizationType.F32, gguf_mod.LlamaFileType.ALL_F32),
        "f16": (gguf_mod.GGMLQuantizationType.F16, gguf_mod.LlamaFileType.MOSTLY_F16),
        "bf16": (gguf_mod.GGMLQuantizationType.BF16, gguf_mod.LlamaFileType.MOSTLY_BF16),
        "q8_0": (gguf_mod.GGMLQuantizationType.Q8_0, gguf_mod.LlamaFileType.MOSTLY_Q8_0),
        "q5_1": (gguf_mod.GGMLQuantizationType.Q5_1, gguf_mod.LlamaFileType.MOSTLY_Q5_1),
        "q5_0": (gguf_mod.GGMLQuantizationType.Q5_0, gguf_mod.LlamaFileType.MOSTLY_Q5_0),
        "q4_1": (gguf_mod.GGMLQuantizationType.Q4_1, gguf_mod.LlamaFileType.MOSTLY_Q4_1),
        "q4_0": (gguf_mod.GGMLQuantizationType.Q4_0, gguf_mod.LlamaFileType.MOSTLY_Q4_0),
        "mxfp4": (gguf_mod.GGMLQuantizationType.MXFP4, gguf_mod.LlamaFileType.MOSTLY_MXFP4_MOE),
    }
    key = outtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported outtype '{outtype}', expected one of {list(mapping)}")
    return mapping[key]


def quantize_array_for_gguf(gguf_mod: Any, arr: np.ndarray, target_qtype: Any) -> tuple[np.ndarray, Any]:
    """Quantize a numpy float32 tensor for GGUF, falling back cleanly to F16 when shape isn't block-aligned."""
    try:
        qdata = gguf_mod.quants.quantize(arr, target_qtype)
        return qdata, target_qtype
    except gguf_mod.QuantError:
        f16_t = gguf_mod.GGMLQuantizationType.F16
        return gguf_mod.quants.quantize(arr, f16_t), f16_t


def dequantize_gguf_array(gguf_mod: Any, arr: np.ndarray, qtype: Any) -> np.ndarray:
    """Dequantize a GGUF tensor array to float32 numpy array."""
    if qtype == gguf_mod.GGMLQuantizationType.F32:
        return arr.astype(np.float32, copy=False)
    if qtype == gguf_mod.GGMLQuantizationType.F16:
        return arr.astype(np.float32)
    return gguf_mod.quants.dequantize(arr, qtype).astype(np.float32, copy=False)


def write_lora_gguf(
    out_path: str | Path,
    arch: str,
    lora_alpha: float,
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]],
    outtype: str = "q8_0",
    general_name: str | None = None,
    alora_tokens: list[int] | None = None,
    extra_kv: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write logical (A, B) LoRA pairs to a GGUF adapter file and return quantization stats."""
    gguf_mod = ensure_gguf()
    target_qtype, file_type = outtype_to_qtype(gguf_mod, outtype)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf_mod.GGUFWriter(str(out_path), arch=arch, use_temp_file=False)
    writer.add_type(gguf_mod.GGUFType.ADAPTER)
    writer.add_string(gguf_mod.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf_mod.Keys.Adapter.LORA_ALPHA, float(lora_alpha))
    writer.add_file_type(file_type)
    if general_name:
        writer.add_name(general_name)
    writer.add_quantization_version(gguf_mod.GGML_QUANT_VERSION)

    if alora_tokens:
        writer.add_key_value(
            gguf_mod.Keys.Adapter.ALORA_INVOCATION_TOKENS,
            alora_tokens,
            gguf_mod.GGUFValueType.ARRAY,
            gguf_mod.GGUFValueType.UINT32,
        )
    if extra_kv:
        for k, v in extra_kv.items():
            writer.add_string(k, str(v))

    qtype_counts: dict[str, int] = {}
    tensor_bytes = 0
    fallback_tensors: list[str] = []

    for gguf_base, A, B in pairs:
        rank_i = int(A.shape[0])
        # Compensate llama.cpp's global alpha / b->ne[0] rule when per-layer rank differs from lora_alpha.
        a_scaled, b_scaled = A, B
        if lora_alpha != 0.0 and float(rank_i) != float(lora_alpha):
            factor = (float(rank_i) / float(lora_alpha)) ** 0.5
            a_scaled = A * factor
            b_scaled = B * factor

        arr_a, arr_b = ab_to_gguf_arrays(gguf_base, a_scaled, b_scaled)
        for suffix, arr in ((".lora_a", arr_a), (".lora_b", arr_b)):
            tname = gguf_base + suffix
            qdata, actual_qt = quantize_array_for_gguf(gguf_mod, arr, target_qtype)
            if actual_qt != target_qtype:
                fallback_tensors.append(tname)
            qname = actual_qt.name
            qtype_counts[qname] = qtype_counts.get(qname, 0) + 1
            tensor_bytes += int(qdata.nbytes)
            writer.add_tensor(tname, qdata, raw_dtype=actual_qt)

    writer.write_header_to_file(path=out_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "out_path": str(out_path),
        "file_bytes": out_path.stat().st_size,
        "tensor_bytes": tensor_bytes,
        "n_pairs": len(pairs),
        "qtype_counts": qtype_counts,
        "fallback_count": len(fallback_tensors),
        "fallback_sample": fallback_tensors[:6],
    }
