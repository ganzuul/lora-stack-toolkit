---
name: distill-lora
description: "100% local (8 GB VRAM / 24 GiB RAM) LoRA training, online verifiable-reward RL (Stable-RL DPPO + Dr.GRPO + Best-of-G), and same-architecture SVD weight-diff assimilation for Ternary-Bonsai-2-27B-PTQ1_0.gguf. Use when building, adapting, evaluating, or exporting Q8_0 GGUF LoRAs for Bonsai-2-27B from datasets, executable verifier loops, or HF finetunes."
disable-model-invocation: false
---

# `distill-lora`: Local 8 GB VRAM LoRA & Online RL Pipeline for `Ternary-Bonsai-2-27B-PTQ1_0`

**Full engineering spec:** `docs/lora-training-recipe-spec.md` in the
`lora-stack-toolkit` repository (includes the **4-layer Zero-Runtime-Cost SME
Mini-Framework Recipe** — live schema catalog, canonical boundary contract,
IR-first `<think>`, structural verifier — plus empirical laws, promotion gates,
and VRAM budget).
**Runtime hot-swap skill:** `.pi/agent/skills/lora-router/SKILL.md` in the same
repository (`POST /lora-adapters` controller).

Training scripts live in `scripts/distill-lora/` (with shared GGUF/SVD helpers
in `scripts/`). Commands below show the original workstation's model and tool
paths; update them for your environment. Every workflow targets
`Ternary-Bonsai-2-27B-PTQ1_0.gguf` (`qwen35`, 64 layers: 48 Gated DeltaNet + 16
Full Attention) inside an **8 GB VRAM (`RTX 2070 SUPER`, `sm_75`) / 23 GiB RAM**
budget. Keep specialist LoRAs (`r=16`, `L32..63`, `~61 MiB`) **unmerged per
domain** rather than merging overlapping dialects into `r=16`.

Before using the command examples, set `LORA_STACK_REPO` to this repository's
root and `MODEL_DIR` to the directory containing your model weights. Configure
`BASE_GGUF`, `BLENDER_BIN`, `LLAMA_SERVER_BIN`, `QWEN_CHAT_TEMPLATE`, and
`BONSAI_GGUF_PY` as described in the repository README. The examples require
running from the repository root.

## Mandatory Hardware & VRAM Rules

1. **Hide unsupported `sm_61` GPUs & avoid CUDA fragmentation:** Always prefix CUDA commands with:
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0
   ```
   PyTorch `2.11.0+cu130` and `Bonsai-demo`'s CUDA `llama-server` both fail on the secondary `GTX 1050 Ti`.
2. **Mutually exclusive GPU occupancy:** Never run `llama-swap` (`http://127.0.0.1:8080/running`), `llama-server`, and PyTorch `BonsaiPTQStudent` simultaneously. Unload `llama-swap` (`/unload`) before training or benchmarking (`BonsaiPTQStudent` uses ~5.8 GiB VRAM).
3. **Python 3.14 HF dataset rule:** `datasets.load_dataset` hits a `dill` pickling bug under Python 3.14; read raw Parquet/JSON files directly via `huggingface_hub.hf_hub_download` + `pyarrow.parquet` or `json`.
4. **Hadamard + GDN V-head conventions:**
   - `bonsai_ptq_model.py` runs `PTQ1_0` Triton dequantization through a single ~178 MB FP16 scratch buffer, applies `prism.hadamard` (`block_size=1024` + sign vectors) dynamically in the base forward/backward path, and trains LoRA `(lora_a, lora_b)` in the **unrotated primal basis**.
   - `optimize_lora.py` / `lora_gguf_common.py` permute `linear_attn` V-heads from HF grouped `[nk, rep, hd]` to `llama.cpp` tiled `[rep, nk, hd]` (`--no-balance` when keeping exact trained factors) and quantize to `Q8_0` GGUF (`--verify` confirms reconstruction accuracy).

---

## Workflow A — Dataset SFT LoRA (`train_lora.py`)

Use when warm-starting or fine-tuning from curated `(system, prompt, response[, reasoning])` JSONL datasets using **State-Carrying Tiled-BPTT** (`window_len=1024`), full-vocab chunked cross-entropy (`16k` rows/slice), and `bitsandbytes` `AdamW8bit`.

1. **Curate, classify (T1–T4 curriculum), and execution-verify Blender data** across all 5 HF datasets (`BlendNet`, `3dcodeverse-2m`, `blenderllm-v2`, `3DCodeEdit`, `BlenderBench`):
   ```bash
   python3 -u scripts/distill-lora/prepare_blender_data.py \
     --out scripts/distill-lora/work/curriculum_pool.jsonl \
     --build-cycle-batch scripts/distill-lora/work/curriculum_c1_batch.jsonl
   ```
   Prepends a top-of-file Blender 4.2 preamble (`bpy.data.objects.remove` safe cleanup, `make_material`, `assign_material`), normalizes Blender 3.x $\to$ 4.2 BSDF sockets, verifies in 8 parallel headless Blender (`4.2.10`) workers, tags API complexity (`T1` Foundations $\to$ `T2` Multi-part Composition $\to$ `T3` Procedural BMesh/Modifiers/Nodes $\to$ `T4` Harvested `BlenderWorld` RL Winners), and tracks seen IDs in `scripts/distill-lora/work/curriculum_ledger.json`.

2. **Train primal-basis single-domain LoRA & export `Q8_0` GGUF** (target ~80–130 optimizer updates; pass `--init-peft <prev_peft>` only when continuing the same domain):
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 \
   python3 -u scripts/distill-lora/train_lora.py \
     --data scripts/distill-lora/work/strat_a_verified.jsonl \
     --max-samples 60 --max-seq-len 2560 --window-len 512 --epochs 2 --grad-accum 2 \
     --start-layer 32 --rank 16 --alpha 32 --lr 2.2e-4 --shuffle --seed 42 \
     --peft-out scripts/distill-lora/work/peft_strat_a_sft \
     --gguf-out "$MODEL_DIR/Bonsai-2-27B-blender-stratA-sft-Q8_0.gguf"
   ```
   `run_sft()` skips sequences exceeding `--max-seq-len` so SFT never trains on truncated code fences. If VRAM approaches limit on long sequences, raise `--start-layer 40` or lower `--rank 8`.

---

## Workflow B — Single-Domain Online RL (`online_rl_blender.py`: Stable-RL DPPO + Dr.GRPO)

Runs execution-gated online RL over unseen **single-domain** tasks anchored to Stage-1 SFT (`--ref-anchor sft`; avoid mixing competing dialects into one RL run):
1. **Rollout + Verifier Phase:** Starts `Bonsai-demo/bin/cuda/llama-server` with the current LoRA GGUF (and `--reasoning-budget-message` when thinking is enabled), samples `G` completions with token logprobs via `chat_complete_with_logprobs`, and grades every script in headless Blender (`blender_verifier.py`). Passing rollouts (`reward >= 0.53`) are harvested into `work/rl_harvested_winners.jsonl` / `work/rl_gn_harvested_winners.jsonl`.
2. **DPPO Binary-TV + Dr.GRPO Update Phase:** Drops `0/G` pass groups (`--require-geonodes` enforces `is_geonodes_pass()`), uses **Dr.GRPO** unnormalized reward centering (`adv = reward - mean`, fixed `norm_tokens=2048.0`), aligns native `llama-server` rollout logprobs (`bypass_mode`, falling back cleanly if token IDs differ), applies DPPO Binary-TV divergence masking (`clip=0.15`, `is_clip=3.0`) + SFT-anchored `k3` KL penalty (`kl_beta=0.015..0.04`) via Tiled-BPTT (`ppo_epochs=2`, `minibatch_size=8`), and exports updated PEFT + `Q8_0` GGUF.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 \
python3 -u scripts/distill-lora/online_rl_blender.py \
  --init-peft scripts/distill-lora/work/peft_c4_gn_sft \
  --init-gguf "$MODEL_DIR/Bonsai-2-27B-blender-c4-gn-sft-Q8_0.gguf" \
  --peft-out scripts/distill-lora/work/peft_c4_gn_dppo \
  --gguf-out "$MODEL_DIR/Bonsai-2-27B-blender-c4-gn-dppo-Q8_0.gguf" \
  --tasks-jsonl scripts/distill-lora/work/geonodes_rl_tasks.jsonl --require-geonodes \
  --algo dppo --dppo-variant binary_tv --dppo-clip 0.15 --dppo-is-clip 3.0 \
  --dppo-norm-tokens 2048.0 --ref-anchor sft --ppo-epochs 2 --minibatch-size 8 \
  --rounds 2 --tasks-per-round 4 --task-offset 0 --group-size 4 \
  --max-tokens 2560 --reasoning-budget 0 --max-seq-len 3072 --window-len 512 \
  --lr 2.5e-5 --kl-beta 0.015
```

**Held-Out Evaluation (`Qwen/BlenderWorld` text split & Geometry Nodes):**
```bash
CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 \
python3 -u scripts/distill-lora/benchmark_blenderworld.py \
  --limit 8 --max-tokens 2560 --reasoning-budget 0 \
  --lora "$MODEL_DIR/Bonsai-2-27B-blender-c5-unified-sft-v2-Q8_0.gguf:1.0" \
  --out scripts/distill-lora/work/bw_eval.json

CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 \
python3 -u scripts/distill-lora/benchmark_geonodes.py \
  --max-tokens 2560 --reasoning-budget 0 \
  --lora "$MODEL_DIR/Bonsai-2-27B-blender-c4-gn-rl-Q8_0.gguf:1.0" \
  --out scripts/distill-lora/work/gn_eval.json
```

To build a pipeline for another verifiable domain (e.g. SQL, CadQuery, Manim, Three.js): instantiate the 4-layer SME Mini-Framework (Sections 4 and 7 of `docs/lora-training-recipe-spec.md`), replace `blender_verifier.py` with a schema-aware `<domain>_verifier.py` returning `{syntax_ok, exec_ok, reward}`, and reuse `train_lora.py` + `online_rl_blender.py`.

---

## Workflow C — Same-Architecture 27B Finetune Assimilation (`extract_svd_lora.py` + `optimize_lora.py`)

Use when extracting a LoRA directly from an existing same-architecture HF checkpoint $\Delta W = W_{\text{finetune}} - W_{\text{base}}$ (e.g. a `Qwen3.5-27B` / `Qwen3.8-27B` fine-tune) via layer-streamed shard loading and GPU Gram-eigh SVD. **`--rank 32` is the validated 8 GB VRAM core rank** (`77.61%` global BF16 energy / `94.83%` of `r=64` energy at `213 MiB` `Q8_0` vs `426 MiB` at `r=64`):

```bash
CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 \
python3 -u scripts/extract_svd_lora.py \
  --base Qwen/Qwen3.8-27B \
  --finetune <org/finetune-27b-or-local-dir> \
  --out scripts/distill-lora/work/peft_svd \
  --gguf-out "$MODEL_DIR/Bonsai-2-27B-svd-r32-Q8_0.gguf" \
  --outtype q8_0 --svd-mode gram --rank 32
```
For disk-constrained downloads, pass `--hf-cache-dir /run/media/nos/games/hf-cache --evict-shards`.
To re-quantize, prune, rank-reduce, or verify any existing PEFT or GGUF LoRA:
```bash
python3 scripts/optimize_lora.py <peft-dir-or-gguf> \
  --base Qwen/Qwen3.8-27B -o <out.gguf> --outtype q8_0 --verify
```
- **Core LoRA merging (`merge_loras.py`, Law 4):** When merging two orthogonal core LoRAs under an 8 GiB `ctx=64000` budget (e.g. `pi-lora` + `qwopus-r32`), never use unweighted `naive` SVD (`pi` collapses to `52.4%`) or `Q5_0` (`3/6` stacked GeoNodes); use FP32 `maximin` rank-allocated merging into block-32 aligned `Q8_0` (`213.21 MiB`, `0` F16 fallbacks, `5/6` stacked GeoNodes, fits `ctx=64000` + `2` dormant SMEs):
  ```bash
  python3 -u scripts/merge_loras.py --a <coreA> --b <coreB> --pre-rank-b 32 --method maximin --rank 32 --outtype q8_0 -o <merged_core_Q8_0.gguf> --verify
  ```

---

## Self-Tests & Diagnostics

- **Headless Blender verifier:** `python3 scripts/distill-lora/blender_verifier.py --selftest`
- **Triton `PTQ1_0` exactness:** `CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 python3 scripts/distill-lora/bonsai_ptq_model.py --test-triton`
- **Live `llama.cpp` logit equivalence:** `CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 python3 scripts/distill-lora/bonsai_ptq_model.py --verify-equivalence`

$ARGUMENTS
