# Specialist & Core LoRA Training Recipe Specification (`Ternary-Bonsai-2-27B-PTQ1_0`)

**Scope:** Reproducible engineering specification for training, assimilating, evaluating, and serving `Q8_0` LoRA adapters on `/home/nos/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (`qwen35`, `64` blocks: `48` Gated DeltaNet + `16` Full Attention) within an **8 GiB VRAM (`RTX 2070 SUPER`, `sm_75`) / 23 GiB RAM** budget.

The bundled training/benchmark scripts live under `scripts/distill-lora/` and
shared GGUF/SVD utilities live under `scripts/`. For live hot-swapping on
`qwen27b`, see the bundled Pi harness skill at
`../.pi/agent/skills/lora-router/SKILL.md`. The separate Gemini
`distill-lora` skill is not included in this repository.

---

## 1. Target Architecture: Core Stack + Hot-Swapped SMEs

The production architecture operates as a **two-tier Expert-LoRA Router** over the quantized general core: **Base (`PTQ1_0`) → Core Stack (`pi-lora` + `qwopus-r32`) → Hot-Swapped SME (Subject Matter Expert) LoRAs**.

| Tier | Lifecycle | Adapter Layout | Verified Reference Artifacts |
|---|---|---|---|
| **Base Model** | Always resident | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (`token_embd.weight=CPU`, dynamic `prism.hadamard` `block_size=1024` + sign vectors) | `/home/nos/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf` |
| **Core Stack** | Always-on (`scale=1.0`) | General harness behavior + reasoning (`L0..63`, FFN + full-attn + GDN) | **Merged `Q8_0` (`ctx=64000` recommended):** `Qwen3.8-27B-core-pi-qwopus-maximin-r32-Q8_0.gguf` (`213.21 MiB`, `380` pairs, `maximin` `r=32`, `Q8_0`)<br>*Separate `2× Q8_0` gold reference (`374.66 MiB`):* `Qwen3.8-27B-pi-lora-Q8_0.gguf` (`161.50 MiB`, `256` pairs, `\|\|ΔW\|\|=2.97`) + `Qwen3.8-27B-qwopus-thinkingcap-swift-r32-Q8_0.gguf` (`213.16 MiB`, `380` pairs, `\|\|ΔW\|\|=9.68`) |
| **SME LoRAs** | Pre-loaded dormant (`:0.0`), toggled via `POST /lora-adapters` (`0.25–0.61 ms`) | `r=16`, `alpha=32`, `L32..63` (`128` pairs: `8×4` Full-Attn `q/k/v/o` + `32×3` FFN `gate/up/down`), `~61 MiB` `Q8_0` | **GeoNodes SME:** `Bonsai-2-27B-blender-c6-gn-sft-Q8_0.gguf` (`6/6` standalone `2K` think, `6/6` stacked `2K` with `2× Q8_0`, `5/6` stacked `2K` with `maximin-r32` core)<br>**BlenderWorld SME:** `Bonsai-2-27B-blender-c5-unified-sft-v2-Q8_0.gguf` (`7/8 = 87.5%` standalone) |

---

## 2. Hard Environment & Numerics Invariants

1. **GPU Masking & CUDA Allocator:** Always prefix GPU commands with:
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0
   ```
   With `CUDA_DEVICE_ORDER=FASTEST_FIRST`, GPU `0` is the `RTX 2070 SUPER` (`sm_75`). The secondary `GTX 1050 Ti` (`sm_61`) is unsupported by both PyTorch `2.11.0+cu130` and `Bonsai-demo/bin/cuda/llama-server`.
2. **Mutually Exclusive GPU Occupancy:** Never run `llama-swap` (`http://127.0.0.1:8080`), standalone `llama-server`, and PyTorch `BonsaiPTQStudent` simultaneously (`BonsaiPTQStudent` requires `>5000 MiB` free VRAM and peaks near `~5.8 GiB`). Query `GET http://127.0.0.1:8080/running` and call `POST http://127.0.0.1:8080/api/models/unload` first (`train_lora.py` / `benchmark_blenderworld.py` do this automatically).
3. **Python 3.14 HF Dataset Rule:** `datasets.load_dataset` hits a `dill` pickling bug under Python 3.14. Always read Parquet/JSON files directly via `huggingface_hub.hf_hub_download` + `pyarrow.parquet` or stdlib `json`.
4. **Primal-Basis Training & GDN V-Head Layout:**
   - `bonsai_ptq_model.py` keeps packed `PTQ1_0` weights in VRAM (`~5.38 GB` blocks + `0.28 GB` output head, `token_embd` on CPU), dequantizes one `Linear` at a time into a single `~178 MB` FP16 scratch buffer, applies `prism.hadamard` in the base path, and trains LoRA `(lora_A, lora_B)` in the **unrotated primal basis**.
   - `optimize_lora.py` (with `--no-balance` when keeping exact trained factors) permutes `linear_attn` V-heads from HF grouped `[nk, rep, hd]` to `llama.cpp` tiled `[rep, nk, hd]` (including `linear_attn.out_proj -> ssm_out`), writes `Q8_0` GGUF, and `--verify` confirms reconstruction accuracy (`0` F16 fallbacks, `cos >= 0.99996`).
5. **Memory-Constrained Long Sequences:** `train_lora.py` uses **State-Carrying Tiled-BPTT** (`--window-len 512` default on CLI / `1024` function default), gradient checkpointing, chunked full-vocab logprobs (`16k` rows/slice), and `bitsandbytes.optim.AdamW8bit`. If VRAM approaches limit on long traces, raise `--start-layer` from `32` to `40` or reduce `--rank` to `8`.

---

## 3. Empirically Derived Architecture Laws

These laws come directly from Cycle 1–5 evaluations, SVD D3 comparison, pairwise weight-contestation audits, `r=32 -> r=16` projection audits, and the live 3-LoRA causal isolation run (`eval_gn_01/02/06`).

### Law 1 — Cross-Domain Stacking Is Orthogonal (Router-Safe)
Specialists trained in `L32..63` (`128` pairs across full-attn + FFN) have **noise-floor weight overlap** with both `pi-lora` (`256` pairs `L0..63`) and `qwopus-r32` (`380` pairs `L0..63`):

| Pair | Shared Tensors | Mean Matrix Cosine (`max \|cos\|`) | Neg Cosines | Energy Overlap (Col / Row) | Empirical Stacking Outcome |
|---|---:|---:|---:|---:|---|
| `pi-lora` vs `qwopus-r32` *(control)* | 256 (`L0..63`) | `-0.0000` (`0.0020`) | `130/256` | `1.3%` / `0.5%` | **4/4 coherent** (`6,736 MiB`) |
| `c4_gn` vs `pi-lora` | 128 (`L32..63`) | `+0.0000` (`0.0009`) | `56/128` | `1.1%` / `0.5%` | **3/3** `gn_pass` (`R=1.00, 1.00, 1.00`) |
| `c4_gn` vs `qwopus-r32` | 128 (`L32..63`) | `+0.0001` (`0.0013`) | `56/128` | `1.2%` / `0.7%` | `2/3` (`wave_panel` failed alone) |
| **`c4_gn + pi + qwopus-r32`** | 128 (`L32..63`) | — | — | — | **3/3 `gn_pass`** (`R=1.00, 0.90, 1.00`, `3,055–3,331` chars) |

### Law 2 — Never Merge Overlapping Same-Domain Dialects Into `r=16`
Merging `c4_gn` (`GeometryNode*` modifiers) and `c5_bw` (`bpy.ops.mesh` / `bmesh`) at `0.90*c4_gn + 0.75*c5_bw` into `r=16` dropped both benchmarks (`GeoNodes`: `5/6 -> 3/6`; `BlenderWorld`: `7/8 -> 5/8`) due to two distinct mechanisms:
1. **Late-Layer Output-Subspace Contestation (`0/128` sign conflicts, `mean cos = +0.0858`, `col_ov = 26.2%`, `mean max_pcos = 0.666`):**
   - Modules most heavily contested: `attn_v` (`37.0%` col overlap), `attn_q` (`33.9%` col / `10.6%` row, `cos=+0.170`), `attn_output` (`29.9%` col), `ffn_gate` (`28.4%` col / `8.2%` row, `cos=+0.142`).
   - Peak tensors: `blk.63.attn_q` (`cos=+0.4147`, `52.8%` col / `32.9%` row, top-1 output vector alignment `|u1_0·u2_0|=0.766`), `blk.63.attn_k` (`44.3%` col), `blk.62.ffn_up` (`41.2%` col), `blk.55.attn_q` (`39.7%` col), `blk.61.ffn_gate` (`36.9%` col). Because `75–95%` of each matrix's energy sits in its top 3 singular directions, averaging blends competing token dialects inside the same output heads.
2. **Rank-16 Truncation Loss (`r=32 -> r=16`):** Modules carrying distinct features lost substantial energy: `k_proj` (`13.6%` avg, `L43` `17.3%`), `up_proj` (`12.1%` avg, `L39` `17.6%`), and `down_proj` (`11.3%` avg, `L58` `15.7%`). Only `81.4%` (`c4_gn`) / `77.8%` (`c5_bw`) of `k_proj` energy survived inside the `r=16` subspace.

**Rule:** Keep dialect/sub-skill experts as separate `61 MiB` `r=16` adapters and route between them via `POST /lora-adapters` (`0.3–0.6 ms`). Only if a single file is strictly unavoidable, use lossless `r=32` orthogonal concatenation (`122 MiB`, zero rank truncation) rather than `r=16` compression.

### Law 3 — Norm Hygiene & Pure-Domain RL (Prevent Policy Boundary Brittleness)
- `qwopus-r32` injects `||ΔW|| = 5.68` across `185` tensors in `L0..31` (peaking in `L24..31` right before layer 32, e.g., `blk.29/31/28.ffn_gate = 0.94/0.92/0.90`) plus `67` Gated DeltaNet tensors (`ssm_out`, `attn_qkv`, `attn_gate`) in `L32..63` that `L32..63` specialists never touch. `pi-lora` adds another `||ΔW|| = 1.97` in `L0..31`.
- Pure `c4_gn` (`||ΔW|| = 11.49`) absorbs that upstream hidden-state shift (`3/3` stacked). By contrast, mixed-domain DPPO on the blurred `r=16` merge inflated `c5_dppo` to `||ΔW|| = 15.09` (`ffn_gate` `7.65 -> 10.53`, `attn_q` `3.41 -> 4.84`, with a `kl=8.02` RL spike on `rl_gn_06`), causing `eval_gn_02`/`03` to cross back into `BlenderWorld` habits, hit `finish=length` (`8,919–10,211` chars), and drop to `1/6` stacked (`gn_eval_c5_dppo_stacked.json`).
- **Rule:** Never run mixed-domain RL on a merged adapter, keep specialist total Frobenius norm compact (`||ΔW|| ≲ 12.5`), and abort/tighten `--kl-beta` if per-sample `kl > 1.0`.

### Law 4 — Core Adapter Merging & Quantization (Rank-Allocated FP32 Merge + `Q8_0` Beats Sub-8-Bit Quantization)
When compressing the orthogonal always-on core stack (`pi-lora-F16` [`256` pairs, `r=32`] + `qwopus-r32` [`380` pairs = `256` shared + `124` DeltaNet-only]) to fit `ctx=64000` + hot-swapped SMEs on 8 GiB VRAM, controlled CPU/GPU experiments established four mechanical rules:
1. **Lossless orthogonal merge saves `0 MiB` of weight VRAM:** Because `pi-lora` and `qwopus-r32` are orthogonal (`mean cos ≈ 0`, `1.3%` col overlap), concatenating them without truncation requires `r=64` on the `256` shared pairs + `r=32` on the `124` DeltaNet pairs = **`374.66 MiB`** in `Q8_0` (identical weight footprint to the separate `2× Q8_0` files).
2. **Naive unweighted `r=32` QR-SVD starves the smaller adapter under norm mismatch:** On shared tensors, $\|\Delta W_{\text{qw}}\| \approx 2.8\times \|\Delta W_{\text{pi}}\|$ (`ffn_gate` `3.68×`, `8–13×` Frobenius² energy). Naive `r=32` QR-SVD retains `98.42%` of `qwopus` but collapses `pi-lora` to **`52.40%` (`p10 = 10.1%`, `min = 1.8%`)** and degraded deterministic 4-test coherence to **`3/4`** (`test3_python_debug` AST failure). Per-tensor **maximin rank-allocated FP32 concatenation** — choosing $k_a + k_b = 32$ to maximize $\min(E_a(k_a), E_b(k_b))$ (`mean k_pi = 21.625`) before exact lossless `r=32` QR-SVD balancing — protects both adapters (`pi = 88.08%` [`min 63.4%`], `qw32 = 90.67%` [`min 63.3%`]) and recovers **`4/4`** coherence (`min_cos = 0.999960`).
3. **`Q8_0` block-32 alignment cliff (`r ∈ {32, 64}` only):** In GGUF, `lora_b` has `ne[0] = rank`; non-multiples of `32` fall back to `F16` (`2×` bytes/param). Consequently, `r=24` (`234.3 MiB`) is larger than `r=32` (`213.2 MiB`), and `r=48` (`417.0 MiB`) is larger than lossless `r=64` (`374.7 MiB`). Mixed-rank merges must snap strictly to **`r ∈ {32, 64}`** (`0` `F16` fallbacks). `llama.cpp` natively scales per-tensor rank via `b->ne[0]` (`llama-adapter.h`), which `write_lora_gguf` compensates cleanly (`KL_all = 0.00086`, `-46%` `KL_qw` vs `maximin-r32`).
4. **Tail-truncated `Q8_0` beats full-rank `Q5_0` on multi-step synthesis, and 1 core LoRA cuts CUDA graph overhead:** Even though `2× Q5_0` showed small single-token KL (`0.00065`), stochastic `Q5_0` noise across *all* dominant singular modes compounded over 2K-token rollouts and dropped stacked GeoNodes (`+ c6-gn-sft-Q8_0`) to **`3/6` (`50.0%`, `R=0.5750`)**, whereas `maximin-r32` `Q8_0` scored **`5/6` (`83.3%`, `R=0.8583`)** in `-29.3 MiB` smaller file size (note `5/6` vs `hybrid-64`'s `4/6` is within single-run `temp=0.4` variance, while `maximin-r32` wins decisively on VRAM). Collapsing `2` core LoRAs into `1` also cuts `llama-server` `ctx=64000` CUDA graph capture cost from `94 MiB` → **`72 MiB`** (`1` SME) and from OOM → **`104 MiB`** (`2` simultaneous SMEs).

| Core Profile | Weight Size | Retained `pi` / `qw32` | `KL_all` (`pi` / `qw`) vs `2× Q8_0` | Pi Stress / Coh | Stacked `2K` GN | `ctx=64000` Free (`1-SME` / `2-SME` end) |
|---|---:|---:|---:|---:|---:|---|
| `2× Q8_0` separate *(gold ref)* | `374.66 MiB` | `100%` / `100%` | `0.00000` (`0` / `0`) | `4/4` / `4/4` | **`6/6` (`R=1.000`)** | Boot OOM w/ `1` dormant SME (`58` < `94 MiB` needed) |
| `2× Q5_0` separate | `242.51 MiB` | full-rank `~0.998 cos` | `0.00065` (`0.00078` / `0.00051`) | `4/4` / `4/4` | `3/6` (`R=0.575`) | **`94 MiB`** free (`1-SME`) / **OOM** (`3rd graph`) |
| `naive-r32` merged `Q8_0` | `213.21 MiB` | `52.40%` (`min 1.8%`) / `98.42%` | `0.00051` (`0.00088` / `0.00015`) | `4/4` / **`3/4`** *(fail)* | `—` | `—` |
| **`maximin-r32` merged `Q8_0` *(winner)*** | **`213.21 MiB`** | **`88.08%` (`63.4%`) / `90.67%` (`63.3%`)** | `0.00137` (`0.00108` / `0.00166`) | **`4/4` / `4/4`** | **`5/6` (`R=0.858`)** | **`158 MiB` (`72Δ`) / `64 MiB` (`104Δ`, 0.37–0.39 ms)** |
| `hybrid-64` merged `{32, 64}` `Q8_0` | `257.97 MiB` | `94.20%` / `94.98%` | `0.00086` (`0.00082` / `0.00090`) | `4/4` / `4/4` | `4/6` (`R=0.717`) | **`112 MiB` (`72Δ`)** / `18 MiB` (`104Δ`, tight) |

**Canonical Core Merge Command (`merge_loras.py`):**
```bash
python3 -u /home/nos/scripts/merge_loras.py \
  --a /home/nos/models/Qwen3.8-27B-pi-lora-F16.gguf \
  --b /home/nos/scripts/distill-lora/work/peft_qwopus_tc_swift_r64 \
  --pre-rank-b 32 --method maximin --rank 32 --outtype q8_0 \
  --device cpu --local-files-only --verify \
  --name core-pi-qwopus-maximin-r32 \
  -o /home/nos/models/Qwen3.8-27B-core-pi-qwopus-maximin-r32-Q8_0.gguf
```
*(Pass `--method hybrid --rank 32 --upgrade-top 64` for a `257.97 MiB` 1-SME-only variant that prioritizes single-token distribution fidelity while staying block-32 aligned in `Q8_0`.)*

---

## 4. The Zero-Runtime-Cost SME Mini-Framework Recipe

### 4.1 Core Principle: Elaboration-Time Framework, Zero-Dependency Output
Audit of Cycle 5 failures alongside `/home/nos/engram-lse/docs/reference/Argon_on_bit-rot.md` revealed that **where helper conventions reside decides whether an SME learns clean API semantics or accumulates cross-dataset drift**:

1. **Avoid heavyweight runtime SDK rot:** Shipping a custom wrapper module (e.g., `import bpy_kit`) inside generated scripts couples every output to harness plumbing that breaks outside the sandbox or across upstream releases. Emitted code must remain plain, standalone stock API.
2. **Avoid local-only verification (`exec_ok` alone):** Checking each example in isolation misses cross-dataset convention mismatches ("patching cocycles"). In Cycle 5 (`curriculum_c5_unified.jsonl`, `128` rows), every row passed headless execution on the **same** `Blender 5.4.0 Alpha` binary, yet the dataset simultaneously labeled prompts `Blender 4.2.10` (`54` rows) vs. `Blender 5.4` (`74` rows) and split identical material setup across **three incompatible helper conventions**: `make_material` + `assign_material` (`54` rows), `make_mat` (`39` rows), and `pbr_mat` (`35` rows). Training one `r=16` adapter across those conflicting local conventions burned capacity on boilerplate arbitration, contributed to `blk.63.attn_q` `52.8%` output-subspace contestation, and produced literal mashup crashes (`NameError: name 'make_pbr_mat' is not defined`).

**Design Rule:** Move the mini-framework completely out of runtime outputs and enforce it **strictly at dataset-elaboration and RL-verification time**. Every script stays 100% vanilla target API, while the dataset builder and verifier enforce one flat, schema-checked canonical contract.

### 4.2 The 4-Layer Abstract SME Mini-Framework

When building any API or tool-domain SME LoRA, instantiate these four layers:

1. **Layer 1 — Live Runtime Schema Catalog (`schema_catalog.json`):**
   - Introspect the actual installed target binary/runtime before generating data or verifying rollouts to dump exact symbol/node IDs, input/output port names, value types, and the exact runtime version string.
   - Prevents version lag and hallucinated APIs observed in Cycle 5 (e.g., `GeometryNodeCurveCircle`, `GeometryNodeCurvePrimitiveBezier`, `"Sweep Angle"`, `"Sweep Radius"`, `"Middle Radius"`).
2. **Layer 2 — Canonical Boundary Contract (Flat Boilerplate Gauge & Anti-Prompt-Crutch Rule):**
   - Explicitly partition each script into **invariant boundary plumbing** (scene/resource reset, material/connection init, render/serialization target, version header) vs. **core domain synthesis**.
   - Lock every prompt, system message, SFT example, RL task, and held-out eval onto **one target version** and **one canonical boundary idiom**. Reject legacy version tags (`4.2.10` on a `5.4` binary) or unregistered helper aliases (`pbr_mat` vs `make_mat` vs `make_material`) at dataset-build time.
   - **Never condition SFT exclusively on prompt API hints ("prompt-crutch overfitting"):** Including exact node/socket hints inside `75/75` training prompts (`c6-gn-sft`) scored `5/6` with the benchmark suffix but collapsed to **`0/4`** Turn-1 pass (`write=2/4`, `repair=False`, `mean <think>=3,236 chars`, `71.1 min` session) on bare Pi-agent prompts (`c6` forgot `bpy.data.node_groups.new(..., "GeometryNodeTree")` + `NodeGroupOutput`). Cycle 7 (`108` rows: `54.6%` bare prompts, `16` Pi `write`, `8` 1-step `edit` repairs, `4` `unload` traces, `16` radial assemblies) jumped bare Pi Turn-1 pass to **`2/4`** (`write=4/4`, `repair=True`, `unload=True`, `mean <think>=127 chars` [`-96%`]), though `r=16` SFT alone traded standard benchmark-suffix `2K` GeoNodes (`5/6` `c6` → `4/6` `c7-ep2` → `3/6` `c7-ep3`), so `config.yaml` stays on `c6-gn-sft` pending either Stage-2 DPPO on `c7` or explicit selection.
3. **Layer 3 — Mental-Model IR Inside `<think>` (Domain IR → Stock Code):**
   - Rather than free-form prose, train reasoned traces to formulate the domain's native mental model as a compact, typed intermediate representation before emitting stock code.
   - As `qwen27b` diagnosed for Geometry Nodes — *"getting comfortable with pin-based data flow and the 'data is geometry' mental model (vertices/attributes/points/instances are just data types flowing between nodes)"* — the GeoNodes IR is a pin-level DAG checked mentally before lowering to `bpy`:
     ```text
     [Flow] CurveSpiral(Rotations, Start/End Radius, Height)[Curve]
       -> SetCurveRadius(Radius <- SplineParameter[Factor] -> MapRange[Result])[Curve]
       -> CurveToMesh(Profile <- CurvePrimitiveCircle[Curve])[Mesh]
       -> SetShadeSmooth -> SetMaterial -> GroupOutput[Geometry]
     ```
   - Pair IR-structured `<think>` traces with explicit `--reasoning-budget-message` cutoff examples so bounded thinking (`reasoning_budget=2048`) transitions cleanly into executable code.
4. **Layer 4 — Structural Verifier & Execution-Model Charting:**
   - Upgrade verifiers beyond process exit `0` to check Layer 1 schema validity and structural graph invariants: valid node/symbol types, valid pin names and type-compatible links, zero disconnected dead nodes, and proper boundary handoff (e.g., GeoNodes mesh realized cleanly on object/modifier).
   - Separate SMEs along **execution-model boundaries** rather than forcing conflicting paradigms into one `r=16` adapter (e.g., **Imperative Scene Mutation** [`BlenderWorld`: object placement, lights, cameras, materials] vs. **Declarative Pin-Based Dataflow** [`GeoNodes`: node graphs synthesizing procedural meshes]). Compose them at the asset/harness boundary via `lora-router` (`POST /lora-adapters`) and certify composition with Gates C & D.

### 4.3 Instantiation Matrix Across Domains

| Domain / Execution Model | Layer 1: Live Schema Catalog | Layer 2: Canonical Boundary Contract | Layer 3: `<think>` Mental-Model IR | Layer 4: Structural Verifier Invariants |
|---|---|---|---|---|
| **Blender 5.4 GeoNodes** *(Declarative Pin Dataflow SME)* | `blender-5.4.0` dump of `GeometryNode*`/`ShaderNode*` IDs + socket names & socket types | `Blender 5.4` prompts only; one unified minimal scene/reset/Principled-BSDF/Cycles preamble | Pin-typed node DAG (`Node(in <- Src[Pin])[Out] -> GroupOutput[Geometry]`) | Valid node/pin names, socket type compatibility, connected non-empty `GeometryNodeTree`, realized `>=32` verts |
| **Blender 5.4 `BlenderWorld`** *(Imperative Scene SME)* | `blender-5.4.0` `bpy.ops` / `bmesh` / `Principled BSDF` socket catalog (`Coat Weight`, `Specular IOR Level`) | Same canonical `Blender 5.4` scene/material/camera/lighting contract; strict non-empty Cycles/blend save | Spatial layout plan (coords, bounding volumes, object hierarchy, lighting/material table) | Headless Blender exit `0`, object count/rubric assertions, zero missing context/operator `poll()` errors |
| **CadQuery** *(CSG / B-Rep SME)* | Installed `cadquery` Workplane/selector API + kernel signatures | Unified units (`mm`), single export helper (`cq.exporters.export`), zero viewer imports | Parametric feature/CSG tree (sketches, constraints, extrudes, booleans, fillets) | Non-null solid `isValid()`, positive volume/bbox, watertight manifold shell export |
| **SQL** *(Relational Algebra SME)* | Introspected `PRAGMA`/catalog table & column schema + foreign keys + dialect functions | Read-only transaction / explicit schema prefix, deterministic ordering & alias style | Relational operator plan (joins, filter predicates, group-by grain, window partition) | EXPLAIN/AST parse validity, zero unknown columns/ambiguous joins, exact result-set assertions |

---

## 5. End-to-End Specialist Training Recipe

### Stage 0 — Domain Contract, Headless Verifier & Leak-Free Split
Instantiate the **4-Layer SME Mini-Framework (Section 4)** before curating data:
1. **Schema & Structural Verifier:** Extract Layer 1 schema truth from the target binary and build a deterministic headless verifier returning `{syntax_ok, exec_ok, reward ∈ [0, 1]}` plus structural domain checks (e.g., `blender_verifier.py` checks headless Blender exit, non-empty scene, rubric hits, and `is_geonodes_pass()` requires `>=1` `GeometryNodeTree`, `>=4` nodes, `>=3` links, `>=32` vertices).
2. **Boundary Lock & Hard Split Separation:** Enforce Layer 2's flat version/boilerplate contract across every row, and separate **(a) SFT training pool**, **(b) RL prompt bank**, and **(c) held-out eval suite** with automated string/ID anti-leakage checks (`0` shared prompt/concept signatures across all text fields, reasoning included).
3. **Mode, IR & Budget Alignment:** Format both reasoning-off (`reasoning=""`, `<think>\n\n</think>\n\n`) and Layer 3 IR `<think>` traces cleanly with `/home/nos/scripts/qwen_chat_template.jinja`. Whenever training thinking-budget transitions, include explicit budget-cutoff examples ending with:
   ```text
   [Thinking budget reached. Stop planning and immediately output the complete, executable Blender Python script now.]
   ```
   and configure inference servers identically (`--reasoning-budget-message ...`).

### Stage 1 — Execution-Verified SFT Warm-Start (`train_lora.py`)
1. **Dataset Sizing & Mix:**
   - Curate **`60–150` execution-verified single-dialect examples** (`~60–70%` direct `reasoning=""` + `~30–40%` concise `reasoning` traces, targeting completion lengths well inside `--max-seq-len`).
   - Do **not** mix competing dialects if you want peak single-skill fidelity (in Cycle 5 SFT v2, adding `54` BlenderWorld + `74` GeoNodes lifted BlenderWorld to `7/8` [`87.5%`] while dropping GeoNodes to `1/6`).
2. **Compute Sufficient Optimizer Steps (~80–130 updates):**
   - Cycle 5 Empirical Lesson: `32` updates at `lr=1.5e-4` undertrained (`||B||=4.77` → `1/6` GN, `1/8` BW), whereas `128` updates at `lr=2.2e-4` converged cleanly (`||B||=9.97` → **`7/8` BW**, `mean_reward=0.6734`).
   - **Never train on truncated code fences:** `train_lora.py`'s `run_sft()` skips sequences exceeding `--max-seq-len` automatically (`truncate=False` during filtering).
3. **Canonical SFT Command:**
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 \
   python3 -u /home/nos/scripts/distill-lora/train_lora.py \
     --data /home/nos/scripts/distill-lora/work/<domain_sft>.jsonl \
     --max-seq-len 2560 --window-len 512 --epochs 2 --grad-accum 2 \
     --start-layer 32 --rank 16 --alpha 32 --lr 2.2e-4 --shuffle --seed 42 \
     --peft-out /home/nos/scripts/distill-lora/work/peft_<name>_sft \
     --gguf-out /home/nos/models/Bonsai-2-27B-<name>-sft-Q8_0.gguf
   ```
   *(Pass `--init-peft <prev_peft>` only when continuing the **same** specialist across curriculum cycles.)*

### Stage 2 — Single-Domain Online RL (`online_rl_blender.py`: Stable-RL DPPO + Dr.GRPO)
Run on-policy verifiable RL over **unseen same-domain prompts** anchored to the Stage 1 SFT checkpoint (`--ref-anchor sft`):
- **DPPO Binary-TV (`--algo dppo --dppo-variant binary_tv --dppo-clip 0.15 --dppo-is-clip 3.0`):** Masks tokens whose probability deviates too far from the `llama-server` rollout policy (`|p_π - p_roll| > 0.15` in the update direction), eliminating off-policy quantization/kernel ratio spikes.
- **Native `bypass_mode`:** `chat_complete_with_logprobs` captures exact `llama-server` rollout token IDs and logprobs; `_align_rollout_logprobs` reuses them directly when `>=98%` token-ID match holds (verified `256/256` exact match; falls back to `compute_sequence_logps` only if alignment fails).
- **Dr.GRPO Loss Normalization (`--dppo-norm-tokens 2048.0`):** Uses unnormalized reward centering (`adv = reward - mean(group_rewards)`, mild `0.25` floor only when all `G/G` rollouts pass identically) divided by a fixed token constant (`2048.0`) instead of `std` division or variable `1/L_i` scaling — removing both difficulty bias and length bias.
- **Strict Execution Gating:** Groups with `0/G` `domain_pass` (`gn_pass` or `exec_ok`) receive zero advantage (never reinforce non-executing syntax); non-domain fallback executions are capped at `reward <= 0.25`. Passing trajectories with `reward >= 0.53` are harvested into `work/rl_gn_harvested_winners.jsonl` / `work/rl_harvested_winners.jsonl`.
- **Canonical DPPO RL Command (Single-Domain Specialist):**
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 \
  python3 -u /home/nos/scripts/distill-lora/online_rl_blender.py \
    --init-peft /home/nos/scripts/distill-lora/work/peft_<name>_sft \
    --init-gguf /home/nos/models/Bonsai-2-27B-<name>-sft-Q8_0.gguf \
    --peft-out /home/nos/scripts/distill-lora/work/peft_<name>_dppo \
    --gguf-out /home/nos/models/Bonsai-2-27B-<name>-dppo-Q8_0.gguf \
    --tasks-jsonl /home/nos/scripts/distill-lora/work/<domain_rl_tasks>.jsonl \
    --algo dppo --dppo-variant binary_tv --dppo-clip 0.15 --dppo-is-clip 3.0 \
    --dppo-norm-tokens 2048.0 --ref-anchor sft --ppo-epochs 2 --minibatch-size 8 \
    --rounds 2 --tasks-per-round 4 --task-offset 0 --group-size 4 \
    --max-tokens 2560 --reasoning-budget 0 --max-seq-len 3072 --window-len 512 \
    --lr 2.5e-5 --kl-beta 0.03
  ```
  Add `--require-geonodes` when `domain_pass` requires Geometry Nodes, and keep `--mix-blenderworld-tasks 0` for pure specialists. *(Watch `kl=` in logs: keep mean `kl < 0.3`; if any rollout spikes `> 1.0`, lower `--lr` to `1.5e-5`–`2e-5` or raise `--kl-beta` to `0.04–0.05`.)*

### Stage 3 — Optional Interleaved Cycle (Same-Domain SFT ↔ DPPO)
When scaling capability within one domain:
1. Fold harvested `rl_*_harvested_winners.jsonl` (`R >= 0.53`, deduplicated by `task_id`) back into the SFT dataset alongside newly curated recipes for failure modes.
2. Re-run SFT (`--init-peft` or fresh clean warm-start) followed by another DPPO pass advancing `--task-offset` to fresh RL prompts.

### Stage 4 — Alternative Path: Same-Architecture HF Finetune SVD Extraction (Core / General Adapters)
Use `extract_svd_lora.py` + `optimize_lora.py` when assimilating an existing full-weight `Qwen3.8-27B` / `Qwen3.5-27B` fine-tune (`ΔW = W_finetune - W_base`) rather than training from scratch:
```bash
CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 \
python3 -u /home/nos/scripts/extract_svd_lora.py \
  --base Qwen/Qwen3.8-27B \
  --finetune <org/finetune-27b-or-local-dir> \
  --out /home/nos/scripts/distill-lora/work/peft_svd \
  --gguf-out /home/nos/models/Bonsai-2-27B-svd-r32-Q8_0.gguf \
  --outtype q8_0 --svd-mode gram --rank 32 \
  --hf-cache-dir /run/media/nos/disk/hf-cache-svd --evict-shards
```
- **`r=32` is the validated sweet spot** for general/core adapters on 8 GB VRAM: on Qwopus-ThinkingCap (`380` pairs, `213.21 MiB`), `r=32` retains **`77.61%`** of global BF16 energy vs `81.84%` at `r=64` (`426.38 MiB`) — losing only `-4.23%` (`94.83%` of `r=64`'s captured energy) at half the VRAM, with Q8_0 readback cosine `0.999964–0.999982`.

---

## 6. Promotion Gates & 8 GiB VRAM Deployment Budget

Before adding any new specialist to `llama-swap` (`qwen27b`), verify all 5 gates:

| Gate | Check | Pass Criterion |
|---|---|---|
| **A. Quant & V-head integrity** | `python3 /home/nos/scripts/optimize_lora.py <peft-or-gguf> --base Qwen/Qwen3.8-27B -o <out.gguf> --outtype q8_0 --no-balance --verify` | `F16 fallbacks: 0`, `min_cos >= 0.9999` |
| **B. Standalone held-out eval** | `benchmark_geonodes.py` / `benchmark_blenderworld.py` (`--reasoning-budget 0` and optionally `2048`) | Matches or beats prior best on target domain (`0` leaked tasks) |
| **C. Norm & orthogonality check** | Frobenius `\|\|ΔW\|\|` and shared-tensor cosine against `pi-lora` + `qwopus-r32` | `\|\|ΔW\|\| ≲ 12.5` and `\|mean cos\| < 0.002` |
| **D. Stacked 3-LoRA eval** | Stress-test with `--lora pi-lora:1.0,qwopus-r32:1.0,specialist:1.0` + general coherence check | Domain pass preserved cleanly without `length` runaways |
| **E. Production `ctx=64000` VRAM** | `nvidia-smi` after loading `qwen27b` via `llama-swap` | Fits inside `8,192 MiB` (`>30 MiB` free margin post-gen) |

**Measured VRAM Envelope on `RTX 2070 SUPER` (`7,786 MiB` CUDA-usable, `ctx=64000`, `q4_0` KV, `--kv-offload`, `token_embd.weight=CPU`):**

The full-scale Gate D command above is a historical robustness/stress test, not
the current router's operational default. For routine specialist use, start
with the damped core/specialist settings in [scale-selection.md](./scale-selection.md),
then validate the exact adapter pair and scales on held-out tasks.

- **`2× Q8_0` separate core (`374.66 MiB`) alone:** `7,666 MiB` used / `120 MiB` free. Pre-loading even **1 dormant `r=16` SME (`~61 MiB`)** leaves `58 MiB` at boot (`7,728 MiB`), which **OOMs on `cudaGraphInstantiate`** because toggling 1 SME across `2` separate core LoRAs requires **`94 MiB`** of CUDA graph capture allocations.
- **`2× Q5_0` separate core (`242.51 MiB`):** Passes **1 SME** (`188 MiB` boot free → `94 MiB` free post-toggle, `94 MiB` graph delta), **OOMs on the 3rd CUDA graph** with **2 SMEs** (`126 → 70 → 28 → OOM`), and regresses stacked `2K` GeoNodes (`3/6`).
- **`maximin-r32` merged `Q8_0` core (`213.21 MiB`, recommended):** Merging `2` core LoRAs → `1` drops CUDA graph overhead (`94 → 72 MiB` for 1 SME, `104 MiB` for 2 SMEs) and passes both multi-SME headroom and multi-step coherence (`5/6` stacked `2K` GeoNodes):
  - **Core + 1 dormant `r=16` SME (`60.9 MiB`):** Boots with `230 MiB` free → active `184 MiB` free → deactivated **`158 MiB` free** (`72 MiB` graph delta, `0.39–0.44 ms` swap).
  - **Core + 2 dormant `r=16` SMEs (`121.8 MiB`):** Boots with `168 MiB` free → SME #1 `122 MiB` → SME #2 `92 MiB` → both off **`64 MiB` free** (`104 MiB` graph delta across all 3 graph topologies, `0.37–0.39 ms` swaps).
- **`hybrid-64` merged `Q8_0` core (`257.97 MiB`):** Passes **1 SME** cleanly (`184 → 138 → 112 MiB` free, `72 MiB` graph delta), while **2 simultaneous SMEs** run tightly (`122 → 18 MiB` free, below the `>30 MiB` Gate E margin).
- **Pre-loading `3+` dormant `r=16` SMEs simultaneously** at `ctx=64000` still requires either (1) compressing specialists to `r=8` (`~31 MiB` via `optimize_lora.py --rank 8 --verify`), (2) lowering `--ctx-size` (e.g. `49152` or `32768`), or (3) grouping specialists into separate `llama-swap` model profiles.

---

## 7. Adapting This Recipe to Any New Verifiable Domain

To train a new hot-swappable SME LoRA (`SQL`, `CadQuery`, `Manim`, `Lean4`, `Three.js`, shader GLSL, etc.):
1. Instantiate the **4-Layer SME Mini-Framework (Section 4)** for the target domain (`schema_catalog.json` + flat canonical boundary contract + `<think>` IR convention + structural verifier `<domain>_verifier.py` returning `VerifyResult(syntax_ok, exec_ok, reward, scene_summary, error)` and strict `is_<domain>_pass(vr)`).
2. Build a `60–150`-row single-execution-model SFT JSONL (`system`, `prompt`, `response[, reasoning]`) + an unseen `rl_tasks.jsonl` + held-out eval suite (`0` lexical/concept overlap).
3. Run **Stage 1 (`train_lora.py`)** → verify Stage 1 standalone → run **Stage 2 (`online_rl_blender.py` with your verifier, `--algo dppo --ref-anchor sft`)** → verify **Gates A–E (Section 6)**.
4. Register the new `:0.0` `Q8_0` GGUF in `qwen27b`'s `--lora-scaled` list and toggle it on demand with `../.pi/agent/skills/lora-router/scripts/lora_ctl.py`.
