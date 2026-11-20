# LoRA Stack Toolkit

Curated training and runtime-routing materials for Q8_0 LoRA adapters on
Ternary-Bonsai-2-27B. This repository brings together the training recipe,
the core training/export code, and the llama.cpp/llama-swap hot-swap router.

## What is included

- [Training recipe](docs/lora-training-recipe-spec.md): end-to-end specialist
  design, data/evaluation gates, training workflow, adapter merging, and the
  measured deployment budget.
- [`distill-lora` Pi skill](.pi/agent/skills/distill-lora/SKILL.md) and its
  source workflows for data preparation, SFT, verifiable RL, evaluation, and
  dataset construction.
- [Scale selection summary](docs/scale-selection.md): current conservative
  defaults, tuning advice, and validation caveats.
- [Full scaling note](docs/lora-stack-scaling.md): the complete reference
  discussion of exclusive switching, stacking, and operational caveats.
- [`.pi` router skill](.pi/agent/skills/lora-router/SKILL.md), including the
  [controller](.pi/agent/skills/lora-router/scripts/lora_ctl.py) and
  [Geometry Nodes verifier](.pi/agent/skills/lora-router/scripts/gn_check.py):
  safe model-state checks, selector aliases, read-modify-write updates, and
  verification.
- Training and export implementation: `scripts/distill-lora/` plus the shared
  utilities in `scripts/`, including the SVD extractor.
- Blender 5.4 schema snapshot and small GeoNodes eval/RL prompt suites under
  `scripts/distill-lora/work/`.
- The shared Qwen chat template at `scripts/qwen_chat_template.jinja`.

Model weights, datasets, trained adapters, caches, and benchmark outputs are
intentionally not included.

The bundled source workflows are complete, but execution still requires
external assets and tools: the Ternary-Bonsai model checkpoint, HF datasets
used by data preparation, trained PEFT/GGUF adapters, a compatible Blender
binary, llama.cpp/Bonsai `gguf-py` and server binaries, and the Python/CUDA
packages described by the recipe. The checked-in schema is a Blender 5.4.0
Alpha snapshot; regenerate it with `scripts/distill-lora/blender_schema_5_4.py`
when using a different Blender version.

Set these variables for a non-default machine as needed:

```bash
export MODEL_DIR=/path/to/model-files
export BASE_GGUF=/path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
export LORA_STACK_REPO="$PWD"
export BLENDER_BIN=/path/to/blender
export LLAMA_SERVER_BIN=/path/to/llama-server
export QWEN_CHAT_TEMPLATE="$PWD/scripts/qwen_chat_template.jinja"
export BONSAI_GGUF_PY=/path/to/llama.cpp/gguf-py
```

Install the included Pi skills with:

```bash
mkdir -p ~/.pi/agent/skills
cp -a .pi/agent/skills/lora-router .pi/agent/skills/distill-lora ~/.pi/agent/skills/
```

## Important: starting scales

For the currently wired adapter order `[merged core, GeoNodes, BlenderWorld]`,
the router defaults are:

| Use | Core | GeoNodes | BlenderWorld |
|---|---:|---:|---:|
| Idle/core only | 1.00 | 0.00 | 0.00 |
| GeoNodes task | 0.80 | 0.35 | 0.00 |
| BlenderWorld task | 0.80 | 0.00 | 0.35 |

These are conservative **starting heuristics**, not universal optimal scales or
a claim that the exact defaults were quality-benchmarked by this toolkit
change. LoRA deltas add to the model weights; scales do not behave like
probabilities, and their sum is not a reliable measure of the combined update.
Evaluate held-out domain quality and general instruction-following behavior
when tuning. Competing Blender specialists should remain exclusive unless
stacking is intentional and separately evaluated.

The controller's `load` defaults implement the specialist rows above. `unload`
restores the idle row. `--scale` and `--core-scale` allow an explicit sweep.
The launch configuration should keep specialists dormant (`:0.0`); runtime
loads are separate state and must be unloaded when the specialist task ends.

Run the router's hardware-independent tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Repository layout

```text
docs/
  lora-training-recipe-spec.md
  lora-stack-scaling.md
  scale-selection.md
.pi/agent/skills/lora-router/
  SKILL.md
  scripts/
    lora_ctl.py
    gn_check.py
.pi/agent/skills/distill-lora/
  SKILL.md
scripts/
  distill-lora/
    train_lora.py
    bonsai_ptq_model.py
    online_rl_blender.py
    prepare_blender_data.py
    blender_verifier.py
    blender_schema_5_4.py
    benchmark_*.py
    build_*_dataset.py
    work/
      blender_5_4_gn_schema.json
      geonodes_eval_suite.json
      geonodes_rl_tasks.jsonl
  qwen_chat_template.jinja
  lora_gguf_common.py
  merge_loras.py
  optimize_lora.py
  extract_svd_lora.py
tests/
  test_lora_ctl.py
```

## Environment notes

The recipe and scripts were developed for an 8 GiB RTX 2070 SUPER with a
specific Ternary-Bonsai-2 checkpoint, CUDA/PyTorch/Triton stack, tokenizer, and
local llama.cpp fork. The recipe contains machine-specific absolute paths and
runtime assumptions; adapt those paths and verify the model/basis and GPU setup
before using it elsewhere. The training scripts expect the layout shown above
and require Python dependencies described by the recipe and source imports.

The router defaults to llama-swap at `http://127.0.0.1:8080` and model ID
`qwen27b`. It refuses to modify the server unless that model is already
running. Use `--direct` only when intentionally targeting a standalone
llama-server.

## Quick router commands

```bash
python3 .pi/agent/skills/lora-router/scripts/lora_ctl.py list
python3 .pi/agent/skills/lora-router/scripts/lora_ctl.py load geonodes
python3 .pi/agent/skills/lora-router/scripts/lora_ctl.py load blenderworld
python3 .pi/agent/skills/lora-router/scripts/lora_ctl.py unload
```
