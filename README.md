# LoRA Stack Toolkit

Curated training and runtime-routing materials for Q8_0 LoRA adapters on
Ternary-Bonsai-2-27B. This repository brings together the training recipe,
the core training/export code, and the llama.cpp/llama-swap hot-swap router.

## What is included

- [Training recipe](docs/lora-training-recipe-spec.md): end-to-end specialist
  design, data/evaluation gates, training workflow, adapter merging, and the
  measured deployment budget.
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
  utilities in `scripts/`.

Model weights, datasets, trained adapters, caches, and benchmark outputs are
intentionally not included.

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
scripts/
  distill-lora/
    train_lora.py
    bonsai_ptq_model.py
  lora_gguf_common.py
  merge_loras.py
  optimize_lora.py
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
