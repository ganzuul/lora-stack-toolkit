---
name: lora-router
description: "Hot-load and unload specialist Q8_0 LoRA adapters on local llama-server (via llama-swap POST /upstream/<model>/lora-adapters) in <1 ms while preserving the merged pi-lora + qwopus-r32 + maximin core stack. Use when entering or leaving a specialized domain task (e.g. Blender Geometry Nodes, BlenderWorld bpy scripts, or other pre-loaded domain LoRAs) on qwen27b, or when inspecting/resetting active LoRA scales."
disable-model-invocation: false
---

# `lora-router`: Dynamic Specialist LoRA Hot-Swapping

Install this skill from the repository with the instructions in `README.md`.
The examples assume `$LORA_STACK_REPO` points to the repository root and that
the skill package is installed under `~/.pi/agent/skills/lora-router/`.
Use `~/.pi/agent/skills/lora-router/scripts/lora_ctl.py` to toggle domain
specialist adapters on `qwen27b` (`Ternary-Bonsai-2-27B-PTQ1_0` + merged core)
at runtime in **`<1 ms`** with **zero model reload**.

---

## 1. When to Load vs. When to Unload

**Always check what is loaded first:**
```bash
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py list
```

- **Load a specialist** immediately before generating code or answers inside a pre-loaded domain you have a specialist for (for example: Geometry Nodes `Blender 5.4` scripts, `BlenderWorld` scene scripts, etc.) when `qwen27b` is the active running model (`list` shows the matching dormant adapter at `scale: 0.00`).
- **Unload immediately afterward:** As soon as the domain subtask finishes — and before switching to general coding, planning, shell ops, chat, or a different specialist — run `~/.pi/agent/skills/lora-router/scripts/lora_ctl.py unload` before returning to core behavior.
- **Use damped core + specialist scales by default:** `load <name>` defaults to specialist `0.35` and core `0.8`, while zeroing other specialists. Tune with `--scale` and `--core-scale`; `unload` restores core `1.0` and all specialists `0.0`. Do not stack competing same-domain Blender specialists; `--stack` is an explicit opt-in.
- **Do not accidentally trigger llama-swap eviction:** `lora_ctl.py` checks `GET http://127.0.0.1:8080/running` first and refuses (exit code `2`) if `--model` (`qwen27b` / `$PI_MODEL`) is not currently loaded, so calling it during an `occamy` or remote session never evicts the GPU.

---

## 2. Quick Commands

```bash
# 1. Inspect pre-loaded adapters, roles (core vs specialist), and current scales
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py list

# 2. Hot-load a single specialist by alias (geonodes, blenderworld, blender),
# ID, or filename substring; defaults to core 0.8 + specialist 0.35
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py load geonodes
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py load c4-gn --scale 0.5 --core-scale 0.75
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py load sft-v2

# 3. Unload all non-core specialists and restore core to 1.0
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py unload

# 4. Unload only one specific specialist
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py unload geonodes
```

Useful flags: `--model <id>` (defaults to `$PI_MODEL` or `qwen27b`), `--json` (machine-readable output), `--direct http://127.0.0.1:18089` (bypass `llama-swap` when hitting a standalone `llama-server`), `--autoload` (explicitly allow `llama-swap` to swap `--model` in).

---

## 3. Critical `POST /lora-adapters` Footgun (Read-Modify-Write Rule)

1. **`POST /lora-adapters` replaces the active list for ALL pre-loaded adapters:** Any loaded adapter ID **omitted** from the JSON array is **silently set to `scale: 0.0`**. If you send only `[{"id": 2, "scale": 1.0}]`, `llama-server` disables `pi-lora` (`id=0`) and `qwopus-r32` (`id=1`). Always `GET` first and send every adapter ID in the `POST` body (which `lora_ctl.py` automates).
2. **Pre-load requirement:** `llama-server` can only toggle adapters that were registered at server launch (`--lora-scaled ...:0.0`). It cannot open new `.gguf` files from disk without restarting `qwen27b`.

---

## 4. Raw `curl` Fallback

If `lora_ctl.py` is unavailable, always Read-Modify-Write the full array:

```bash
# Inspect exact IDs and current scales on qwen27b (0 = merged core, 1 = geonodes, 2 = blenderworld)
curl -s http://127.0.0.1:8080/upstream/qwen27b/lora-adapters | jq .

# Enable specialist id=1 (geonodes) at the conservative [0.8, 0.35, 0.0] starting point
curl -s -X POST http://127.0.0.1:8080/upstream/qwen27b/lora-adapters \
  -H 'Content-Type: application/json' \
  -d '[{"id":0,"scale":0.8},{"id":1,"scale":0.35},{"id":2,"scale":0.0}]'

# Unload all specialists and restore the core-only baseline
curl -s -X POST http://127.0.0.1:8080/upstream/qwen27b/lora-adapters \
  -H 'Content-Type: application/json' \
  -d '[{"id":0,"scale":1.0},{"id":1,"scale":0.0},{"id":2,"scale":0.0}]'
```

*(Never run raw `/upstream/qwen27b/...` `curl` commands without checking `curl -s http://127.0.0.1:8080/running` first.)*

---

## 5. Live Wired Configuration & Operator Setup (`llama-swap` `config.yaml`)

`qwen27b` in `/home/nos/.config/llama-swap/config.yaml` (`--ctx-size 64000`, `q4_0` KV, `--kv-offload`, `--spec-type ngram-map-k`) wires the merged FP32-`maximin`-`r=32` `Q8_0` core plus two dormant (`:0.0`) Blender SMEs:

| ID | Role | Selector / Alias | File |
|---:|---|---|---|
| `0` | `core` (`1.00`, protected) | `qwopus`, `maximin` | `/home/nos/models/Qwen3.8-27B-core-pi-qwopus-maximin-r32-Q8_0.gguf` (`213.21 MiB`) |
| `1` | `specialist` (`0.00` dormant) | `geonodes`, `c7-gn-sft` | `/home/nos/models/Bonsai-2-27B-blender-c7-gn-sft-Q8_0.gguf` (`60.90 MiB`) |
| `2` | `specialist` (`0.00` dormant) | `blenderworld`, `sft-v2` | `/home/nos/models/Bonsai-2-27B-blender-c5-unified-sft-v2-Q8_0.gguf` (`60.90 MiB`) |

```yaml
--lora-scaled /home/nos/models/Qwen3.8-27B-core-pi-qwopus-maximin-r32-Q8_0.gguf:1.0,/home/nos/models/Bonsai-2-27B-blender-c7-gn-sft-Q8_0.gguf:0.0,/home/nos/models/Bonsai-2-27B-blender-c5-unified-sft-v2-Q8_0.gguf:0.0
```

**Measured 8 GiB VRAM Headroom (`RTX 2070 SUPER`, `7,786 MiB` CUDA-usable):**
- Boot (`core` only, 2 dormant `r=16` SMEs): **`7,632 MiB` used / `154 MiB` free**.
- After loading `geonodes` → switching to `blenderworld` → unloading all (`3` CUDA graph topologies captured): **`7,696 MiB` used / `90 MiB` free** (`0.42–0.81 ms` swaps).
- Adding **3+ dormant `r=16` SMEs** at `--ctx-size 64000` would tighten that `90 MiB` headroom; either shrink extra dormant SMEs to `r=8` (`~31 MiB` via `optimize_lora.py --rank 8 --verify`), lower `--ctx-size`, or group additional SMEs into a separate `llama-swap` model alias.

---

## 6. Blender 5.4 Runtime & Schema Quick Reference (`geonodes` / `blenderworld`)

When running headless Blender tasks after loading `geonodes` or `blenderworld`:
- **Headless binary:** `$BLENDER_BIN --background --factory-startup --python <script.py>`
  *(Always `write` a `.py` file first and run with `--python`; do not search `PATH` or rely on `--python-expr`, whose stdout can be suppressed.)*
- **Live Layer-1 node & socket catalog:** `$LORA_STACK_REPO/scripts/distill-lora/work/blender_5_4_gn_schema.json`
  *(Grep this JSON directly for exact `GeometryNode*` / `ShaderNode*` / `FunctionNode*` IDs and `inputs`/`outputs` pin names instead of writing ad-hoc Blender probe scripts.)*
- **Structural verifier helper:** `$LORA_STACK_REPO/scripts/distill-lora/blender_verifier.py`
  *(To inspect an existing `.blend` scene in headless Blender 5.4, open it via `bpy.ops.wm.open_mainfile(filepath=...)`, evaluate via `deps = bpy.context.evaluated_depsgraph_get()`, and extract evaluated meshes via `eval_obj = obj.evaluated_get(deps); mesh = eval_obj.to_mesh()`.)*

### Geometry Nodes verification loop (mandatory)

1. `write` the script to a `.py` file (never `blender --python -` / heredoc: Blender cannot read a script from stdin).
2. Verify with **`~/.pi/agent/skills/lora-router/scripts/gn_check.py <script.py>`** instead of raw `blender`.
   - Exit 0 and a line starting `VERIFIED:` means the tree has a linked `NodeGroupOutput`, valid links, a bound `NODES` modifier and evaluated geometry.
   - Any line starting `Error:` is a failure, including Blender `WARNING`s (e.g. "Node group must have a group output node"). `Saved as ...` is NOT success.
3. Fix with a targeted `edit`, then re-run `gn_check.py`. If the same `Error:` appears twice, change approach; do not repeat the same command.
4. Every tree needs `ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')` and a `NodeGroupOutput` node linked from the final geometry.
5. When finished, run `~/.pi/agent/skills/lora-router/scripts/lora_ctl.py unload geonodes` before replying.

$ARGUMENTS
