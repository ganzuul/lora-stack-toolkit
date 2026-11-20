In `llama.cpp`, loaded LoRA adapters contribute additively to the base model
weights $W_0$ during graph evaluation:

$$W_{\text{eff}} = W_0 + \sum_{i} s_i \cdot \left(\frac{\alpha_i}{r_i}\right) B_i A_i$$

Here, $s_i$ is the runtime `scale` passed through `--lora-scaled` or
`POST /lora-adapters`. The base weights are not rewritten when scales change;
setting one adapter to `0.0` removes that adapter's contribution. This does not
guarantee that a different scale or stack has better output quality.

The numerical update is additive, but behavior depends on the adapter weights,
their training bases, overlap, scales, model, and inference runtime. Treat scale
vectors as experimental settings to evaluate—not probabilities or a universal
measure of update magnitude.

---

### 1. Exclusive `0/1` Switching (`[1.0, 0.0, 0.0]` $\rightarrow$ `[0.0, 1.0, 0.0]`)

* **Weight behavior:** Only the selected adapter contributes, at the selected scale; the base weights remain unchanged.
* **Quality behavior:** Switching is not a quality guarantee. Each adapter should be validated against its intended base model and evaluated independently.
* **Runtime behavior:** KV-cache handling after a scale update depends on the llama.cpp fork/version and request path. Verify it in the target runtime instead of assuming cache invalidation or reuse behavior.

### 2. Additive `1/1` Stacking (`[1.0, 1.0, 0.0]`)

Multiple adapters at `1.0` add their full configured deltas. This can work, but
does not imply that the combination will improve behavior:

* Updates can reinforce or cancel. The result depends on the actual tensor deltas, not just scale values.
* Overlapping modules can interact; independent training does not guarantee independent behavior when stacked.
* A larger sum of scales does not prove a larger Frobenius norm, nor does a smaller sum guarantee a safer or higher-quality stack.

---

### 3. How to Scale a Stack Properly

For the currently wired **merged core** plus one domain specialist, the values
below are candidate starting points for an evaluation sweep, not universal
recommendations:

| Goal | Candidate Scale Vector | Interpretation |
| --- | --- | --- |
| **Core + light specialist influence** | `[0.8, 0.35, 0.0]` | Current router default; a conservative initial experiment. |
| **Equal numeric scales** | `[0.5, 0.5, 0.0]` | A point to test; equal scales do not imply equal influence. |
| **Specialist-forward experiment** | `[0.4, 0.75, 0.0]` | A more specialist-weighted point to test against general behavior. |
| **Three-adapter experiment** | `[0.5, 0.3, 0.3]` | Only use after evaluating the exact three-way stack. |

**Tuning procedure:** Hold one adapter's scale fixed, sweep the specialist scale,
and compare held-out domain tasks as well as general instruction/tool-use
behavior. Choose based on measured results. Do not infer safety or quality from
the sum of scales.

For the live `qwen27b` router, the controller now uses the conservative starting point `[0.8, 0.35, 0.0]`: merged core at `0.8`, selected specialist at `0.35`, and the other specialist at `0.0`.

```bash
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py load geonodes
# or
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py load blenderworld

# Restore the core-only baseline: [1.0, 0.0, 0.0]
~/.pi/agent/skills/lora-router/scripts/lora_ctl.py unload
```

`load <specialist> --scale X --core-scale Y` allows deliberate tuning. The
launch-time configuration remains core-only (`1.0`) with specialists dormant
(`0.0`); unloading the last specialist restores that baseline.

---

### 4. Operational Gotchas in Your Command

* **Omitted IDs default to `0.0` in `POST /lora-adapters`:** When you send a `POST /lora-adapters` request, any loaded adapter ID **omitted** from the JSON array is automatically disabled (`scale: 0.0`). To stack adapter `0` and adapter `1`, you must send both explicitly:
```bash
curl -X POST http://127.0.0.1:8080/lora-adapters \
  -H "Content-Type: application/json" \
  -d '[{"id": 0, "scale": 0.75}, {"id": 1, "scale": 0.45}]'
```

* **Per-request overrides:** Instead of mutating the global server state with `POST /lora-adapters`, you can pass a `"lora": [{"id": 0, "scale": 0.75}, {"id": 1, "scale": 0.45}]` array directly inside a `/completion` request body to override scales for that single call.
* **Basis mismatch check:** Your command lists `Qwen3.8-27B-core-pi-qwopus-maximin-r32-Q8_0.gguf` alongside `Bonsai-2-27B-blender-*`. Make sure all three adapters were trained or rotated into the **same coordinate basis** as the base model you are loading (standard Qwen3.8 vs. Ternary Bonsai 2's Hadamard-rotated basis). If one adapter is in the wrong basis, scaling it above `0.0` will inject structured noise rather than useful signal regardless of the scale factor.