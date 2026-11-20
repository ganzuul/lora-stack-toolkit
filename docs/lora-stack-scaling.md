In `llama.cpp`, loaded LoRA adapters are applied **additively** to the base model weights $W_0$ during graph evaluation:

$$W_{\text{eff}} = W_0 + \sum_{i} s_i \cdot \left(\frac{\alpha_i}{r_i}\right) B_i A_i$$

Here, $s_i$ is the runtime `scale` you pass via `--lora-scaled` or `POST /lora-adapters`. Because the base weights $W_0$ are never mutated in memory, changing $s_i$ is mathematically lossless—setting an adapter to `0.0` completely removes its contribution ($s_i = 0 \implies W_{\text{eff}} = W_0$).

What treating `scale` like a `0/1` switch actually does to model behavior depends on whether you are **swapping exclusively** or **stacking simultaneously**.

---

### 1. Exclusive `0/1` Switching (`[1.0, 0.0, 0.0]` $\rightarrow$ `[0.0, 1.0, 0.0]`)

* **Behavior:** Clean and lossless. Each adapter sees the exact base weight manifold $W_0$ it was trained on.
* **Side effect:** Every time you change the active scale(s), `llama-server` must **invalidate and recompute the KV cache**, because previously cached $K$ and $V$ projections were computed using the old $W_{\text{eff}}$. If you are hot-swapping mid-conversation, expect a full prompt-reprocessing pause on the next turn.

### 2. Additive `1/1` Stacking (`[1.0, 1.0, 0.0]`)

Treating multiple adapters as independent `0/1` switches and turning two or more to `1.0` simultaneously is where behavior degrades:

* **Norm Inflation & Activation Drift:** Each adapter was trained assuming $W_0$ as its baseline, not $W_0 + \Delta W_1$. Summing $\Delta W_1 + \Delta W_2$ at full `1.0` scale increases the Frobenius norm of the weight update and pushes intermediate layer activations out of distribution.
* **Subspace Interference:** If your core adapter (`r32`) and a domain SFT adapter both modify the same attention heads or FFN projections in conflicting directions, they cancel or distort each other—typically showing up as formatting collapse, repetition loops, or loss of tool-calling discipline.

---

### 3. How to Scale a Stack Properly

When combining a **general/core adapter** (e.g., `core-pi-qwopus-maximin-r32`) with a **narrow domain SFT adapter** (e.g., `blender-c7-gn-sft`), use damped, sub-unit scaling rather than `1.0 + 1.0`:

| Goal | Starting Scale Vector | Why It Works |
| --- | --- | --- |
| **Core reasoning + light domain knowledge** | `[0.8, 0.35, 0.0]` | Keeps the core reasoning/formatting manifold dominant while injecting domain vocabulary and Geometry Nodes patterns without blowing up activation norms. |
| **Equal blend of two adapters** | `[0.5, 0.5, 0.0]` | Convex combination ($\sum s_i = 1.0$); preserves overall perturbation magnitude. |
| **Domain-heavy with core guardrails** | `[0.4, 0.75, 0.0]` | Lets the Blender SFT drive syntax/API calls while the core adapter stabilizes long-horizon coherence. |
| **Three-way stack** | `[0.5, 0.3, 0.3]` | Keep the total sum of scales roughly in the **1.0 to 1.2** range. |

**Tuning heuristic:** Lock your core adapter at `0.7`–`0.8` and sweep the domain adapter from `0.25` upward in `0.15` steps. Stop as soon as the domain-specific knowledge activates cleanly; pushing further usually degrades general instruction following before it improves domain recall.

For the live `qwen27b` router, the controller now uses the conservative starting point `[0.8, 0.35, 0.0]`: merged core at `0.8`, selected specialist at `0.35`, and the other specialist at `0.0`.

```bash
/home/nos/.pi/agent/skills/lora-router/scripts/lora_ctl.py load geonodes
# or
/home/nos/.pi/agent/skills/lora-router/scripts/lora_ctl.py load blenderworld

# Restore the core-only baseline: [1.0, 0.0, 0.0]
/home/nos/.pi/agent/skills/lora-router/scripts/lora_ctl.py unload
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