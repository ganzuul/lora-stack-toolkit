# Selecting LoRA Stack Scales

LoRA adapters add weighted deltas to the base model:

$$W_{\mathrm{eff}} = W_0 + \sum_i s_i \left(\frac{\alpha_i}{r_i}\right) B_i A_i$$

Here, `s_i` is the runtime scale. Setting a scale to `0.0` removes that
adapter's contribution without rewriting the base weights. Stacking adapters
trained independently against the same base changes the effective update and
can cause interference or activation drift. A scale vector is not a probability
distribution: the sum of its scales is not, by itself, a meaningful bound on
the resulting weight or behavior change.

## Starting points

The recipe's general guidance for core + one domain specialist is to start
with a damped core and specialist rather than assuming `1.0 + 1.0` is safe:

| Intended behavior | Core | Specialist | Other specialists |
|---|---:|---:|---:|
| Core reasoning with light domain influence | 0.80 | 0.35 | 0.00 |
| Balanced initial experiment | 0.50 | 0.50 | 0.00 |
| Domain-heavy with core guardrails | 0.40 | 0.75 | 0.00 |

The live router's default is the first row. You can tune explicitly:

```bash
python3 .pi/agent/skills/lora-router/scripts/lora_ctl.py load geonodes --core-scale 0.75 --scale 0.50
```

`load` enables only the selected specialist unless `--stack` is given.
`unload` turns off every specialist and restores core scale `1.0`. Runtime
changes do not alter the launch-time configuration.

## Tuning and validation

Treat these values as starting hypotheses. Sweep the specialist scale in small
increments while holding the core fixed around `0.7`–`0.8`; compare held-out
domain tasks and general instruction/tool-use behavior. Stop when domain
performance is sufficient rather than maximizing scale. If changing multiple
adapters, evaluate the exact combination: overlapping updates can interfere,
and empirical orthogonality for one adapter pair does not prove that another
stack is safe.

The detailed recipe includes historical adapter-specific evaluations and
resource measurements. Those results are not a guarantee for newly trained
adapters, model versions, or hardware. In particular, the controller's default
scales were chosen as a conservative starting point; the controller tests
verify scale transitions, not model-output quality.

## Router safety

`POST /lora-adapters` applies the complete active scale set. Always include all
preloaded adapter IDs in manual requests; omitted IDs may be disabled. Prefer
the bundled
`.pi/agent/skills/lora-router/scripts/lora_ctl.py`, which reads the current
adapter list, sends the complete payload, verifies the returned scales, and
refuses to touch a model that is not already running in llama-swap.

The controller recognizes the merged core by filename patterns such as
`qwopus`. If the core adapter is named differently, set `LORA_CORE_PATTERNS`
or pass `--core-pattern` so core scales are protected and restored correctly.
