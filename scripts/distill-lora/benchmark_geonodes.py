#!/usr/bin/env python3
"""Benchmark Geometry Nodes (GeometryNodeTree) execution & graph metrics on held-out tasks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_blenderworld as bbw  # noqa: E402
import blender_verifier as bv  # noqa: E402

DEFAULT_EVAL_SUITE = SCRIPT_DIR / "work" / "geonodes_eval_suite.json"
DEFAULT_RL_TASKS = SCRIPT_DIR / "work" / "geonodes_rl_tasks.jsonl"

_SUFFIX = (
    "\nWrite a clean, standalone Blender 5.4 Python (`bpy`) script using Geometry Nodes "
    "(`GeometryNodeTree` with `ng.interface.new_socket`, `ng.nodes.new`, `ng.links.new`, and `NodeGroupOutput`) "
    "to build this procedural object from scratch, attach a `NODES` modifier or apply cleanly while keeping "
    "the node group in `bpy.data.node_groups`, assign `ShaderNodeBsdfPrincipled` materials, add a SUN light, "
    "active CAMERA, Cycles (64 samples), and save `/workspace/exports/final.blend`."
)

HELD_OUT_EVAL_TASKS = [
    {
        "id": "eval_gn_01_globular_cactus",
        "prompt": "Build a ribbed globular barrel cactus with radial spine ridges on a small terracotta pot base procedurally." + _SUFFIX,
    },
    {
        "id": "eval_gn_02_apple",
        "prompt": "Build an organic sculpted apple fruit body with subtle top/bottom dimples and a short curved stem swept along a Bezier spline." + _SUFFIX,
    },
    {
        "id": "eval_gn_03_spiral_stair",
        "prompt": "Build a helical spiral staircase with curved cylindrical tubular handrails swept along a 3D curve profile." + _SUFFIX,
    },
    {
        "id": "eval_gn_04_fluted_vase",
        "prompt": "Build a fluted ceramic vase swept from a curved Bezier profile and modulated with radial sinusoidal ribs using Set Position." + _SUFFIX,
    },
    {
        "id": "eval_gn_05_helical_spring",
        "prompt": "Build a brushed steel helical coil spring swept cleanly using Curve Spiral, Curve Circle, and Curve to Mesh." + _SUFFIX,
    },
    {
        "id": "eval_gn_06_wave_panel",
        "prompt": "Build a procedural architectural wave-relief wall panel from a high-resolution Mesh Grid displaced smoothly via Set Position and sine math." + _SUFFIX,
    },
]

RL_TRAIN_TASKS = [
    {
        "id": "rl_gn_01_moss_tuft",
        "prompt": "Build a cluster of arched tapered moss strands swept from Bezier curve segments and green Principled BSDF materials." + _SUFFIX,
    },
    {
        "id": "rl_gn_02_twisted_torus",
        "prompt": "Build a sculpted twisted ornamental ring torus with radial surface ridges using GeometryNodeCurvePrimitiveCircle, SetPosition, and CurveToMesh." + _SUFFIX,
    },
    {
        "id": "rl_gn_03_scalloped_bowl",
        "prompt": "Build a glazed ceramic scalloped decorative bowl with gentle wavy rim ribs displaced via GeometryNodeSetPosition." + _SUFFIX,
    },
    {
        "id": "rl_gn_04_spiky_urchin",
        "prompt": "Build a geometric sea-urchin sphere with tapered cone spines distributed across faces via Distribute Points on Faces and Instance on Points." + _SUFFIX,
    },
    {
        "id": "rl_gn_05_bamboo_stalk",
        "prompt": "Build a segmented bamboo stalk with subtle joint bulges swept along a curved stem spline." + _SUFFIX,
    },
    {
        "id": "rl_gn_06_corrugated_canopy",
        "prompt": "Build an arched corrugated pavilion canopy roof panel starting from a subdivided mesh grid extruded and displaced by sine waves." + _SUFFIX,
    },
    {
        "id": "rl_gn_07_coiled_rope",
        "prompt": "Build a coiled natural fiber rope helix using Curve Spiral swept with a circular profile curve." + _SUFFIX,
    },
    {
        "id": "rl_gn_08_pine_bough",
        "prompt": "Build a curved conifer twig with tapered pine needles instanced along points of a central Bezier stem." + _SUFFIX,
    },
]


def ensure_suites() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    DEFAULT_EVAL_SUITE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_EVAL_SUITE.write_text(json.dumps(HELD_OUT_EVAL_TASKS, indent=2) + "\n", encoding="utf-8")
    DEFAULT_RL_TASKS.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in RL_TRAIN_TASKS) + "\n",
        encoding="utf-8",
    )
    return HELD_OUT_EVAL_TASKS, RL_TRAIN_TASKS


def _chat_complete_meta(
    api_base: str,
    model_name: str,
    prompt_text: str,
    max_tokens: int = 4096,
    temperature: float = 0.4,
    system_prompt: str = (
        "You are an expert Blender 5.4 Python programmer. Generate clean, self-contained, executable `bpy` code."
    ),
) -> tuple[str, str, str]:
    """Query an OpenAI-compatible /chat/completions endpoint and return (content, reasoning, finish_reason)."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
    }
    url = api_base.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    choice = res["choices"][0]
    msg = choice["message"]
    content = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning_content") or msg.get("reasoning") or "")
    finish_reason = str(choice.get("finish_reason") or "")
    return content, reasoning, finish_reason


def run_benchmark(
    model_gguf: Path,
    lora_specs: list[str],
    out_json: Path,
    export_dir: Path | None = None,
    max_tokens: int = 4096,
    reasoning_budget: int = 0,
    temperature: float = 0.4,
    api_base: str | None = None,
    api_model: str = "local",
    reasoning_budget_message: str | None = bbw.DEFAULT_REASONING_BUDGET_MESSAGE,
) -> dict[str, Any]:
    """Evaluate model + LoRA(s) on the locked 6-task Geometry Nodes benchmark suite."""
    eval_tasks, _ = ensure_suites()
    proc = None
    endpoint_base = api_base
    if endpoint_base is None:
        proc = bbw.start_llama_server(
            model_gguf,
            lora_specs,
            reasoning_budget=reasoning_budget,
            reasoning_budget_message=reasoning_budget_message,
        )
        endpoint_base = "http://127.0.0.1:18089/v1"
        endpoint_model = "local"
    else:
        endpoint_model = api_model

    rows: list[dict[str, Any]] = []
    try:
        for item in eval_tasks:
            tid = item["id"]
            prompt = item["prompt"]
            t0 = time.perf_counter()
            content, reasoning, finish_reason = _chat_complete_meta(
                endpoint_base,
                endpoint_model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            dt_gen = time.perf_counter() - t0
            save_blend = (export_dir / f"{tid}.blend") if export_dir else None
            vr = bv.verify_script(content, task_bundle=None, save_blend_path=save_blend)
            gn_pass = bv.is_geonodes_pass(vr)
            ss = vr.scene_summary or {}
            if export_dir and gn_pass:
                export_dir.mkdir(parents=True, exist_ok=True)
                (export_dir / f"{tid}.py").write_text(bv.extract_python_code(content) + "\n", encoding="utf-8")
            elif save_blend and save_blend.is_file() and not gn_pass:
                save_blend.unlink()

            rows.append({
                "task_id": tid,
                "gen_sec": round(dt_gen, 2),
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
                "finish_reason": finish_reason,
                "truncated": finish_reason == "length",
                "syntax_ok": vr.syntax_ok,
                "exec_ok": vr.exec_ok,
                "gn_pass": gn_pass,
                "geonode_trees_count": ss.get("geonode_trees_count", 0),
                "geonode_nodes_total": ss.get("geonode_nodes_total", 0),
                "geonode_links_total": ss.get("geonode_links_total", 0),
                "total_verts": ss.get("total_verts", 0),
                "mesh_count": ss.get("mesh_count", 0),
                "reward": vr.reward,
                "error": vr.error[:600],
            })
            print(
                f"[{tid}] syntax={vr.syntax_ok} exec={vr.exec_ok} gn_pass={gn_pass} "
                f"gn_trees={ss.get('geonode_trees_count', 0)} gn_nodes={ss.get('geonode_nodes_total', 0)} "
                f"verts={ss.get('total_verts', 0)} reward={vr.reward:.3f} finish={finish_reason} "
                f"r_chars={len(reasoning)} ({dt_gen:.1f}s)",
                flush=True,
            )
    finally:
        if proc is not None:
            bbw.stop_llama_server(proc)

    n = max(1, len(rows))
    summary = {
        "model": endpoint_model if api_base else str(model_gguf),
        "api_base": api_base,
        "loras": lora_specs,
        "reasoning_budget": reasoning_budget,
        "max_tokens": max_tokens,
        "tasks": len(rows),
        "syntax_rate": round(sum(1 for r in rows if r["syntax_ok"]) / n, 4),
        "exec_rate": round(sum(1 for r in rows if r["exec_ok"]) / n, 4),
        "gn_pass_rate": round(sum(1 for r in rows if r["gn_pass"]) / n, 4),
        "mean_gn_nodes": round(sum(r["geonode_nodes_total"] for r in rows) / n, 2),
        "mean_verts": round(sum(r["total_verts"] for r in rows) / n, 1),
        "mean_reward": round(sum(r["reward"] for r in rows) / n, 4),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-suites-only", action="store_true", help="Write eval & RL suites without running inference.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("BASE_GGUF", "/home/nos/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf")),
    )
    parser.add_argument("--lora", action="append", default=[], help="LoRA spec path:scale (repeatable)")
    parser.add_argument("--api-base", default=None, help="Optional existing OpenAI-compatible API base URL (e.g. http://127.0.0.1:8080/v1).")
    parser.add_argument("--api-model", default="local", help="Model name when --api-base is set.")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "work" / "gn_eval_baseline.json")
    parser.add_argument("--export-dir", type=Path, default=None, help="Directory to save passing .py + .blend scenes.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-budget", type=int, default=0)
    parser.add_argument("--reasoning-budget-message", type=str, default=bbw.DEFAULT_REASONING_BUDGET_MESSAGE)
    parser.add_argument("--temp", type=float, default=0.4)
    args = parser.parse_args()

    if args.init_suites_only:
        ensure_suites()
        print(f"Initialized suites: {DEFAULT_EVAL_SUITE} & {DEFAULT_RL_TASKS}")
        return 0

    run_benchmark(
        args.model,
        args.lora,
        args.out,
        export_dir=args.export_dir,
        max_tokens=args.max_tokens,
        reasoning_budget=args.reasoning_budget,
        temperature=args.temp,
        api_base=args.api_base,
        api_model=args.api_model,
        reasoning_budget_message=args.reasoning_budget_message,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
