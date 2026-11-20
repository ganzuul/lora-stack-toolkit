#!/usr/bin/env python3
"""Cycle 6 Layers 2 & 3 Builder: Pure Blender 5.4 Canonical GeoNodes Dataset + Pin-Flow IR + Unseen RL Tasks."""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_blenderworld as bbw  # noqa: E402
import benchmark_geonodes as bgn  # noqa: E402
import blender_schema_5_4 as bs54  # noqa: E402
import blender_verifier as bv  # noqa: E402
import build_compact_geonodes_dataset as bcg  # noqa: E402
import build_cycle5_unified_dataset as bc5  # noqa: E402
import train_lora as tlora  # noqa: E402

EVAL_FORBIDDEN_TERMS = (
    "cactus",
    "apple",
    "spiral stair",
    "fluted vase",
    "helical spring",
    "coil spring",
    "wave-relief",
    "wave panel",
    "wave relief",
    "vase",
    "stair",
    "spring",
)

CANONICAL_MAKE_MAT_LINES = [
    "def make_mat(name, color, rough=0.45, metallic=0.0):",
    "    mat = bpy.data.materials.new(name)",
    '    bsdf = mat.node_tree.nodes["Principled BSDF"]',
    '    bsdf.inputs["Base Color"].default_value = (*color, 1.0)',
    '    bsdf.inputs["Roughness"].default_value = rough',
    '    bsdf.inputs["Metallic"].default_value = metallic',
    "    return mat",
]

CANONICAL_CLEANUP_LINES = [
    "for o in list(bpy.data.objects):",
    "    bpy.data.objects.remove(o, do_unlink=True)",
    "for m in list(bpy.data.materials):",
    "    bpy.data.materials.remove(m)",
    "for g in list(bpy.data.node_groups):",
    "    bpy.data.node_groups.remove(g)",
]


def _rewrite_mat_calls_via_ast(src: str) -> str:
    """Safely rename pbr_mat(...) -> make_mat(...), normalize keywords, and fix SetShadeSmooth legacy pins via AST."""
    tree = ast.parse(src)
    # Collect (start_char, end_char, replacement) across function name, keyword names & legacy socket literals.
    line_starts: list[int] = [0]
    for ln in src.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(ln))

    def to_offset(lineno: int, col: int) -> int:
        return line_starts[lineno - 1] + col

    edits: list[tuple[int, int, str]] = []
    kw_map = {"roughness": "rough", "metal": "metallic", "mat_name": "name", "rgb": "color"}
    shade_smooth_vars: set[str] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and bs54._is_nodes_new_call(node.value) == "GeometryNodeSetShadeSmooth"
        ):
            shade_smooth_vars.add(node.targets[0].id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("pbr_mat", "make_mat"):
            if node.func.id == "pbr_mat":
                s = to_offset(node.func.lineno, node.func.col_offset)
                e = to_offset(node.func.end_lineno, node.func.end_col_offset)  # type: ignore[arg-type]
                edits.append((s, e, "make_mat"))
            for kw in node.keywords:
                if kw.arg in kw_map:
                    ks = to_offset(kw.lineno, kw.col_offset)
                    ke = to_offset(kw.end_lineno, kw.end_col_offset)  # type: ignore[arg-type]
                    kw_text = src[ks:ke]
                    new_arg = kw_map[kw.arg]
                    replaced = re.sub(rf"^{re.escape(kw.arg)}(\s*=)", rf"{new_arg}\1", kw_text, count=1)
                    if replaced != kw_text:
                        edits.append((ks, ks + len(kw.arg), new_arg))
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr in ("inputs", "outputs")
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in shade_smooth_vars
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "Geometry"
        ):
            ss = to_offset(node.slice.lineno, node.slice.col_offset)
            se = to_offset(node.slice.end_lineno, node.slice.end_col_offset)  # type: ignore[arg-type]
            quote = src[ss] if src[ss] in ("'", '"') else '"'
            edits.append((ss, se, f"{quote}Mesh{quote}"))

    out = src
    for s, e, rep in sorted(edits, key=lambda x: x[0], reverse=True):
        out = out[:s] + rep + out[e:]
    return out


def canonicalize_blender54_code(raw_code: str) -> str:
    """Normalize any C4/C5/RL GeoNodes script onto the flat Blender 5.4 canonical boundary contract."""
    code = bv.extract_python_code(raw_code).strip()
    lines = code.splitlines()

    # 1. Replace def make_mat / def pbr_mat with exact canonical 7-line Blender 5.4 helper
    tree = ast.parse(code)
    fn_spans: list[tuple[int, int]] = []
    use_nodes_lines: set[int] = set()
    cleanup_lines: set[int] = set()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("make_mat", "pbr_mat"):
            fn_spans.append((node.lineno - 1, (node.end_lineno or node.lineno) - 1))
        elif isinstance(node, ast.For):
            seg = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
            if "bpy.data.objects.remove" in seg or "bpy.data.materials.remove" in seg or "bpy.data.node_groups.remove" in seg:
                for li in range(node.lineno - 1, (node.end_lineno or node.lineno)):
                    cleanup_lines.add(li)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == "use_nodes":
                    for li in range(node.lineno - 1, (node.end_lineno or node.lineno)):
                        use_nodes_lines.add(li)

    fn_start_to_end = {s: e for s, e in fn_spans}
    fn_all = {li for s, e in fn_spans for li in range(s, e + 1)}
    first_cleanup = min(cleanup_lines) if cleanup_lines else None

    rebuilt: list[str] = []
    i = 0
    inserted_cleanup = False
    while i < len(lines):
        if i in cleanup_lines:
            if not inserted_cleanup and i == first_cleanup:
                rebuilt.extend(CANONICAL_CLEANUP_LINES)
                inserted_cleanup = True
            i += 1
            continue
        if i in fn_start_to_end:
            rebuilt.extend(CANONICAL_MAKE_MAT_LINES)
            i = fn_start_to_end[i] + 1
            continue
        if i in fn_all or i in use_nodes_lines:
            i += 1
            continue
        rebuilt.append(lines[i])
        i += 1

    code2 = "\n".join(rebuilt).strip() + "\n"
    code3 = _rewrite_mat_calls_via_ast(code2)
    # Ensure exact syntax validity after rewrite
    ast.parse(code3)
    return code3


def format_pin_flow_ir(dag_ir: list[dict[str, Any]]) -> str:
    """Serialize live Blender 5.4 GeometryNodeTree DAG telemetry into a compact <think> Pin-Flow IR."""
    lines = ["[Pin-Flow IR (Blender 5.4)]"]
    for tree_info in dag_ir[:2]:
        tname = str(tree_info.get("tree") or "GeoTree")
        nodes = list(dict.fromkeys(str(n) for n in (tree_info.get("nodes") or [])))
        edges = tree_info.get("edges") or []
        if len(dag_ir) > 1:
            lines.append(f"Tree: {tname}")
        lines.append("Nodes: " + " -> ".join(nodes[:16]))
        if edges:
            lines.append("Wires:")
            for src_n, src_p, src_t, dst_n, dst_p, dst_t in edges[:18]:
                s_short = (
                    str(src_n)
                    .replace("GeometryNode", "")
                    .replace("ShaderNode", "Shader")
                    .replace("FunctionNode", "Fn")
                )
                d_short = (
                    str(dst_n)
                    .replace("GeometryNode", "")
                    .replace("ShaderNode", "Shader")
                    .replace("FunctionNode", "Fn")
                )
                lines.append(f"  {s_short}[{src_p}:{src_t}] -> {d_short}[{dst_p}:{dst_t}]")
    lines.append(
        "Boundary: attach NODES modifier, assign make_mat Principled BSDF materials, "
        "SUN light, active CAMERA, Cycles 64, save /workspace/exports/final.blend."
    )
    return "\n".join(lines)


def _check_eval_leakage(id_str: str, texts: list[str]) -> None:
    eval_ids = {str(t["id"]).lower() for t in bgn.HELD_OUT_EVAL_TASKS}
    assert id_str.lower() not in eval_ids, f"Eval ID collision: {id_str}"
    hay = f"{id_str} " + " ".join(texts)
    low = hay.lower()
    for term in EVAL_FORBIDDEN_TERMS:
        assert term not in low, f"Forbidden held-out eval term '{term}' found in '{id_str}'"


def _build_c6_rl_tasks() -> list[dict[str, str]]:
    """Build 8 completely unseen Cycle 6 Blender 5.4 GeoNodes RL tasks."""
    kept_c4_rl = [
        t for t in bgn.RL_TRAIN_TASKS if t["id"] in ("rl_gn_02_twisted_torus", "rl_gn_05_bamboo_stalk", "rl_gn_08_pine_bough")
    ]
    new_c6_rl = [
        {
            "id": "rl_gn_09_braided_bracelet",
            "prompt": (
                "Build a sculpted metallic braided cuff bracelet built from three intertwined sinusoidal tubular strands "
                "swept along circular profiles around a split wrist arc."
            ) + bgn._SUFFIX,
        },
        {
            "id": "rl_gn_10_pagoda_lantern",
            "prompt": (
                "Build an ornamental carved stone courtyard lantern tower with flared multi-tiered eaves, "
                "chamfered pillars, and warm glowing shoji panels."
            ) + bgn._SUFFIX,
        },
        {
            "id": "rl_gn_11_dandelion_puff",
            "prompt": (
                "Build a botanical translucent white dandelion seed head sphere with fine slender filaments "
                "and tiny feathered pappus tufts distributed radially via Distribute Points on Faces and Instance on Points."
            ) + bgn._SUFFIX,
        },
        {
            "id": "rl_gn_12_twisted_candlestick",
            "prompt": (
                "Build a turned brass helical barley-twist candlestick holder with a flared pedestal foot, "
                "twisted spiral shaft, and shallow drip bobeche cup."
            ) + bgn._SUFFIX,
        },
        {
            "id": "rl_gn_13_honeycomb_tile",
            "prompt": (
                "Build a procedural geometric hexagonal basalt floor relief tile with varied bevelled prism columns "
                "instanced across a grid using Set Position height ripples."
            ) + bgn._SUFFIX,
        },
    ]
    tasks = kept_c4_rl + new_c6_rl
    assert len(tasks) == 8
    for t in tasks:
        _check_eval_leakage(t["id"], [t["prompt"]])
        assert "4.2" not in t["prompt"], f"Legacy version in RL task {t['id']}"
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "work" / "curriculum_c6_gn_sme.jsonl")
    ap.add_argument("--rl-out", type=Path, default=SCRIPT_DIR / "work" / "geonodes_rl_tasks_c6.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2250)
    args = ap.parse_args()

    schema = bs54.extract_schema(out_json=bs54.DEFAULT_SCHEMA_JSON)
    print(f"[c6_data] extracted live Blender schema ({schema['blender_version']}): {schema['node_count']} nodes", flush=True)

    tok = tlora.load_qwen35_tokenizer()

    # 1. Load 60 C4 GeoNodes rows
    c4_path = SCRIPT_DIR / "work" / "curriculum_c4_compact_geonodes.jsonl"
    c4_all = [json.loads(l) for l in c4_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    c4_rows = [r for r in c4_all if r.get("source") == "c4_compact_geonodes"]
    assert len(c4_rows) == 60, f"Expected 60 C4 GeoNodes rows, got {len(c4_rows)}"

    # 2. Load 10 C5 Spiral GeoNodes rows
    c5_rows = bc5._build_c5_spiral_recipes()
    assert len(c5_rows) == 10, f"Expected 10 C5 spiral rows, got {len(c5_rows)}"

    # 3. Load 5 Harvested RL GeoNodes Winners
    rl_win_path = SCRIPT_DIR / "work" / "rl_gn_harvested_winners.jsonl"
    rl_rows = [json.loads(l) for l in rl_win_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rl_rows) == 5, f"Expected 5 rl_gn_harvested_winners rows, got {len(rl_rows)}"

    # Combine & Canonicalize (Layer 2 Boundary Gauge)
    staged: list[dict[str, Any]] = []
    for idx, (src_group, raw) in enumerate(
        [("c4_gn", r) for r in c4_rows] + [("c5_spiral_gn", r) for r in c5_rows] + [("rl_gn_winner", r) for r in rl_rows]
    ):
        rid = str(raw["id"])
        tid = str(raw.get("task_id") or rid)
        canon_code = canonicalize_blender54_code(str(raw["response"]))
        prompt = str(raw["prompt"]).strip()
        assert "4.2" not in prompt, f"Legacy 4.2 version in prompt of {rid}"
        assert "/workspace/exports/final.blend" in prompt, f"Missing export path in prompt of {rid}"

        # Lint with Layer 1 live schema + Layer 2 canonical boundary enforcement
        lint_errs = bs54.lint_geonodes_script(canon_code, schema, enforce_canonical_boundary=True)
        assert not lint_errs, f"Lint errors in {rid}:\n" + "\n".join(lint_errs)

        _check_eval_leakage(tid, [prompt, canon_code])

        staged.append({
            "id": rid,
            "task_id": tid,
            "domain": "geonodes",
            "source": src_group,
            "system": bcg.GN_SYSTEM_PROMPT,
            "prompt": prompt,
            "reasoning": "",
            "response": f"```python\n{canon_code.strip()}\n```",
            "_idx": idx,
        })

    assert len(staged) == 75
    print(f"[c6_data] schema + canonical boundary lint PASSED on all 75 GeoNodes scripts", flush=True)

    # 4. Headless Blender 5.4 Structural Verification (Layer 4 Gate)
    print(f"[c6_data] running Blender 5.4 structural verifier on 75 scripts (workers={args.workers})...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(bv.verify_script, [r["response"] for r in staged]))

    failures: list[str] = []
    for row, vr in zip(staged, results):
        gp = bv.is_geonodes_pass(vr)
        ss = vr.scene_summary or {}
        verts = int(ss.get("total_verts", 0))
        ok = (
            vr.syntax_ok
            and vr.exec_ok
            and gp
            and ss.get("geonode_invalid_links", 1) == 0
            and ss.get("geonode_reaches_output") is True
            and verts >= 32
            and vr.reward >= 0.85
        )
        row["reward"] = vr.reward
        row["verts"] = verts
        row["gn_nodes"] = ss.get("geonode_nodes_total", 0)
        row["gn_links"] = ss.get("geonode_valid_links", 0)
        row["gn_connected_ratio"] = ss.get("geonode_connected_ratio", 0.0)
        row["_dag_ir"] = ss.get("geonode_dag_ir") or []
        if not ok:
            failures.append(
                f"{row['id']}: syn={vr.syntax_ok} exec={vr.exec_ok} gp={gp} inv={ss.get('geonode_invalid_links')} "
                f"reach={ss.get('geonode_reaches_output')} verts={verts} R={vr.reward} err={vr.error[:200]}"
            )
    if failures:
        raise SystemExit("Verification failures:\n" + "\n".join(failures))
    print(f"[c6_data] 75/75 PASSED Blender 5.4 structural verification (0 invalid links, 75/75 reach output)", flush=True)

    # 5. Layer 3 Stratified Pin-Flow IR <think> Traces (34/75 total: ~26/60 C4 across F1-F7, 5/10 C5, 3/5 RL)
    c4_by_fam: dict[str, list[dict[str, Any]]] = {}
    c5_list: list[dict[str, Any]] = []
    rl_list: list[dict[str, Any]] = []
    for r in staged:
        if r["source"] == "c4_gn":
            fam = r["id"].split("_", 1)[0]
            c4_by_fam.setdefault(fam, []).append(r)
        elif r["source"] == "c5_spiral_gn":
            c5_list.append(r)
        else:
            rl_list.append(r)

    chosen_ids: set[str] = set()
    # Pick ~3-4 per C4 family (total 26 across F1..F7)
    fam_round = 0
    while sum(1 for r in staged if r["source"] == "c4_gn" and r["id"] in chosen_ids) < 26:
        for fam in sorted(c4_by_fam.keys()):
            if sum(1 for r in staged if r["source"] == "c4_gn" and r["id"] in chosen_ids) >= 26:
                break
            lst = c4_by_fam[fam]
            if fam_round < len(lst):
                cand = lst[fam_round]
                ir_txt = format_pin_flow_ir(cand["_dag_ir"])
                test = copy.deepcopy(cand)
                test["reasoning"] = ir_txt
                n_tok = len(tlora.format_conversation(test, tok, max_seq_len=99999, truncate=False)[0])
                if n_tok <= args.max_tokens:
                    chosen_ids.add(cand["id"])
        fam_round += 1

    # Pick 5/10 C5 spirals & 3/5 RL winners
    for pool_list, target_n in [(c5_list, 5), (rl_list, 3)]:
        cnt = 0
        for cand in pool_list[:: max(1, len(pool_list) // target_n)] + pool_list:
            if cnt >= target_n:
                break
            if cand["id"] in chosen_ids:
                continue
            ir_txt = format_pin_flow_ir(cand["_dag_ir"])
            test = copy.deepcopy(cand)
            test["reasoning"] = ir_txt
            n_tok = len(tlora.format_conversation(test, tok, max_seq_len=99999, truncate=False)[0])
            if n_tok <= args.max_tokens:
                chosen_ids.add(cand["id"])
                cnt += 1

    cutoff_msg = bbw.DEFAULT_REASONING_BUDGET_MESSAGE.strip()
    reasoned_rows: list[dict[str, Any]] = []
    for r in staged:
        if r["id"] in chosen_ids:
            r["reasoning"] = format_pin_flow_ir(r["_dag_ir"])
            reasoned_rows.append(r)

    # Add cutoff marker to 6 reasoned rows spread across groups
    cutoff_count = 0
    step_c = max(1, len(reasoned_rows) // 6)
    for cand in reasoned_rows[::step_c] + reasoned_rows:
        if cutoff_count >= 6:
            break
        if cutoff_msg in cand["reasoning"]:
            continue
        new_r = cand["reasoning"].rstrip() + "\n" + cutoff_msg
        test = copy.deepcopy(cand)
        test["reasoning"] = new_r
        n_tok = len(tlora.format_conversation(test, tok, max_seq_len=99999, truncate=False)[0])
        if n_tok <= args.max_tokens:
            cand["reasoning"] = new_r
            cand["cutoff_example"] = True
            cutoff_count += 1
    assert cutoff_count == 6, f"Expected 6 cutoff examples, got {cutoff_count}"

    # Final serialize & invariant assertions
    tok_lens: list[int] = []
    clean_rows: list[dict[str, Any]] = []
    for r in staged:
        r.pop("_dag_ir", None)
        r.pop("_idx", None)
        ids, _ = tlora.format_conversation(r, tok, max_seq_len=99999, truncate=False)
        r["tokens"] = len(ids)
        assert r["tokens"] <= args.max_tokens, f"Row {r['id']} exceeded {args.max_tokens} tok: {r['tokens']}"
        tok_lens.append(r["tokens"])
        clean_rows.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in clean_rows) + "\n", encoding="utf-8")

    rl_tasks = _build_c6_rl_tasks()
    args.rl_out.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in rl_tasks) + "\n", encoding="utf-8")

    n_reason = sum(1 for r in clean_rows if r["reasoning"])
    print(f"\n=== CYCLE 6 GEONODES SME DATASET READY: {args.out} ===", flush=True)
    print(f"  Total rows       : {len(clean_rows)} (100% GeoNodes, Blender 5.4 canonical)", flush=True)
    print(f"  Pin-Flow IR      : {n_reason}/{len(clean_rows)} reasoned ({100*n_reason/len(clean_rows):.1f}%), cutoff={cutoff_count}", flush=True)
    print(f"  Token lengths    : min={min(tok_lens)}, med={int(statistics.median(tok_lens))}, max={max(tok_lens)}", flush=True)
    print(f"  RL tasks (n={len(rl_tasks)})  : {args.rl_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
