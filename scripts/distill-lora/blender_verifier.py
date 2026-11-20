#!/usr/bin/env python3
"""Headless Blender 4.2.10 verifier and shaped reward scorer for Qwen/BlenderWorld."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BLENDER_BIN = Path(os.environ.get("BLENDER_BIN", str(SCRIPT_DIR / "blender")))
DEFAULT_CACHE_DIR = SCRIPT_DIR / "work" / "blenderworld_cache"

CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class VerifyResult:
    syntax_ok: bool
    exec_ok: bool
    r_syntax: float
    r_exec: float
    r_scene: float
    r_rubric: float
    reward: float
    error: str = ""
    scene_summary: dict[str, Any] | None = None
    rubric_hits: int = 0
    rubric_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_geonodes_pass(
    vr: VerifyResult,
    min_nodes: int = 4,
    min_links: int = 3,
    min_verts: int = 32,
) -> bool:
    """True when script executed cleanly and produced a real linked GeometryNodeTree mesh."""
    ss = vr.scene_summary or {}
    return bool(
        vr.syntax_ok
        and vr.exec_ok
        and ss.get("geonode_trees_count", 0) >= 1
        and ss.get("geonode_nodes_total", 0) >= min_nodes
        and ss.get("geonode_links_total", 0) >= min_links
        and ss.get("total_verts", 0) >= min_verts
        and ss.get("geonode_invalid_links", 0) == 0
        and bool(ss.get("geonode_reaches_output", True))
    )


def extract_python_code(text: str) -> str:
    """Extract the most likely standalone bpy Python script from an LLM response."""
    cleaned = THINK_TAG_RE.sub("", text or "").strip()
    blocks = CODE_FENCE_RE.findall(cleaned)
    if blocks:
        bpy_blocks = [b for b in blocks if "import bpy" in b or "from bpy" in b]
        return (bpy_blocks[-1] if bpy_blocks else max(blocks, key=len)).strip()
    if "import bpy" in cleaned:
        m = re.search(r"^(?:import |from |#!|\"\"\"|''')", cleaned, re.MULTILINE)
        idx = m.start() if m else cleaned.find("import bpy")
        return cleaned[idx:].strip()
    return cleaned


def load_blenderworld_text_tasks(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    eval_count: int = 8,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Load/cache the 78 text-only Qwen/BlenderWorld tasks and deterministic train/eval split."""
    from huggingface_hub import HfApi, hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    split_path = cache_dir / "split.json"
    manifest_path = cache_dir / "text_manifest.json"

    if manifest_path.is_file() and split_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = json.loads(split_path.read_text(encoding="utf-8"))
        task_ids = manifest["text_task_ids"][:max_tasks] if max_tasks else manifest["text_task_ids"]
        return {"manifest": manifest, "split": split, "task_ids": task_ids}

    api = HfApi()
    info = api.dataset_info("Qwen/BlenderWorld")
    has_media: set[str] = set()
    all_bw: set[str] = set()
    siblings_by_task: dict[str, list[str]] = {}

    for s in info.siblings:
        fn = s.rfilename
        if not fn.startswith("bw_"):
            continue
        tid = fn.split("/", 1)[0]
        all_bw.add(tid)
        siblings_by_task.setdefault(tid, []).append(fn)
        if "/inputs/references/" in fn or fn.endswith((".mp4", ".mov", ".webm", ".mkv")):
            has_media.add(tid)

    tax_path = hf_hub_download("Qwen/BlenderWorld", "taxonomy.json", repo_type="dataset")
    taxonomy = json.loads(Path(tax_path).read_text(encoding="utf-8"))
    tax_map = {t["task_id"]: t for t in taxonomy.get("tasks", [])}

    text_ids = sorted(all_bw - has_media)
    # Deterministic stratified-order split (every ~10th task held out for eval).
    step = max(1, len(text_ids) // max(1, eval_count))
    eval_ids = [text_ids[i * step] for i in range(min(eval_count, len(text_ids)))]
    eval_set = set(eval_ids)
    train_ids = [tid for tid in text_ids if tid not in eval_set]

    manifest = {
        "text_task_ids": text_ids,
        "taxonomy": {tid: tax_map.get(tid, {}) for tid in text_ids},
        "attachments": {
            tid: [
                fn for fn in siblings_by_task.get(tid, [])
                if "/environment/inputs/attachments/" in fn
            ]
            for tid in text_ids
        },
    }
    split = {"eval": eval_ids, "train": train_ids}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    _ = hf_hub_download  # imported for downstream loaders
    task_ids = text_ids[:max_tasks] if max_tasks else text_ids
    return {"manifest": manifest, "split": split, "task_ids": task_ids}


def get_blenderworld_task(task_id: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    """Fetch and cache a single BlenderWorld task prompt, attachments, and rubric JSON."""
    from huggingface_hub import hf_hub_download

    tdir = cache_dir / task_id
    cached_json = tdir / "task_bundle.json"
    if cached_json.is_file():
        return json.loads(cached_json.read_text(encoding="utf-8"))

    meta = load_blenderworld_text_tasks(cache_dir=cache_dir)
    q_path = hf_hub_download("Qwen/BlenderWorld", f"{task_id}/environment/inputs/query.txt", repo_type="dataset")
    r_path = hf_hub_download(
        "Qwen/BlenderWorld", f"{task_id}/tests/dataset/tasks/{task_id}/rubric.json", repo_type="dataset"
    )
    query = Path(q_path).read_text(encoding="utf-8").strip()
    rubric = json.loads(Path(r_path).read_text(encoding="utf-8"))

    attachments: dict[str, str] = {}
    for rel_fn in meta["manifest"].get("attachments", {}).get(task_id, []):
        if rel_fn.endswith((".md", ".txt", ".json", ".yaml", ".yml", ".csv")):
            ap = hf_hub_download("Qwen/BlenderWorld", rel_fn, repo_type="dataset")
            name = rel_fn.split("/attachments/", 1)[-1]
            attachments[name] = Path(ap).read_text(encoding="utf-8", errors="replace")[:12000]

    bundle = {
        "task_id": task_id,
        "query": query,
        "attachments": attachments,
        "rubric": rubric,
        "taxonomy": meta["manifest"].get("taxonomy", {}).get(task_id, {}),
    }
    tdir.mkdir(parents=True, exist_ok=True)
    cached_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle


def format_task_prompt(bundle: dict[str, Any], include_rubrics: bool = False) -> str:
    """Format a BlenderWorld task (+ attachments) as a self-contained Blender Python coding prompt."""
    parts = [bundle["query"].strip()]
    for name, content in (bundle.get("attachments") or {}).items():
        parts.append(f"\n### Attached specification: {name}\n{content.strip()}")
    if include_rubrics and bundle.get("rubric", {}).get("rubrics"):
        crit_lines = [f"- [{r.get('id', '')}] {r.get('criterion', '')}" for r in bundle["rubric"]["rubrics"][:12]]
        parts.append("\n### Quality checklist (satisfy all items cleanly in geometry, materials, and lighting):\n" + "\n".join(crit_lines))
    parts.append(
        "\nWrite a complete, standalone Blender 4.2.10 Python (`bpy`) script that builds this entire scene from scratch: "
        "clear existing objects via `for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)` "
        "(do not call `bpy.ops.mesh.delete` or `mode_set` on an empty scene), define `make_material(...)` and "
        "`assign_material(obj, mat)` using Blender 4.2 Principled BSDF names (`'Specular IOR Level'`, `'Coat Weight'`, `'Transmission Weight'`) "
        "and `obj.data.materials.append(mat)`, configure lighting, active camera, and Cycles (`64` samples), "
        "and save `/workspace/exports/final.blend`. Keep the script concise (under 130 lines) in one ```python block."
    )
    return "\n".join(parts)


_BLENDER_HARNESS = textwrap.dedent(r"""
    import json
    import os
    import sys
    import traceback
    from pathlib import Path
    import bpy
    from mathutils import Vector

    cand_path = Path(os.environ["BW_CANDIDATE_PY"])
    work_root = Path(os.environ["BW_SANDBOX_ROOT"])
    summary_path = Path(os.environ["BW_SUMMARY_JSON"])
    exports_dir = work_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Redirect /workspace paths written in candidate scripts into the sandbox directory.
    src = cand_path.read_text(encoding="utf-8")
    src = src.replace('"/workspace', f'"{work_root.as_posix()}').replace("'/workspace", f"'{work_root.as_posix()}")

    # Keep verification fast on CPU: capture render intent without tracing full 1080p Cycles frames.
    orig_render = bpy.ops.render.render
    render_calls = []

    def _fast_render_stub(*args, **kwargs):
        sc = bpy.context.scene
        render_calls.append({
            "engine": sc.render.engine,
            "res_x": int(sc.render.resolution_x),
            "res_y": int(sc.render.resolution_y),
            "samples": int(getattr(sc.cycles, "samples", 0)) if hasattr(sc, "cycles") else 0,
            "filepath": str(sc.render.filepath),
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        })
        return {"FINISHED"}

    bpy.ops.render.render = _fast_render_stub

    # Also guard slow bpy.ops.wm.save_as_mainfile missing parent dirs
    orig_save = bpy.ops.wm.save_as_mainfile

    def _safe_save(*args, **kwargs):
        fp = kwargs.get("filepath")
        if fp:
            Path(bpy.path.abspath(fp)).parent.mkdir(parents=True, exist_ok=True)
        return orig_save(*args, **kwargs)

    bpy.ops.wm.save_as_mainfile = _safe_save

    ns = {"__name__": "__main__", "__file__": str(cand_path)}
    try:
        exec(compile(src, str(cand_path), "exec"), ns, ns)
    except SystemExit as se:
        code = int(se.code) if isinstance(se.code, int) else 1
        if code != 0:
            traceback.print_exc()
            sys.exit(code)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    sc = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    objects_info = []
    mesh_count = 0
    total_verts = 0
    total_polys = 0
    distinct_locs = set()
    world_min = [1e9, 1e9, 1e9]
    world_max = [-1e9, -1e9, -1e9]

    for obj in sc.objects:
        loc = tuple(round(float(v), 2) for v in obj.matrix_world.translation)
        row = {
            "name": obj.name,
            "type": obj.type,
            "location": list(loc),
            "dimensions": [round(float(d), 3) for d in obj.dimensions],
            "materials": [m.name for m in getattr(obj.data, "materials", []) if m],
        }
        if obj.type == "MESH" and obj.data is not None:
            mesh_count += 1
            ev = obj.evaluated_get(deps)
            ev_mesh = ev.to_mesh()
            try:
                mesh_eval = ev_mesh if ev_mesh is not None else obj.data
                nv = len(mesh_eval.vertices)
                np_ = len(mesh_eval.polygons)
                ev_mats = [m.name for m in getattr(mesh_eval, "materials", []) if m]
                if ev_mats:
                    row["materials"] = ev_mats
            finally:
                if ev_mesh is not None:
                    ev.to_mesh_clear()
            total_verts += nv
            total_polys += np_
            row["verts"] = nv
            row["polys"] = np_
            distinct_locs.add(loc)
            for c in ev.bound_box:
                wc = ev.matrix_world @ Vector(c)
                for ax in range(3):
                    world_min[ax] = min(world_min[ax], float(wc[ax]))
                    world_max[ax] = max(world_max[ax], float(wc[ax]))
        objects_info.append(row)

    materials_info = []
    principled_non_default = 0
    for mat in bpy.data.materials:
        m_entry = {"name": mat.name, "use_nodes": bool(mat.node_tree is not None), "principled": False}
        if mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    m_entry["principled"] = True
                    bc = tuple(round(float(x), 3) for x in node.inputs["Base Color"].default_value)
                    rough = round(float(node.inputs["Roughness"].default_value), 3)
                    metal = round(float(node.inputs["Metallic"].default_value), 3)
                    m_entry["base_color"] = list(bc)
                    m_entry["roughness"] = rough
                    m_entry["metallic"] = metal
                    if bc != (0.8, 0.8, 0.8, 1.0) or rough != 0.5 or metal != 0.0 or len(mat.node_tree.links) > 1:
                        principled_non_default += 1
                    break
        materials_info.append(m_entry)

    lights = [{"name": o.name, "kind": o.data.type, "energy": float(o.data.energy)} for o in sc.objects if o.type == "LIGHT" and o.data]
    cameras = [{"name": o.name, "active": o == sc.camera, "lens": float(o.data.lens)} for o in sc.objects if o.type == "CAMERA" and o.data]
    bbox_span = [max(0.0, world_max[i] - world_min[i]) if mesh_count else 0.0 for i in range(3)]
    bbox_vol = bbox_span[0] * bbox_span[1] * bbox_span[2]
    gn_trees = [ng for ng in bpy.data.node_groups if getattr(ng, "bl_idname", "") == "GeometryNodeTree"]
    geonode_trees_count = len(gn_trees)
    geonode_nodes_total = sum(len(ng.nodes) for ng in gn_trees)
    geonode_links_total = sum(len(ng.links) for ng in gn_trees)
    geonode_valid_links = 0
    geonode_invalid_links = 0
    non_layout_total = 0
    connected_non_layout = 0
    reaches_output_any = False
    geonode_dag_ir = []
    gn_tree_set = set(gn_trees)
    geonode_modifier_bound = False
    for obj in sc.objects:
        if obj.type == "MESH":
            for mod in getattr(obj, "modifiers", []):
                if getattr(mod, "type", "") == "NODES" and getattr(mod, "node_group", None) in gn_tree_set:
                    geonode_modifier_bound = True
                    break
            if geonode_modifier_bound:
                break
    if not geonode_modifier_bound and geonode_trees_count >= 1 and total_verts >= 32:
        geonode_modifier_bound = True
    for ng in gn_trees:
        ng_valid = [lk for lk in ng.links if lk.is_valid]
        geonode_valid_links += len(ng_valid)
        geonode_invalid_links += len(ng.links) - len(ng_valid)
        conn_nodes = set()
        for lk in ng.links:
            conn_nodes.add(lk.from_node)
            conn_nodes.add(lk.to_node)
        non_layout = [n for n in ng.nodes if n.bl_idname not in ("NodeFrame", "NodeReroute")]
        non_layout_total += len(non_layout)
        connected_non_layout += sum(1 for n in non_layout if n in conn_nodes)
        visited = {n for n in ng.nodes if n.bl_idname == "NodeGroupOutput"}
        changed = True
        while changed:
            changed = False
            for lk in ng_valid:
                if lk.to_node in visited and lk.from_node not in visited:
                    visited.add(lk.from_node)
                    changed = True
        if len(visited) >= 3:
            reaches_output_any = True
        edges = [
            [lk.from_node.bl_idname, lk.from_socket.name, lk.from_socket.type, lk.to_node.bl_idname, lk.to_socket.name, lk.to_socket.type]
            for lk in ng_valid
        ][:64]
        geonode_dag_ir.append({
            "tree": ng.name,
            "nodes": [n.bl_idname for n in ng.nodes],
            "edges": edges,
        })
    geonode_connected_ratio = round(connected_non_layout / max(1, non_layout_total), 4) if non_layout_total else 0.0
    geonode_reaches_output = bool(reaches_output_any) if geonode_trees_count else False

    summary = {
        "mesh_count": mesh_count,
        "total_verts": total_verts,
        "total_polys": total_polys,
        "distinct_mesh_locs": len(distinct_locs),
        "bbox_span": [round(x, 3) for x in bbox_span],
        "bbox_vol": round(bbox_vol, 4),
        "lights_count": len(lights),
        "cameras_count": len(cameras),
        "has_active_camera": any(c["active"] for c in cameras),
        "materials_count": len(materials_info),
        "principled_non_default": principled_non_default,
        "world_use_nodes": bool(sc.world and sc.world.node_tree is not None),
        "geonode_trees_count": geonode_trees_count,
        "geonode_nodes_total": geonode_nodes_total,
        "geonode_links_total": geonode_links_total,
        "geonode_valid_links": geonode_valid_links,
        "geonode_invalid_links": geonode_invalid_links,
        "geonode_connected_ratio": geonode_connected_ratio,
        "geonode_reaches_output": geonode_reaches_output,
        "geonode_modifier_bound": geonode_modifier_bound,
        "geonode_dag_ir": geonode_dag_ir,
        "engine": str(sc.render.engine),
        "res_x": int(sc.render.resolution_x),
        "res_y": int(sc.render.resolution_y),
        "samples": int(getattr(sc.cycles, "samples", 0)) if hasattr(sc, "cycles") else 0,
        "render_calls": render_calls,
        "saved_blend_exists": any(exports_dir.rglob("*.blend")),
        "objects": objects_info[:120],
        "materials": materials_info[:60],
        "lights": lights[:20],
        "cameras": cameras[:10],
        "collections": [c.name for c in bpy.data.collections],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if os.environ.get("BW_FORCE_SAVE_BLEND") == "1" and not summary["saved_blend_exists"]:
        orig_save(filepath=str(exports_dir / "final.blend"))
""")


def _score_rubrics(summary: dict[str, Any], rubric: dict[str, Any] | None, code: str) -> tuple[float, int, int]:
    """Score deterministic alignment between the generated Blender scene/code and task rubrics."""
    if not rubric or not rubric.get("rubrics"):
        # For generic non-BlenderWorld bpy scripts, grade against core Blender 4.2 best practices.
        score = 0.0
        if summary["mesh_count"] >= 4 or (summary.get("geonode_trees_count", 0) >= 1 and summary.get("geonode_links_total", 0) >= 3):
            score += 0.10
        if summary["principled_non_default"] >= 2 and summary["lights_count"] >= 1:
            score += 0.10
        if summary["has_active_camera"] and (summary["saved_blend_exists"] or "save_as_mainfile" in code):
            score += 0.10
        return round(score, 4), int(round(score * 10)), 3

    items = rubric["rubrics"]
    haystack = " ".join(
        [
            " ".join(o["name"] for o in summary.get("objects", [])),
            " ".join(m["name"] for m in summary.get("materials", [])),
            " ".join(summary.get("collections", [])),
            summary.get("engine", ""),
            code[:8000],
        ]
    ).lower()

    stop = {
        "must", "with", "that", "this", "from", "should", "clearly", "include", "scene", "render",
        "shows", "using", "across", "visible", "overall", "remain", "requested", "present", "image",
        "views", "view", "based", "clean", "readable", "model", "objects", "object", "material",
    }

    hits = 0
    for item in items:
        cat = str(item.get("category", "")).lower()
        subj = str(item.get("subject", "")).lower()
        crit = str(item.get("criterion", "")).lower()
        tokens = [w for w in re.findall(r"[a-z0-9]{4,}", subj + " " + crit) if w not in stop]

        matched = False
        if "cycles" in crit or "cycles" in haystack:
            if "cycles" in crit and summary.get("engine") == "CYCLES":
                matched = True
        if not matched and "camera" in cat or "camera" in subj or "view" in subj:
            matched = summary.get("cameras_count", 0) >= 1
        if not matched and ("light" in subj or "lighting" in cat or "illuminat" in crit):
            matched = summary.get("lights_count", 0) >= 1 or summary.get("world_use_nodes", False)
        if not matched and "material" in cat:
            matched = summary.get("principled_non_default", 0) >= 2 and (not tokens or any(t in haystack for t in tokens[:8]))
        if not matched and tokens:
            uniq = list(dict.fromkeys(tokens))[:10]
            overlap = sum(1 for t in uniq if t in haystack)
            matched = overlap >= min(2, max(1, len(uniq) // 4))
        if matched:
            hits += 1

    frac = hits / max(1, len(items))
    return round(0.30 * frac, 4), hits, len(items)


def verify_script(
    response_text: str,
    task_bundle: dict[str, Any] | None = None,
    blender_bin: Path = DEFAULT_BLENDER_BIN,
    timeout_sec: int = 45,
    save_blend_path: Path | None = None,
) -> VerifyResult:
    """Compile and execute a generated bpy script inside headless Blender 4.2.10, returning R in [0, 1]."""
    code = extract_python_code(response_text)
    if not code or ("bpy" not in code):
        return VerifyResult(False, False, 0.0, 0.0, 0.0, 0.0, 0.0, error="Empty code or missing bpy import")

    try:
        compile(code, "<candidate_bpy>", "exec")
    except SyntaxError as exc:
        return VerifyResult(False, False, 0.0, 0.0, 0.0, 0.0, 0.0, error=f"SyntaxError: {exc}")

    r_syntax = 0.15
    if not Path(blender_bin).exists():
        raise FileNotFoundError(f"Blender binary not found at {blender_bin}")

    with tempfile.TemporaryDirectory(prefix="bw_eval_") as tmp:
        tmp_path = Path(tmp)
        cand_py = tmp_path / "candidate.py"
        harness_py = tmp_path / "harness.py"
        summary_json = tmp_path / "scene_summary.json"
        cand_py.write_text(code, encoding="utf-8")
        harness_py.write_text(_BLENDER_HARNESS, encoding="utf-8")

        env = os.environ.copy()
        env["BW_CANDIDATE_PY"] = str(cand_py)
        env["BW_SANDBOX_ROOT"] = str(tmp_path)
        env["BW_SUMMARY_JSON"] = str(summary_json)
        if save_blend_path is not None:
            env["BW_FORCE_SAVE_BLEND"] = "1"

        try:
            proc = subprocess.run(
                [str(blender_bin), "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(harness_py)],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult(True, False, r_syntax, 0.0, 0.0, 0.0, r_syntax, error=f"Blender timeout after {timeout_sec}s")

        if proc.returncode != 0 or not summary_json.is_file():
            err_tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-25:])
            return VerifyResult(True, False, r_syntax, 0.0, 0.0, 0.0, r_syntax, error=err_tail or f"Blender exited {proc.returncode}")

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        if save_blend_path is not None:
            exports_dir = tmp_path / "exports"
            found_blend = exports_dir / "final.blend"
            if not found_blend.is_file():
                found_blend = next(exports_dir.rglob("*.blend"), None)
            if found_blend and found_blend.is_file():
                save_blend_path = Path(save_blend_path)
                save_blend_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(found_blend, save_blend_path)

    r_exec = 0.25
    r_scene = 0.0
    # Default Blender startup has only 1 "Cube" mesh; require meaningful multi-part creation or edits.
    if summary.get("mesh_count", 0) >= 3 or (
        summary.get("geonode_trees_count", 0) >= 1
        and summary.get("geonode_nodes_total", 0) >= 4
        and summary.get("total_verts", 0) >= 40
    ):
        r_scene += 0.10
    if (
        summary.get("total_verts", 0) >= 24
        and summary.get("bbox_vol", 0.0) > 1e-4
        and (summary.get("distinct_mesh_locs", 0) >= 2 or summary.get("geonode_trees_count", 0) >= 1)
    ):
        r_scene += 0.10
    if (
        summary.get("principled_non_default", 0) >= 1
        and (summary.get("lights_count", 0) >= 1 or summary.get("world_use_nodes", False))
        and summary.get("cameras_count", 0) >= 1
    ):
        r_scene += 0.10

    rubric = task_bundle.get("rubric") if task_bundle else None
    r_rubric, hits, total = _score_rubrics(summary, rubric, code)
    reward = round(r_syntax + r_exec + r_scene + r_rubric, 4)

    return VerifyResult(
        syntax_ok=True,
        exec_ok=True,
        r_syntax=r_syntax,
        r_exec=r_exec,
        r_scene=round(r_scene, 4),
        r_rubric=r_rubric,
        reward=reward,
        error="",
        scene_summary=summary,
        rubric_hits=hits,
        rubric_total=total,
    )


def _selftest() -> None:
    good_py = textwrap.dedent("""
    ```python
    import bpy
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for i, pos in enumerate([(0, 0, 0.5), (2.2, 0, 0.5), (-2.2, 0, 0.5), (0, 0, -0.05)]):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=pos)
        obj = bpy.context.active_object
        obj.name = f"Part_{i}"
        mat = bpy.data.materials.new(name=f"Mat_{i}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.1 + 0.2 * i, 0.3, 0.7, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.25
        bsdf.inputs["Specular IOR Level"].default_value = 0.6
        obj.data.materials.append(mat)

    bpy.ops.object.light_add(type='SUN', location=(4, -4, 6))
    bpy.ops.object.camera_add(location=(5, -6, 3))
    bpy.context.scene.camera = bpy.context.active_object
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = 64
    sc.render.resolution_x = 1024
    sc.render.resolution_y = 1024
    bpy.ops.wm.save_as_mainfile(filepath="/workspace/exports/final.blend")
    ```
    """)

    bad_api_py = textwrap.dedent("""
    import bpy
    mat = bpy.data.materials.new("OldMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Specular"].default_value = 0.5
    """)

    syntax_err_py = "import bpy\ndef broken(:\n  pass\n"

    live_gn_py = textwrap.dedent("""
    import bpy
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    ng = bpy.data.node_groups.new("LiveSpiral", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    go = ng.nodes.new("NodeGroupOutput")
    sp = ng.nodes.new("GeometryNodeCurveSpiral")
    sp.inputs["Resolution"].default_value = 32
    sp.inputs["Rotations"].default_value = 4.0
    sp.inputs["Start Radius"].default_value = 0.4
    sp.inputs["End Radius"].default_value = 0.4
    sp.inputs["Height"].default_value = 1.2
    ci = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
    ci.inputs["Resolution"].default_value = 12
    ci.inputs["Radius"].default_value = 0.05
    c2m = ng.nodes.new("GeometryNodeCurveToMesh")
    ng.links.new(sp.outputs["Curve"], c2m.inputs["Curve"])
    ng.links.new(ci.outputs["Curve"], c2m.inputs["Profile Curve"])
    ng.links.new(c2m.outputs["Mesh"], go.inputs["Geometry"])
    obj = bpy.data.objects.new("LiveSpiralObj", bpy.data.meshes.new("LiveSpiralMesh"))
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("LiveGeo", "NODES").node_group = ng
    bpy.ops.object.light_add(type="SUN", location=(4, -4, 6))
    bpy.ops.object.camera_add(location=(4, -5, 3))
    bpy.context.scene.camera = bpy.context.active_object
    bpy.ops.wm.save_as_mainfile(filepath="/workspace/exports/final.blend")
    """)

    r_good = verify_script(good_py)
    r_bad = verify_script(bad_api_py)
    r_syn = verify_script(syntax_err_py)
    r_gn = verify_script(live_gn_py)

    print("SELFTEST good:", r_good.reward, "exec:", r_good.exec_ok, "scene:", r_good.r_scene, "rubric:", r_good.r_rubric)
    print("SELFTEST bad_api:", r_bad.reward, "syntax:", r_bad.syntax_ok, "exec:", r_bad.exec_ok, "err_line:", r_bad.error.splitlines()[-1])
    print("SELFTEST syntax_err:", r_syn.reward, "syntax:", r_syn.syntax_ok, "err:", r_syn.error)
    print("SELFTEST live_gn:", r_gn.reward, "exec:", r_gn.exec_ok, "gn_pass:", is_geonodes_pass(r_gn), "verts:", (r_gn.scene_summary or {}).get("total_verts"))

    ss_gn = r_gn.scene_summary or {}
    assert r_good.syntax_ok and r_good.exec_ok and r_good.reward >= 0.70, r_good
    assert r_bad.syntax_ok and not r_bad.exec_ok and "Specular" in r_bad.error, r_bad
    assert not r_syn.syntax_ok and r_syn.reward == 0.0, r_syn
    assert r_gn.syntax_ok and r_gn.exec_ok and is_geonodes_pass(r_gn) and ss_gn.get("total_verts", 0) >= 1000, r_gn
    assert ss_gn.get("geonode_invalid_links") == 0, ss_gn
    assert ss_gn.get("geonode_reaches_output") is True, ss_gn
    assert len(ss_gn.get("geonode_dag_ir") or []) >= 1, ss_gn
    print("SELFTEST ALL PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Verify Blender 4.2.10 headless scoring on 3 fixtures.")
    parser.add_argument("--script", type=Path, help="Verify a standalone .py script file.")
    parser.add_argument("--task-id", help="Optional Qwen/BlenderWorld task ID (e.g. bw_0100) for rubric scoring.")
    parser.add_argument("--save-blend", type=Path, help="Copy the generated .blend scene out of the sandbox to this path.")
    parser.add_argument("--list-text-tasks", action="store_true", help="Print cached text-only task ID split.")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return 0
    if args.list_text_tasks:
        info = load_blenderworld_text_tasks()
        print(json.dumps(info["split"], indent=2))
        return 0
    if args.script:
        bundle = get_blenderworld_task(args.task_id) if args.task_id else None
        res = verify_script(
            args.script.read_text(encoding="utf-8"),
            task_bundle=bundle,
            save_blend_path=args.save_blend,
        )
        print(json.dumps(res.to_dict(), indent=2))
        return 0 if res.exec_ok else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
