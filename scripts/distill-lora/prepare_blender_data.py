#!/usr/bin/env python3
"""Curate and parallel-verify Blender 4.2.10 Python scripts across 5 Hugging Face datasets."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import blender_verifier as bv  # noqa: E402

_SOCKET_REPLACEMENTS = (
    ('inputs["Specular"]', 'inputs["Specular IOR Level"]'),
    ("inputs['Specular']", "inputs['Specular IOR Level']"),
    ('inputs["Clearcoat"]', 'inputs["Coat Weight"]'),
    ("inputs['Clearcoat']", "inputs['Coat Weight']"),
    ('inputs["Clearcoat Roughness"]', 'inputs["Coat Roughness"]'),
    ("inputs['Clearcoat Roughness']", "inputs['Coat Roughness']"),
    ('inputs["Transmission"]', 'inputs["Transmission Weight"]'),
    ("inputs['Transmission']", "inputs['Transmission Weight']"),
    ('inputs["Subsurface"]', 'inputs["Subsurface Weight"]'),
    ("inputs['Subsurface']", "inputs['Subsurface Weight']"),
)

_TOP_PREAMBLE = """\
import bpy
import bmesh
import colorsys
import colorsys as _colorsys
import math
import mathutils
import numpy as np
import random
import random as _random
from mathutils import Vector, Euler, Matrix, Quaternion, kdtree, noise

for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)


def make_material(name, color=(0.7, 0.7, 0.7, 1.0), roughness=0.45, metallic=0.0, coat=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color if len(color) == 4 else (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.5
        if coat and "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = coat
    return mat


def assign_material(obj, mat):
    if obj is not None and getattr(obj, "data", None) is not None:
        obj.data.materials.append(mat)
"""

_FLAT_FOOTER = """
# --- Studio Materials, Lighting, Camera & Export ---
_pal = [(0.18, 0.36, 0.62, 1.0), (0.68, 0.28, 0.22, 1.0), (0.24, 0.52, 0.36, 1.0), (0.58, 0.50, 0.34, 1.0)]
for _i, _m in enumerate(bpy.data.materials):
    _dc = tuple(float(c) for c in getattr(_m, "diffuse_color", (0.8, 0.8, 0.8, 1.0)))
    _m.use_nodes = True
    _b = _m.node_tree.nodes.get("Principled BSDF") or _m.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    if not _b.inputs["Base Color"].is_linked and tuple(round(float(x), 2) for x in _b.inputs["Base Color"].default_value) == (0.8, 0.8, 0.8, 1.0):
        _b.inputs["Base Color"].default_value = _dc if tuple(round(x, 2) for x in _dc) != (0.8, 0.8, 0.8, 1.0) else _pal[_i % 4]
    if round(float(_b.inputs["Roughness"].default_value), 2) == 0.5:
        _b.inputs["Roughness"].default_value = 0.42
    if "Specular IOR Level" in _b.inputs:
        _b.inputs["Specular IOR Level"].default_value = 0.5
_ms = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.data is not None]
for _i, _o in enumerate(_ms):
    if not _o.data.materials:
        assign_material(_o, make_material(f"AutoMat_{_i}", _pal[_i % 4], roughness=0.4))
while len([o for o in bpy.context.scene.objects if o.type == "MESH" and o.data is not None]) < 3:
    _idx = len(bpy.data.objects)
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0 if _idx % 2 == 0 else 4.0, -0.02 if _idx % 2 == 0 else 2.0))
    assign_material(bpy.context.active_object, make_material(f"StudioPlane_{_idx}", (0.18, 0.20, 0.24, 1.0), roughness=0.7))
if not any(o.type == "LIGHT" for o in bpy.context.scene.objects):
    bpy.ops.object.light_add(type="SUN", location=(4.0, -4.0, 6.0))
    bpy.context.active_object.data.energy = 3.0
if not any(o.type == "CAMERA" for o in bpy.context.scene.objects):
    bpy.ops.object.camera_add(location=(5.0, -6.0, 3.2))
if bpy.context.scene.camera is None:
    bpy.context.scene.camera = next(o for o in bpy.context.scene.objects if o.type == "CAMERA")
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 64
bpy.ops.wm.save_as_mainfile(filepath="/workspace/exports/final.blend")
""".strip()

_CLEAN_GN_HEADER = """\
import bpy, bmesh, math, random, colorsys
import colorsys as _colorsys, random as _random
import numpy as np
from mathutils import Vector, Euler, Matrix, Quaternion, kdtree, noise
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.materials): bpy.data.materials.remove(_m)
for _n in list(bpy.data.node_groups): bpy.data.node_groups.remove(_n)
""".strip()

_CLEAN_GN_FOOTER = """\
if not any(o.type == "LIGHT" for o in bpy.context.scene.objects):
    bpy.ops.object.light_add(type="SUN", location=(4.0, -4.0, 6.0))
    bpy.context.active_object.data.energy = 3.0
if not any(o.type == "CAMERA" for o in bpy.context.scene.objects):
    bpy.ops.object.camera_add(location=(5.0, -6.0, 3.2))
if bpy.context.scene.camera is None:
    bpy.context.scene.camera = next(o for o in bpy.context.scene.objects if o.type == "CAMERA")
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 64
bpy.ops.wm.save_as_mainfile(filepath="/workspace/exports/final.blend")
""".strip()

_GEONODE_TASK_SUFFIX = (
    "\nWrite a clean, standalone Blender 5.4 Python (`bpy`) script using Geometry Nodes "
    "(`GeometryNodeTree` with `ng.interface.new_socket`, `ng.nodes.new`, `ng.links.new`, and `NodeGroupOutput`) "
    "to build this procedural object from scratch, add lighting, an active camera, Cycles (64 samples), "
    "and save `/workspace/exports/final.blend`."
)

_SLIM_GEONODE_FOOTER = """
# --- Studio Materials, Lighting, Camera & Export ---
_gm1 = make_material("GeoPrimary", (0.28, 0.52, 0.38, 1.0), roughness=0.38)
_gm2 = make_material("GeoAccent", (0.22, 0.24, 0.28, 1.0), roughness=0.55)
for _o in [o for o in bpy.context.scene.objects if o.type == "MESH" and o.data is not None and not o.data.materials]:
    assign_material(_o, _gm1)
if not any(o.type == "LIGHT" for o in bpy.context.scene.objects):
    bpy.ops.object.light_add(type="SUN", location=(4.0, -4.0, 6.0))
    bpy.context.active_object.data.energy = 3.0
if not any(o.type == "CAMERA" for o in bpy.context.scene.objects):
    bpy.ops.object.camera_add(location=(5.0, -6.0, 3.2))
if bpy.context.scene.camera is None:
    bpy.context.scene.camera = next(o for o in bpy.context.scene.objects if o.type == "CAMERA")
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 64
bpy.ops.wm.save_as_mainfile(filepath="/workspace/exports/final.blend")
""".strip()

_CLEAN_GN_HEADER = """\
import bpy, bmesh, math, random, colorsys
import colorsys as _colorsys, random as _random
import numpy as np
from mathutils import Vector, Euler, Matrix, Quaternion, kdtree, noise
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.materials): bpy.data.materials.remove(_m)
for _n in list(bpy.data.node_groups): bpy.data.node_groups.remove(_n)
""".strip()

_CLEAN_GN_FOOTER = """\
if not any(o.type == "LIGHT" for o in bpy.context.scene.objects):
    bpy.ops.object.light_add(type="SUN", location=(4.0, -4.0, 6.0))
    bpy.context.active_object.data.energy = 3.0
if not any(o.type == "CAMERA" for o in bpy.context.scene.objects):
    bpy.ops.object.camera_add(location=(5.0, -6.0, 3.2))
if bpy.context.scene.camera is None:
    bpy.context.scene.camera = next(o for o in bpy.context.scene.objects if o.type == "CAMERA")
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 64
bpy.ops.wm.save_as_mainfile(filepath="/workspace/exports/final.blend")
""".strip()

_GEONODE_TASK_SUFFIX = (
    "\nWrite a clean, standalone Blender 5.4 Python (`bpy`) script using Geometry Nodes "
    "(`GeometryNodeTree` with `ng.interface.new_socket`, `ng.nodes.new`, `ng.links.new`, and `NodeGroupOutput`) "
    "to build this procedural object from scratch, add lighting, an active camera, Cycles (64 samples), "
    "and save `/workspace/exports/final.blend`."
)

_STRIP_EXACT_LINES = {
    "import bpy",
    "import bmesh",
    "import math",
    "import random",
    "import numpy as np",
    "from mathutils import Vector, Euler",
}

_STRIP_UNINDENTED_PREFIXES = (
    "bpy.ops.object.select_all(",
    "bpy.ops.object.select_by_type(",
    "bpy.ops.object.delete(",
    "bpy.ops.mesh.delete(",
)

_POLYHAVEN_PREAMBLE = """\
def _proc_asset_bundle(path_hint):
    import os as _os
    base = _os.path.splitext(_os.path.basename(str(path_hint)))[0] or "Asset"
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=0.8)
    me = bpy.data.meshes.new(base + "_Mesh")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(base, me)
    assign_material(ob, make_material(base + "_Mat", (0.45, 0.38, 0.30, 1.0), roughness=0.5))
    return [ob]


class _ProcLibCtx:
    def __init__(self, path_hint, link=False):
        self.df = type("_DF", (), {"objects": ["Asset"], "materials": ["LibMat"], "worlds": ["LibWorld"]})()
        self.dt = type("_DT", (), {"objects": _proc_asset_bundle(path_hint), "materials": [make_material("LibMat")], "worlds": []})()
    def __enter__(self):
        return (self.df, self.dt)
    def __exit__(self, *args):
        return False


def _proc_lib_load(path_hint, link=False):
    return _ProcLibCtx(path_hint, link=link)


def _proc_img_load(*args, **kwargs):
    return bpy.data.images.new("ProcTex", width=64, height=64)


def _proc_exists(path_hint):
    return True


def _proc_listdir(dir_hint):
    return ["diff.png", "nor.png", "rough.png"]
"""


_DEFAULT_TASK_SUFFIX = (
    "\nWrite a complete, standalone Blender 4.2.10 Python (`bpy`) script that builds this entire scene from scratch: "
    "clear existing objects via `for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)` "
    "(do not call `bpy.ops.mesh.delete` or `mode_set` on an empty scene), define `make_material(...)` and "
    "`assign_material(obj, mat)` using Blender 4.2 Principled BSDF names (`'Specular IOR Level'`, `'Coat Weight'`, `'Transmission Weight'`) "
    "and `obj.data.materials.append(mat)`, configure lighting, active camera, and Cycles (`64` samples), "
    "and save `/workspace/exports/final.blend`. Keep the script concise (under 130 lines) in one ```python block."
)

_STUDIO_FOOTER_MARKER = "# --- Studio Materials, Lighting, Camera & Export ---"


def classify_curriculum_example(
    user_body: str,
    full_code: str,
    source: str,
    scene_summary: dict[str, Any] | None = None,
    is_rl_winner: bool = False,
) -> dict[str, Any]:
    """Tag structural Blender API capabilities, complexity score, and curriculum tier (1..4)."""
    tags: list[str] = []
    if "primitive_" in full_code:
        tags.append("primitives")
    if "camera_add" in full_code or "light_add" in full_code:
        tags.append("camera_light")
    if "Principled BSDF" in full_code or "ShaderNodeBsdfPrincipled" in full_code:
        tags.append("principled_bsdf")
    if "for " in user_body or "def " in user_body:
        tags.append("loops_funcs")
    if ".modifiers.new" in full_code:
        tags.append("modifiers")
    if "bmesh." in user_body:
        tags.append("bmesh")
    if ".links.new" in user_body:
        tags.append("node_links")
    if "GeometryNode" in full_code:
        tags.append("geometry_nodes")
    if "shape_key_add" in full_code or "key_blocks" in full_code:
        tags.append("shape_keys")

    n_lines = len([ln for ln in full_code.splitlines() if ln.strip()])
    mesh_count = int((scene_summary or {}).get("mesh_count", 0))
    gn_nodes_total = int((scene_summary or {}).get("geonode_nodes_total", 0))
    complexity = round(
        0.15 * n_lines
        + 3.0 * mesh_count
        + 18.0 * ("geometry_nodes" in tags)
        + 12.0 * ("bmesh" in tags)
        + 10.0 * ("modifiers" in tags)
        + 8.0 * ("node_links" in tags)
        + 5.0 * ("loops_funcs" in tags),
        2,
    )

    if is_rl_winner:
        tier, tname = 4, "T4_BlenderWorld_RL_Winner"
    elif "bmesh" in tags or "modifiers" in tags or "node_links" in tags or source in ("3dcodeverse-2m", "3DCodeEdit"):
        tier, tname = 3, "T3_Procedural_BMesh_Modifiers_Nodes"
    elif mesh_count >= 4 or "loops_funcs" in tags or "shape_keys" in tags or n_lines > 85:
        tier, tname = 2, "T2_MultiPart_Composition_Materials"
    else:
        tier, tname = 1, "T1_Foundations_Primitives_Camera_Light"

    return {
        "curriculum_tier": tier,
        "tier_name": tname,
        "complexity_score": complexity,
        "api_tags": tags,
        "geonode_nodes_total": gn_nodes_total,
    }


def _extract_user_body(adapted_code: str) -> str:
    """Slice out the original script body between top preamble and studio footer."""
    start = 0
    marker = "obj.data.materials.append(mat)"
    if marker in adapted_code:
        start = adapted_code.find(marker) + len(marker)
    end = adapted_code.find(_STUDIO_FOOTER_MARKER)
    if end == -1:
        end = len(adapted_code)
    return adapted_code[start:end].strip()


def compact_geonode_code(raw_text: str) -> str:
    """Compact docstrings, comments, long floats, and editor UI coordinates cleanly for token budget."""
    code = bv.extract_python_code(raw_text).strip()
    if not code:
        return ""
    stripped_doc = re.sub(r'^[ \t]*"""[\s\S]*?"""\n?', "", code, flags=re.MULTILINE)
    try:
        compile(stripped_doc, "<compact>", "exec")
        code = stripped_doc
    except SyntaxError:
        pass
    code = re.sub(r"(-?\d+\.\d{4})\d+", lambda m: str(round(float(m.group(0)), 4)), code)
    lines: list[str] = []
    for ln in code.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        ln = re.sub(r";\s*[A-Za-z0-9_]+\.location\s*=\s*\([^)]+\)", "", ln)
        if re.match(r"^[A-Za-z0-9_]+\.location\s*=\s*\([^)]+\)$", ln.strip()):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def adapt_to_blender42_scene(
    raw_text: str,
    extra_preamble: str = "",
    slim_footer: bool = False,
    clean_wrap: bool = False,
) -> str:
    """Wrap raw bpy code with Blender helpers + studio/export footer (or minimal clean GeoNodes wrapper)."""
    code = compact_geonode_code(raw_text) if (slim_footer or clean_wrap) else bv.extract_python_code(raw_text).strip()
    if not code:
        return ""

    for old, new in _SOCKET_REPLACEMENTS:
        code = code.replace(old, new)

    future_lines: list[str] = []
    body_lines: list[str] = []
    for line in code.splitlines():
        if line.startswith("from __future__ import"):
            future_lines.append(line)
            continue
        if line.strip() in _STRIP_EXACT_LINES or any(line.startswith(pfx) for pfx in _STRIP_UNINDENTED_PREFIXES):
            continue
        body_lines.append(line)

    parts: list[str] = []
    if future_lines:
        parts.append("\n".join(future_lines))
    parts.append(_CLEAN_GN_HEADER if clean_wrap else _TOP_PREAMBLE.strip())
    if extra_preamble.strip():
        parts.append(extra_preamble.strip())
    cleaned_body = "\n".join(body_lines).strip()
    if cleaned_body:
        parts.append(cleaned_body)
    if "/workspace/exports/final.blend" not in cleaned_body and "save_as_mainfile" not in cleaned_body:
        parts.append(_CLEAN_GN_FOOTER if clean_wrap else (_SLIM_GEONODE_FOOTER if slim_footer else _FLAT_FOOTER))
    return "\n\n".join(parts) + "\n"


def _verify_candidate_batch(
    candidates: list[tuple[str, str, str, str, str]],
    target_count: int,
    min_reward: float,
    max_adapted_chars: int = 6800,
    max_workers: int = 8,
    label: str = "src",
    slim_footer: bool = False,
    clean_wrap: bool = False,
) -> list[dict[str, Any]]:
    """Verify adapted candidate scripts in parallel via headless Blender."""
    prepared: list[tuple[str, str, str, str]] = []
    for cid, source, prompt, raw_code, extra_pre in candidates:
        adapted = adapt_to_blender42_scene(raw_code, extra_preamble=extra_pre, slim_footer=slim_footer, clean_wrap=clean_wrap)
        if not adapted or len(adapted) > max_adapted_chars:
            continue
        prepared.append((cid, source, prompt, adapted))

    if not prepared:
        return []

    def _run_one(item: tuple[str, str, str, str]) -> dict[str, Any] | None:
        cid, source, prompt, adapted = item
        vr = bv.verify_script(adapted, task_bundle=None)
        if vr.exec_ok and vr.reward >= min_reward:
            meta = classify_curriculum_example(
                _extract_user_body(adapted),
                adapted,
                source,
                scene_summary=vr.scene_summary,
                is_rl_winner=False,
            )
            return {
                "id": cid,
                "source": source,
                "system": "You are an expert Blender 4.2 Python programmer. Generate clean, self-contained, executable `bpy` code compatible with Blender 4.2.10.",
                "prompt": prompt,
                "response": f"```python\n{adapted.strip()}\n```",
                "reward": vr.reward,
                "r_scene": vr.r_scene,
                "approx_tokens": len(adapted) // 4,
                **meta,
            }
        return None

    verified: list[dict[str, Any]] = []
    tested = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for res in pool.map(_run_one, prepared):
            tested += 1
            if res is not None:
                verified.append(res)
            if len(verified) >= target_count:
                break
    print(f"[prepare {label}] verified={len(verified)} (candidates_in={len(prepared)}, tested={tested})", flush=True)
    return verified


def collect_blendnet(target_count: int = 36, min_reward: float = 0.65) -> list[dict[str, Any]]:
    p = hf_hub_download("FreedomIntelligence/BlendNet", "BlendNet.jsonl", repo_type="dataset")
    candidates: list[tuple[str, str, str, str, str]] = []
    for idx, line in enumerate(Path(p).read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.strip() or len(candidates) >= max(80, target_count * 3):
            break
        obj = json.loads(line)
        prompt = str(obj.get("instruction") or obj.get("prompt") or "").strip()
        inp = str(obj.get("input") or "").strip()
        if inp:
            prompt = f"{prompt}\n{inp}"
        resp = str(obj.get("script") or obj.get("output") or obj.get("code") or obj.get("response") or "").strip()
        if prompt and 350 <= len(resp) <= 4500:
            candidates.append((f"blendnet_{idx}", "BlendNet", prompt, resp, ""))
    return _verify_candidate_batch(candidates, target_count=target_count, min_reward=min_reward, max_adapted_chars=6800, label="BlendNet")


def collect_3dcodeverse(target_count: int = 55, min_reward: float = 0.65) -> list[dict[str, Any]]:
    parquets = (
        "3dcodebench/factories_tex_llamafactory_text/train.parquet",
        "3dcodebench/factories_geo_llamafactory_text/train.parquet",
        "3dcodebench/instances_tex_llamafactory_text/train.parquet",
        "3dcodebench/instances_geo_llamafactory_text/train.parquet",
    )
    candidates: list[tuple[str, str, str, str, str]] = []
    seen_ids: set[str] = set()
    seen_fams: set[str] = set()
    cap = max(140, target_count * 3)
    per_pq_cap = max(30, cap // len(parquets))
    for rel in parquets:
        p = hf_hub_download("3dcodeverse/3dcodeverse-2m", rel, repo_type="dataset")
        t = pq.read_table(p, columns=["id", "conversations"])
        added_here = 0
        for cid, conv in zip(t["id"].to_pylist(), t["conversations"].to_pylist()):
            sid = str(cid)
            if sid in seen_ids or not conv or len(conv) < 2:
                continue
            prompt = str(conv[0].get("value") or "").strip()
            resp = str(conv[1].get("value") or "").strip()
            if not prompt or not (350 <= len(resp) <= 5200):
                continue
            fam = sid.rsplit("/", 1)[-1].split("_")[0].replace("Factory", "")
            if fam in seen_fams:
                continue
            seen_ids.add(sid)
            seen_fams.add(fam)
            candidates.append((sid, "3dcodeverse-2m", prompt, resp, ""))
            added_here += 1
            if added_here >= per_pq_cap or len(candidates) >= cap:
                break
    # Second pass without family dedup if needed, scanning all parquets cleanly
    if len(candidates) < target_count * 2:
        for rel in parquets:
            p = hf_hub_download("3dcodeverse/3dcodeverse-2m", rel, repo_type="dataset")
            t = pq.read_table(p, columns=["id", "conversations"])
            for cid, conv in zip(t["id"].to_pylist(), t["conversations"].to_pylist()):
                sid = str(cid)
                if sid in seen_ids or not conv or len(conv) < 2:
                    continue
                prompt = str(conv[0].get("value") or "").strip()
                resp = str(conv[1].get("value") or "").strip()
                if prompt and 350 <= len(resp) <= 5200:
                    seen_ids.add(sid)
                    candidates.append((sid, "3dcodeverse-2m", prompt, resp, ""))
                if len(candidates) >= cap:
                    break
            if len(candidates) >= cap:
                break
    return _verify_candidate_batch(candidates, target_count=target_count, min_reward=min_reward, max_adapted_chars=7000, label="3dcodeverse")


def _rewrite_polyhaven_body(raw_code: str) -> str:
    txt = raw_code.replace(
        "C:/Users/limbp/Documents/AI_FIRST/tools/blender-4.2.10-windows-x64/data/polyhaven/",
        "/workspace/assets/",
    )
    txt = txt.replace("bpy.data.libraries.load(", "_proc_lib_load(")
    txt = txt.replace("bpy.data.images.load(", "_proc_img_load(")
    txt = txt.replace("os.listdir(", "_proc_listdir(")
    txt = txt.replace("os.path.exists(", "_proc_exists(")
    txt = txt.replace("\n    return\n", "\n    pass\n")
    return txt


def collect_blenderllm_v2(target_count: int = 20, min_reward: float = 0.65) -> list[dict[str, Any]]:
    p = hf_hub_download("LiM-De/blenderllm-v2-polyhaven-dataset", "train.jsonl", repo_type="dataset")
    candidates: list[tuple[str, str, str, str, str]] = []
    seen_instr: set[str] = set()
    for idx, line in enumerate(Path(p).read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        instr = str(obj.get("instruction") or "").strip()
        out = str(obj.get("output") or "").strip()
        if not instr or "Verify the current scene" in instr or "Search for" in instr or instr in seen_instr:
            continue
        seen_instr.add(instr)
        clean_prompt = (
            instr.replace(" with real Poly Haven models and HDRI lighting", "")
            .replace(" from local Poly Haven library", "")
            .replace(" from Poly Haven", "")
            .strip()
        )
        rewritten = _rewrite_polyhaven_body(out)
        if 250 <= len(rewritten) <= 4500:
            candidates.append((f"blenderllm_v2_{idx}", "blenderllm-v2", clean_prompt, rewritten, _POLYHAVEN_PREAMBLE))
        if len(candidates) >= max(80, target_count * 4):
            break
    return _verify_candidate_batch(candidates, target_count=target_count, min_reward=min_reward, max_adapted_chars=7000, label="blenderllm-v2")


def collect_3dcodeedit(target_count: int = 16, min_reward: float = 0.65) -> list[dict[str, Any]]:
    p_src = hf_hub_download("BinjieW/3DCodeEdit", "data/sources.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in Path(p_src).read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    rows.sort(key=lambda r: len(str(r.get("program", ""))))
    candidates: list[tuple[str, str, str, str, str]] = []
    for r in rows:
        prog = str(r.get("program") or "").strip()
        if not (350 <= len(prog) <= 5800):
            continue
        name = str(r.get("name") or r.get("source_id") or "Model")
        desc = str(r.get("description") or "").strip()
        prompt = f"Write Blender Python code to create a 3D '{name}' object: {desc}".strip()
        candidates.append((str(r.get("source_id", name)), "3DCodeEdit", prompt, prog, ""))
        if len(candidates) >= target_count * 2:
            break
    return _verify_candidate_batch(candidates, target_count=target_count, min_reward=min_reward, max_adapted_chars=7400, label="3DCodeEdit")


def _blenderbench_preamble(goal_src: str) -> str:
    """Create stub Blender objects & shape keys referenced by BlenderBench goal.py scripts."""
    names = sorted(set(re.findall(r"objects\[[\"']([^\"']+)[\"']\]", goal_src)))
    sk_names = re.findall(r"key_blocks\[[\"']([^\"']+)[\"']\]", goal_src)
    lines: list[str] = []
    for idx, nm in enumerate(names):
        low = nm.lower()
        if "camera" in low:
            lines.append(f"bpy.ops.object.camera_add(location=(3.0, -4.0, 2.0))\nbpy.context.active_object.name = {nm!r}")
        elif "lamp" in low or "light" in low:
            lines.append(f"bpy.ops.object.light_add(type='POINT', location=({idx * 1.5 - 1.5}, 0.0, 2.5))\nbpy.context.active_object.name = {nm!r}")
        else:
            lines.append(
                f"bpy.ops.mesh.primitive_cube_add(size=0.8, location=({idx * 1.2}, 0.0, 0.5))\n"
                f"_bb_ob = bpy.context.active_object\n_bb_ob.name = {nm!r}\n"
                f"assign_material(_bb_ob, make_material({nm + '_Mat'!r}, (0.6, 0.4, 0.3, 1.0)))"
            )
    if sk_names:
        lines.append("bpy.ops.mesh.primitive_uv_sphere_add(radius=0.9, location=(0.0, 0.0, 1.0))")
        lines.append("_sk_ob = bpy.context.active_object\n_sk_ob.name = 'Character'\nassign_material(_sk_ob, make_material('CharacterMat', (0.75, 0.55, 0.45, 1.0)))")
        lines.append("_sk_ob.shape_key_add(name='Basis')")
        for kb in dict.fromkeys(sk_names):
            lines.append(f"_sk_ob.shape_key_add(name={kb!r})")
    return "\n".join(lines)


def collect_blenderbench(target_count: int = 20, min_reward: float = 0.65) -> list[dict[str, Any]]:
    api = HfApi()
    info = api.dataset_info("DietCoke4671/BlenderBench")
    tasks = sorted(set("/".join(s.rfilename.split("/")[:4]) for s in info.siblings if s.rfilename.endswith("goal.py")))
    candidates: list[tuple[str, str, str, str, str]] = []
    for td in tasks:
        t_txt = Path(hf_hub_download("DietCoke4671/BlenderBench", f"{td}/task.txt", repo_type="dataset")).read_text().strip()
        g_py = Path(hf_hub_download("DietCoke4671/BlenderBench", f"{td}/goal.py", repo_type="dataset")).read_text().strip()
        extra_pre = _blenderbench_preamble(g_py)
        objs = sorted(set(re.findall(r"objects\[[\"']([^\"']+)[\"']\]", g_py)))
        prompt = f"Build a clean 3D room/studio scene containing {', '.join(objs or ['main subject'])}, and apply: {t_txt}"
        tid = "blenderbench_" + "_".join(td.split("/")[-2:])
        candidates.append((tid, "BlenderBench", prompt, g_py, extra_pre))
    return _verify_candidate_batch(candidates, target_count=target_count, min_reward=min_reward, max_adapted_chars=6800, label="BlenderBench")


def round_robin_interleave(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Round-robin interleave verified lists so any prefix (--max-samples N) spans all 5 sources."""
    out: list[dict[str, Any]] = []
    max_len = max((len(g) for g in groups), default=0)
    for i in range(max_len):
        for g in groups:
            if i < len(g):
                out.append(g[i])
    return out


def _exact_token_len(row: dict[str, Any], tok: Any) -> int:
    """Compute exact ChatML token count matching train_lora.format_conversation()."""
    system = "You are an expert Blender 4.2 Python programmer. Generate clean, self-contained, executable `bpy` code compatible with Blender 4.2.10."
    prompt = str(row.get("prompt") or "").strip()
    if "/workspace/exports/final.blend" not in prompt:
        prompt += _DEFAULT_TASK_SUFFIX
    reasoning = str(row.get("reasoning") or "").strip()
    resp = str(row.get("response") or "").strip()
    if reasoning:
        txt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{resp}<|im_end|>\n"
    else:
        txt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{resp}<|im_end|>\n"
    return len(tok.encode(txt, add_special_tokens=False).ids)


def _load_curriculum_ledger(work_dir: Path, tok: Any, max_seq_len: int) -> tuple[dict[str, Any], set[str]]:
    """Load curriculum ledger or seed Cycle 0 from Strat A."""
    ledger_path = work_dir / "curriculum_ledger.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        return ledger, set(ledger.get("seen_sft_ids", []))
    seen_ids: set[str] = set()
    strat_a_path = work_dir / "strat_a_verified.jsonl"
    if strat_a_path.is_file():
        for line in strat_a_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if _exact_token_len(r, tok) <= max_seq_len:
                seen_ids.add(str(r["id"]))
                if len(seen_ids) >= 60:
                    break
    ledger = {"cycles": [{"cycle": 0, "stage": "StratA_SFT", "n_samples": len(seen_ids), "seen_ids": sorted(seen_ids)}], "seen_sft_ids": sorted(seen_ids)}
    return ledger, seen_ids


def _harvest_rl_winners(
    work_dir: Path,
    seen_ids: set[str],
    tok: Any,
    max_seq_len: int,
    min_rl_reward: float = 0.53,
    allow_seen_fallback: bool = False,
) -> list[dict[str, Any]]:
    """Harvest highest-reward execution-verified BlenderWorld RL rollouts per task."""
    rl_best_by_task: dict[str, dict[str, Any]] = {}
    rl_files = sorted(work_dir.glob("rl_round_*.jsonl")) + ([work_dir / "rl_harvested_winners.jsonl"] if (work_dir / "rl_harvested_winners.jsonl").is_file() else [])
    for rf in rl_files:
        for line in rf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("exec_ok") or float(r.get("reward", 0.0)) < min_rl_reward:
                continue
            tid = str(r.get("task_id") or r.get("id"))
            prev = rl_best_by_task.get(tid)
            if prev is None or float(r["reward"]) > float(prev["reward"]):
                rl_best_by_task[tid] = r

    unseen_rows: list[dict[str, Any]] = []
    seen_rows: list[dict[str, Any]] = []
    for tid in sorted(rl_best_by_task.keys()):
        r = rl_best_by_task[tid]
        winner_id = f"rl_{r['id']}"
        code = bv.extract_python_code(str(r.get("response") or "")).strip()
        if not code:
            continue
        meta = classify_curriculum_example(code, code, "BlenderWorld_RL", is_rl_winner=True)
        row = {
            "id": winner_id,
            "task_id": tid,
            "source": "BlenderWorld_RL",
            "system": "You are an expert Blender 4.2 Python programmer. Generate clean, self-contained, executable `bpy` code compatible with Blender 4.2.10.",
            "prompt": str(r["prompt"]),
            "reasoning": str(r.get("reasoning") or ""),
            "response": f"```python\n{code}\n```",
            "reward": float(r["reward"]),
            "weight": 1.4,
            "approx_tokens": len(code) // 4,
            **meta,
        }
        if _exact_token_len(row, tok) <= max_seq_len:
            if winner_id not in seen_ids:
                unseen_rows.append(row)
            elif allow_seen_fallback:
                seen_rows.append(row)
    unseen_rows.sort(key=lambda x: x["complexity_score"])
    if allow_seen_fallback and not unseen_rows:
        seen_rows.sort(key=lambda x: -float(x["reward"]))
        return seen_rows
    return unseen_rows


def _record_curriculum_cycle(
    ledger_path: Path,
    ledger: dict[str, Any],
    seen_ids: set[str],
    batch_out: Path,
    ordered: list[dict[str, Any]],
) -> None:
    """Append batch metadata to curriculum_ledger.json."""
    tier_counts: dict[str, int] = {}
    src_counts: dict[str, int] = {}
    tag_hist: dict[str, int] = {}
    for r in ordered:
        tkey = f"T{r.get('curriculum_tier', '?')}"
        tier_counts[tkey] = tier_counts.get(tkey, 0) + 1
        src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
        for tg in r.get("api_tags", []):
            tag_hist[tg] = tag_hist.get(tg, 0) + 1

    cycle_num = len(ledger.get("cycles", []))
    new_ids = [str(r["id"]) for r in ordered]
    ledger["seen_sft_ids"] = sorted(seen_ids | set(new_ids))
    ledger.setdefault("cycles", []).append({
        "cycle": cycle_num,
        "batch_path": str(batch_out),
        "n_samples": len(ordered),
        "tier_counts": tier_counts,
        "source_counts": src_counts,
        "api_tag_counts": tag_hist,
        "complexity_range": [
            min((r["complexity_score"] for r in ordered), default=0.0),
            max((r["complexity_score"] for r in ordered), default=0.0),
        ],
        "selected_ids": new_ids,
    })
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"[curriculum] built cycle {cycle_num} batch ({len(ordered)} samples): tiers={tier_counts} sources={src_counts} -> {batch_out}")
    print(f"[curriculum] updated ledger -> {ledger_path}")


def collect_geonodes(
    target_count: int = 60,
    min_reward: float = 0.65,
    max_seq_len: int = 3584,
    max_raw_chars: int = 16000,
    clean_wrap: bool = False,
    exclude_families: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Curate diverse, execution-verified Geometry Nodes scripts from 3dcodeverse-2m & 3DCodeEdit."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download("ccore/Qwen3.5-4B-3D-Coder", "tokenizer.json", local_files_only=True))
    raw_items: list[tuple[str, str, str, str, str, str]] = []
    seen_cids: set[str] = set()
    excl = exclude_families or set()

    parquets = (
        "3dcodebench/instances_geo_llamafactory_text/train.parquet",
        "3dcodebench/instances_tex_llamafactory_text/train.parquet",
        "3dcodebench/factories_geo_llamafactory_text/train.parquet",
        "3dcodebench/factories_tex_llamafactory_text/train.parquet",
    )
    for rel in parquets:
        p = hf_hub_download("3dcodeverse/3dcodeverse-2m", rel, repo_type="dataset")
        t = pq.read_table(p, columns=["id", "conversations"])
        variant = "tex" if "_tex" in rel else "geo"
        for cid, conv in zip(t["id"].to_pylist(), t["conversations"].to_pylist()):
            sid = str(cid)
            if sid in seen_cids or not conv or len(conv) < 2:
                continue
            fam = sid.rsplit("/", 1)[-1].split("_")[0].replace("Factory", "")
            if fam in excl:
                continue
            prompt = str(conv[0].get("value") or "").strip()
            resp = str(conv[1].get("value") or "").strip()
            if "GeometryNode" in resp and prompt and 400 <= len(resp) <= max_raw_chars:
                seen_cids.add(sid)
                if clean_wrap and "/workspace/exports/final.blend" not in prompt:
                    prompt = prompt + _GEONODE_TASK_SUFFIX
                raw_items.append((fam, variant, sid, "3dcodeverse-2m", prompt, resp))

    p_src = hf_hub_download("BinjieW/3DCodeEdit", "data/sources.jsonl", repo_type="dataset")
    for line in Path(p_src).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prog = str(r.get("program") or "").strip()
        if "GeometryNode" not in prog or not (400 <= len(prog) <= max_raw_chars):
            continue
        sid_raw = str(r.get("source_id") or r.get("name") or len(raw_items))
        fam = sid_raw.replace("3dcodebench_", "").split("_")[0].replace("Factory", "")
        if fam in excl:
            continue
        cid = f"3dcodeedit_{sid_raw}"
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        name = str(r.get("name") or sid_raw)
        desc = str(r.get("description") or "").strip()
        prompt = f"Write Blender Python code to create a 3D '{name}' object: {desc}".strip()
        if clean_wrap:
            prompt = prompt + _GEONODE_TASK_SUFFIX
        raw_items.append((fam, "edit", cid, "3DCodeEdit", prompt, prog))

    # Sort ascending by length so compact graphs (most likely under max_seq_len) are prioritized cleanly
    raw_items.sort(key=lambda x: len(x[5]))

    candidates: list[tuple[str, str, str, str, str]] = []
    chosen_ids: set[str] = set()
    seen_fam_var: set[tuple[str, str]] = set()
    # Pass 1: up to 1 geo and 1 tex (or edit) per distinct family
    for fam, var, cid, src, prompt, resp in raw_items:
        key = (fam, var)
        if key in seen_fam_var:
            continue
        seen_fam_var.add(key)
        chosen_ids.add(cid)
        candidates.append((cid, src, prompt, resp, ""))
        if len(candidates) >= 180:
            break

    # Pass 2: up to 6 total per (fam, var) up to 180 candidates
    if len(candidates) < 180:
        counts_fv: dict[tuple[str, str], int] = {k: 1 for k in seen_fam_var}
        for fam, var, cid, src, prompt, resp in raw_items:
            if cid in chosen_ids:
                continue
            key = (fam, var)
            if counts_fv.get(key, 0) >= 6:
                continue
            counts_fv[key] = counts_fv.get(key, 0) + 1
            chosen_ids.add(cid)
            candidates.append((cid, src, prompt, resp, ""))
            if len(candidates) >= 180:
                break

    verified = _verify_candidate_batch(
        candidates,
        target_count=target_count * 2,
        min_reward=min_reward,
        max_adapted_chars=17500,
        label="GeoNodes",
        slim_footer=not clean_wrap,
        clean_wrap=clean_wrap,
    )
    filtered: list[dict[str, Any]] = []
    for r in verified:
        if r.get("geonode_nodes_total", 0) < 3:
            continue
        if _exact_token_len(r, tok) <= max_seq_len:
            filtered.append(r)
        if len(filtered) >= target_count:
            break
    print(f"[prepare GeoNodes] kept={len(filtered)} under max_seq_len={max_seq_len} (from {len(verified)} verified)", flush=True)
    return filtered


def build_clean_geonodes_batch(
    batch_out: Path,
    pool_out: Path,
    max_seq_len: int = 3584,
    batch_size: int = 40,
) -> list[dict[str, Any]]:
    """Curate bloat-free Blender 5.4 Geometry Nodes scripts excluding held-out eval families."""
    held_out = {"GlobularBaseCactus", "FruitApple", "SpiralStaircase", "SnakePlant", "Vase", "Tap"}
    gn_rows = collect_geonodes(
        target_count=80,
        min_reward=0.65,
        max_seq_len=max_seq_len,
        clean_wrap=True,
        exclude_families=held_out,
    )
    pool_out.parent.mkdir(parents=True, exist_ok=True)
    with pool_out.open("w", encoding="utf-8") as f:
        for r in gn_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Round-robin across families sorted easy->hard by (geonode_nodes_total, complexity_score)
    by_fam: dict[str, list[dict[str, Any]]] = {}
    for r in gn_rows:
        fam = str(r["id"]).replace("3dcodeedit_3dcodebench_", "").rsplit("/", 1)[-1].split("_")[0].replace("Factory", "")
        by_fam.setdefault(fam, []).append(r)
    for fam in by_fam:
        by_fam[fam].sort(key=lambda r: (r.get("geonode_nodes_total", 0), r["complexity_score"]))

    picked: list[dict[str, Any]] = []
    fam_order = sorted(by_fam.keys(), key=lambda k: (by_fam[k][0].get("geonode_nodes_total", 0), by_fam[k][0]["complexity_score"]))
    idx = 0
    while len(picked) < batch_size:
        added_any = False
        for fam in fam_order:
            lst = by_fam[fam]
            if idx < len(lst):
                picked.append(lst[idx])
                added_any = True
                if len(picked) >= batch_size:
                    break
        if not added_any:
            break
        idx += 1

    picked.sort(key=lambda r: (r.get("geonode_nodes_total", 0), r["complexity_score"]))
    batch_out.parent.mkdir(parents=True, exist_ok=True)
    with batch_out.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[clean GeoNodes] built batch ({len(picked)} samples across {len(by_fam)} families) -> {batch_out}", flush=True)
    return picked


def build_geonodes_curriculum_batch(
    batch_out: Path,
    pool_out: Path,
    max_seq_len: int = 3584,
    batch_size: int = 36,
    anchor_cap: int = 8,
) -> list[dict[str, Any]]:
    """Curate Geometry Nodes scripts and build an ordered curriculum SFT batch with studio/RL anchors."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download("ccore/Qwen3.5-4B-3D-Coder", "tokenizer.json", local_files_only=True))
    work_dir = SCRIPT_DIR / "work"
    ledger_path = work_dir / "curriculum_ledger.json"
    ledger, seen_ids = _load_curriculum_ledger(work_dir, tok, max_seq_len)

    gn_rows = collect_geonodes(target_count=56, min_reward=0.65, max_seq_len=max_seq_len)
    pool_out.parent.mkdir(parents=True, exist_ok=True)
    with pool_out.open("w", encoding="utf-8") as f:
        for r in gn_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prepare GeoNodes] saved pool ({len(gn_rows)}) -> {pool_out}", flush=True)

    # Select diverse families ordered easy-to-hard by (geonode_nodes_total, complexity_score)
    gn_sorted = sorted(
        gn_rows,
        key=lambda r: (r["id"] in seen_ids, r.get("geonode_nodes_total", 0), r["complexity_score"]),
    )
    gn_quota = max(1, batch_size - anchor_cap)
    picked_gn: list[dict[str, Any]] = []
    picked_ids: set[str] = set()
    seen_fams: set[str] = set()
    for r in gn_sorted:
        fam = str(r["id"]).replace("3dcodeedit_3dcodebench_", "").rsplit("/", 1)[-1].split("_")[0].replace("Factory", "")
        if fam in seen_fams:
            continue
        seen_fams.add(fam)
        picked_ids.add(r["id"])
        picked_gn.append(r)
        if len(picked_gn) >= gn_quota:
            break
    for r in gn_sorted:
        if len(picked_gn) >= gn_quota:
            break
        if r["id"] not in picked_ids:
            picked_ids.add(r["id"])
            picked_gn.append(r)
    picked_gn.sort(key=lambda r: (r.get("geonode_nodes_total", 0), r["complexity_score"]))

    # Blend multi-part studio anchors (half unseen curriculum_pool studio, half top BlenderWorld RL winners)
    rl_need = min(4, anchor_cap)
    t4_all = _harvest_rl_winners(work_dir, seen_ids, tok, max_seq_len, min_rl_reward=0.53, allow_seen_fallback=True)
    rl_anchors = sorted(t4_all, key=lambda x: -float(x["reward"]))[:rl_need]
    rl_anchors.sort(key=lambda x: x["complexity_score"])

    studio_need = max(0, anchor_cap - len(rl_anchors))
    studio_anchors: list[dict[str, Any]] = []
    curr_pool_path = work_dir / "curriculum_pool.jsonl"
    if studio_need > 0 and curr_pool_path.is_file():
        pool_items = [json.loads(l) for l in curr_pool_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        studio_cands = [
            r for r in pool_items
            if r.get("source") in ("BlendNet", "blenderllm-v2", "BlenderBench")
            and r["id"] not in picked_ids
            and _exact_token_len(r, tok) <= max_seq_len
        ]
        studio_cands.sort(key=lambda r: (r["id"] in seen_ids, -float(r.get("r_scene", 0.0)), -float(r.get("reward", 0.0))))
        studio_anchors = studio_cands[:studio_need]

    ordered = studio_anchors + picked_gn + rl_anchors
    batch_out.parent.mkdir(parents=True, exist_ok=True)
    with batch_out.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _record_curriculum_cycle(ledger_path, ledger, seen_ids, batch_out, ordered)
    return ordered


def build_curriculum_batch(
    pool_rows: list[dict[str, Any]],
    batch_out: Path,
    max_seq_len: int = 2350,
    batch_size: int = 36,
    t12_cap: int = 8,
    min_rl_reward: float = 0.53,
) -> list[dict[str, Any]]:
    """Build an explicit easy-to-hard T1->T2->T3->T4 SFT batch from unseen pool items + harvested RL winners."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download("ccore/Qwen3.5-4B-3D-Coder", "tokenizer.json", local_files_only=True))
    work_dir = SCRIPT_DIR / "work"
    ledger_path = work_dir / "curriculum_ledger.json"
    ledger, seen_ids = _load_curriculum_ledger(work_dir, tok, max_seq_len)

    t4_candidates = _harvest_rl_winners(work_dir, seen_ids, tok, max_seq_len, min_rl_reward=min_rl_reward)

    # Filter unseen pool rows that fit completely under max_seq_len
    unseen = [r for r in pool_rows if str(r["id"]) not in seen_ids and _exact_token_len(r, tok) <= max_seq_len]
    t12 = sorted([r for r in unseen if r.get("curriculum_tier", 1) in (1, 2)], key=lambda x: (x["curriculum_tier"], x["complexity_score"]))
    t3 = sorted([r for r in unseen if r.get("curriculum_tier", 1) == 3], key=lambda x: x["complexity_score"])

    picked_t12 = t12[:t12_cap]
    remaining_for_t3 = max(0, batch_size - len(picked_t12) - len(t4_candidates))
    picked_t3 = t3[:remaining_for_t3]

    used_ids = {r["id"] for r in picked_t12 + picked_t3}
    ordered = picked_t12 + picked_t3
    if len(ordered) + len(t4_candidates) < batch_size:
        extras = [r for r in unseen if r["id"] not in used_ids]
        need = batch_size - len(ordered) - len(t4_candidates)
        ordered.extend(extras[:need])
    ordered.extend(t4_candidates)

    batch_out.parent.mkdir(parents=True, exist_ok=True)
    with batch_out.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _record_curriculum_cycle(ledger_path, ledger, seen_ids, batch_out, ordered)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "work" / "curriculum_pool.jsonl")
    parser.add_argument("--build-cycle-batch", type=Path, default=None, help="Optional ordered T1->T4 curriculum SFT batch JSONL.")
    parser.add_argument("--build-geonodes-batch", type=Path, default=None, help="Curate Geometry Nodes dataset and build ordered SFT batch JSONL.")
    parser.add_argument("--build-clean-geonodes-batch", type=Path, default=None, help="Curate minimal-wrapper Geometry Nodes SFT batch excluding eval families.")
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--max-seq-len", type=int, default=2350)
    parser.add_argument("--target-blendnet", type=int, default=70)
    parser.add_argument("--target-3dcodeverse", type=int, default=55)
    parser.add_argument("--target-blenderllm-v2", type=int, default=28)
    parser.add_argument("--target-3dcodeedit", type=int, default=6)
    parser.add_argument("--target-blenderbench", type=int, default=22)
    parser.add_argument("--min-reward", type=float, default=0.65)
    args = parser.parse_args()

    if args.build_clean_geonodes_batch is not None:
        rows = build_clean_geonodes_batch(
            args.build_clean_geonodes_batch,
            SCRIPT_DIR / "work" / "geonodes_clean_pool.jsonl",
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
        return 0 if rows else 1

    if args.build_geonodes_batch is not None:
        rows = build_geonodes_curriculum_batch(
            args.build_geonodes_batch,
            SCRIPT_DIR / "work" / "geonodes_verified_pool.jsonl",
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
        return 0 if rows else 1

    groups = [
        collect_blendnet(args.target_blendnet, args.min_reward),
        collect_3dcodeverse(args.target_3dcodeverse, args.min_reward),
        collect_blenderllm_v2(args.target_blenderllm_v2, args.min_reward),
        collect_3dcodeedit(args.target_3dcodeedit, args.min_reward),
        collect_blenderbench(args.target_blenderbench, args.min_reward),
    ]
    rows = round_robin_interleave(groups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    tier_counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
        tk = f"T{r.get('curriculum_tier', '?')}"
        tier_counts[tk] = tier_counts.get(tk, 0) + 1
    print(f"[prepare] total={len(rows)} by_source={counts} by_tier={tier_counts} -> {args.out}")

    if args.build_cycle_batch is not None:
        build_curriculum_batch(rows, args.build_cycle_batch, max_seq_len=args.max_seq_len, batch_size=args.batch_size)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
