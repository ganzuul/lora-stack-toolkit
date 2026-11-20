#!/usr/bin/env python3
"""Layer 1 SME Mini-Framework: Live Blender 5.4 GeometryNodes Schema Extractor & Static AST Linter."""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BLENDER_BIN = Path(os.environ.get("BLENDER_BIN", str(SCRIPT_DIR / "blender")))
DEFAULT_SCHEMA_JSON = SCRIPT_DIR / "work" / "blender_5_4_gn_schema.json"

DYNAMIC_SOCKET_NODES = frozenset({"NodeGroupInput", "NodeGroupOutput", "GeometryNodeGroup", "NodeReroute"})
BANNED_HELPER_NAMES = frozenset({"pbr_mat", "make_material", "assign_material", "make_pbr_mat"})

_INTROSPECT_EXPR = r"""
import bpy, json
ng = bpy.data.node_groups.new("_introspect", "GeometryNodeTree")
nodes = {}
extra = {"NodeGroupInput", "NodeGroupOutput", "NodeReroute", "NodeFrame"}
for name in sorted(dir(bpy.types)):
    if name.startswith(("GeometryNode", "ShaderNode", "FunctionNode")) or name in extra:
        try:
            n = ng.nodes.new(name)
            nodes[name] = {
                "inputs": [[s.name, s.type] for s in n.inputs],
                "outputs": [[s.name, s.type] for s in n.outputs],
            }
            ng.nodes.remove(n)
        except Exception:
            pass
payload = {
    "blender_version": bpy.app.version_string,
    "node_count": len(nodes),
    "nodes": nodes,
}
print("__SCHEMA_JSON_BEGIN__")
print(json.dumps(payload, separators=(",", ":")))
print("__SCHEMA_JSON_END__")
"""


def extract_schema(
    blender_bin: Path = DEFAULT_BLENDER_BIN,
    out_json: Path = DEFAULT_SCHEMA_JSON,
) -> dict[str, Any]:
    """Introspect installed headless Blender for valid GeometryNodeTree nodes and sockets."""
    if not Path(blender_bin).exists():
        raise FileNotFoundError(f"Blender binary not found: {blender_bin}")
    proc = subprocess.run(
        [str(blender_bin), "--background", "--factory-startup", "--python-expr", _INTROSPECT_EXPR],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    txt = proc.stdout
    b_idx = txt.find("__SCHEMA_JSON_BEGIN__")
    e_idx = txt.find("__SCHEMA_JSON_END__")
    if b_idx < 0 or e_idx <= b_idx:
        raise RuntimeError(f"Failed to parse schema JSON from Blender stdout:\n{txt[-1000:]}")
    raw = txt[b_idx + len("__SCHEMA_JSON_BEGIN__") : e_idx].strip()
    schema = json.loads(raw)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema


def load_schema(path: Path = DEFAULT_SCHEMA_JSON, blender_bin: Path = DEFAULT_BLENDER_BIN) -> dict[str, Any]:
    """Load cached Blender 5.4 node/socket schema JSON (extracting automatically if absent)."""
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return extract_schema(blender_bin=blender_bin, out_json=path)


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_nodes_new_call(node: ast.AST) -> str | None:
    """Return node_bl_idname if AST node is `<expr>.nodes.new("<id>")`, else None."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != "new":
        return None
    owner = node.func.value
    if not isinstance(owner, ast.Attribute) or owner.attr != "nodes":
        return None
    if node.args:
        return _literal_str(node.args[0])
    for kw in node.keywords:
        if kw.arg == "type":
            return _literal_str(kw.value)
    return None


def lint_geonodes_script(
    code: str,
    schema: dict[str, Any],
    enforce_canonical_boundary: bool = True,
) -> list[str]:
    """Statically validate GeometryNode IDs, socket pin names, and canonical Blender 5.4 boundary."""
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]

    nodes_catalog: dict[str, Any] = schema.get("nodes", {})
    # Map variable name -> node bl_idname when assigned in script (conservatively invalidated on overwrite).
    var_to_node_type: dict[str, str] = {}

    for node in ast.walk(tree):
        # Record `var = ...nodes.new("...")` bindings
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            ntype = _is_nodes_new_call(node.value)
            vname = node.targets[0].id
            if ntype is not None:
                prev = var_to_node_type.get(vname)
                if prev is not None and prev != ntype:
                    # Variable reused across different node types; mark polymorphic to avoid false pin checks.
                    var_to_node_type[vname] = "*"
                else:
                    var_to_node_type[vname] = ntype

        # Check all `nodes.new("<id>")` calls against schema
        new_id = _is_nodes_new_call(node)
        if new_id is not None and new_id not in nodes_catalog:
            errors.append(f"L{getattr(node, 'lineno', 0)}: Unknown GeometryNodes type '{new_id}'")

        # Check `var.inputs["<pin>"]` and `var.outputs["<pin>"]`
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            io_kind = node.value.attr
            if io_kind in ("inputs", "outputs") and isinstance(node.value.value, ast.Name):
                pin_name = _literal_str(node.slice)
                vname = node.value.value.id
                ntype = var_to_node_type.get(vname)
                if pin_name is not None and ntype and ntype != "*" and ntype not in DYNAMIC_SOCKET_NODES:
                    spec = nodes_catalog.get(ntype)
                    if spec is not None:
                        valid_pins = {p[0] for p in spec.get(io_kind, [])}
                        if pin_name not in valid_pins:
                            errors.append(
                                f"L{getattr(node, 'lineno', 0)}: Node '{ntype}' has no {io_kind[:-1]} pin '{pin_name}'"
                            )

        if enforce_canonical_boundary:
            if isinstance(node, ast.FunctionDef) and node.name in BANNED_HELPER_NAMES:
                errors.append(f"L{node.lineno}: Non-canonical helper definition '{node.name}' (must use make_mat)")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_HELPER_NAMES:
                errors.append(f"L{node.lineno}: Non-canonical helper call '{node.func.id}' (must use make_mat)")
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr == "use_nodes":
                        errors.append(f"L{node.lineno}: Deprecated Blender <5 assignment '.use_nodes = ...'")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "4.2" in node.value:
                errors.append(f"L{getattr(node, 'lineno', 0)}: Legacy version string '4.2' in string literal")

    return errors


def _selftest(schema: dict[str, Any]) -> None:
    good = """
import bpy
def make_mat(name, color, rough=0.45, metallic=0.0):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    return mat
ng = bpy.data.node_groups.new("Ok", "GeometryNodeTree")
sp = ng.nodes.new("GeometryNodeCurveSpiral")
sp.inputs["Rotations"].default_value = 4.0
ci = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
ci.inputs["Radius"].default_value = 0.2
"""
    errs_good = lint_geonodes_script(good, schema, enforce_canonical_boundary=True)
    assert not errs_good, f"Expected 0 errors on good script, got: {errs_good}"

    bad = """
import bpy
def pbr_mat(mat_name, rgb, roughness=0.4, metal=0.0):
    m = bpy.data.materials.new(mat_name)
    m.use_nodes = True
    return m
ver = "4.2.10"
ng = bpy.data.node_groups.new("Bad", "GeometryNodeTree")
bad_c = ng.nodes.new("GeometryNodeCurveCircle")
sp = ng.nodes.new("GeometryNodeCurveSpiral")
sp.inputs["Sweep Angle"].default_value = 6.28
sp.inputs["Middle Radius"].default_value = 0.5
"""
    errs_bad = lint_geonodes_script(bad, schema, enforce_canonical_boundary=True)
    joined = "\n".join(errs_bad)
    for expected in (
        "Non-canonical helper definition 'pbr_mat'",
        "Deprecated Blender <5 assignment '.use_nodes = ...'",
        "Legacy version string '4.2'",
        "Unknown GeometryNodes type 'GeometryNodeCurveCircle'",
        "no input pin 'Sweep Angle'",
        "no input pin 'Middle Radius'",
    ):
        assert expected in joined, f"Missing expected lint '{expected}' in:\n{joined}"
    print("SCHEMA SELFTEST PASSED")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blender", type=Path, default=DEFAULT_BLENDER_BIN)
    ap.add_argument("--out", type=Path, default=DEFAULT_SCHEMA_JSON)
    ap.add_argument("--extract", action="store_true", help="Extract live Blender 5.4 schema to JSON.")
    ap.add_argument("--selftest", action="store_true", help="Run AST linter selftest.")
    args = ap.parse_args()

    if args.extract or not args.out.is_file():
        schema = extract_schema(blender_bin=args.blender, out_json=args.out)
        print(f"Extracted Blender schema ({schema['blender_version']}): {schema['node_count']} nodes -> {args.out}")
    else:
        schema = load_schema(args.out, blender_bin=args.blender)

    if args.selftest:
        _selftest(schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
