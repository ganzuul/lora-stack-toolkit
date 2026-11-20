#!/usr/bin/env python3
"""Run a Blender Geometry Nodes script headless and print a strict verdict.

Usage: gn_check.py /path/to/script.py [--blender /path/to/blender] [--timeout 120]

Prints `VERIFIED: ...` and exits 0 only when the script ran cleanly, Blender
emitted no WARNING/ERROR lines, and every GeometryNodeTree has a linked group
output with valid links and evaluated geometry. Anything else prints lines that
start with `Error:` (so chat-template tool-failure heuristics fire) and exits 1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "/home/nos/blender-5.4.0/blender")

HARNESS = r'''
import json, os, sys, traceback
import bpy
cand = os.environ["GN_CAND"]; out = os.environ["GN_OUT"]
ns = {"__name__": "__main__", "__file__": cand}
try:
    exec(compile(open(cand, encoding="utf-8").read(), cand, "exec"), ns, ns)
except SystemExit as se:
    if se.code not in (0, None):
        traceback.print_exc(); sys.exit(1)
except Exception:
    traceback.print_exc(); sys.exit(1)
trees = []
for ng in bpy.data.node_groups:
    if ng.bl_idname != "GeometryNodeTree":
        continue
    outs = [n for n in ng.nodes if n.bl_idname == "NodeGroupOutput"]
    bad = [f"{l.from_node.name}.{l.from_socket.name}->{l.to_node.name}.{l.to_socket.name}" for l in ng.links if not l.is_valid]
    reaches = any(s.is_linked for o in outs for s in o.inputs if s.type == "GEOMETRY")
    iface_out = [i.name for i in ng.interface.items_tree if getattr(i, "in_out", "") == "OUTPUT"]
    trees.append({"name": ng.name, "nodes": len(ng.nodes), "links": len(ng.links), "group_outputs": len(outs),
                  "interface_outputs": iface_out, "invalid_links": bad, "reaches_output": reaches})
deps = bpy.context.evaluated_depsgraph_get()
verts = 0; mats = set(); bound = 0
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        if any(m.type == "NODES" and m.node_group for m in obj.modifiers):
            bound += 1
        ev = obj.evaluated_get(deps); me = ev.to_mesh()
        try:
            verts += len(me.vertices)
            for m in me.materials:
                if m: mats.add(m.name)
        finally:
            ev.to_mesh_clear()
json.dump({"trees": trees, "verts": verts, "materials": sorted(mats), "modifier_bound": bound,
           "lights": sum(1 for o in bpy.context.scene.objects if o.type == "LIGHT"),
           "camera": bpy.context.scene.camera is not None}, open(out, "w"))
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script", type=Path)
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    if not args.script.is_file():
        print(f"Error: script not found: {args.script}")
        return 1
    try:
        compile(args.script.read_text(encoding="utf-8"), str(args.script), "exec")
    except SyntaxError as exc:
        print(f"Error: SyntaxError in {args.script}: {exc}")
        return 1

    with tempfile.TemporaryDirectory(prefix="gn_check_") as tmp:
        harness = Path(tmp) / "harness.py"
        out_json = Path(tmp) / "summary.json"
        harness.write_text(HARNESS, encoding="utf-8")
        env = os.environ.copy()
        env["GN_CAND"] = str(args.script.resolve())
        env["GN_OUT"] = str(out_json)
        try:
            proc = subprocess.run(
                [args.blender, "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(harness)],
                env=env, capture_output=True, text=True, timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"Error: Blender timed out after {args.timeout}s")
            return 1

        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        problems: list[str] = []
        if proc.returncode != 0 or not out_json.is_file():
            tail = "\n".join(text.strip().splitlines()[-15:])
            print(f"Error: script failed (exit {proc.returncode}):\n{tail}")
            return 1
        for line in text.splitlines():
            if "WARNING" in line or " ERROR" in line or line.startswith("Error"):
                problems.append(line.strip())

        s = json.loads(out_json.read_text(encoding="utf-8"))
        if not s["trees"]:
            problems.append("no GeometryNodeTree found in bpy.data.node_groups")
        for t in s["trees"]:
            tag = f"tree '{t['name']}'"
            if t["group_outputs"] == 0:
                problems.append(f"{tag} has no NodeGroupOutput node (add ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry') and ng.nodes.new('NodeGroupOutput'))")
            elif not t["reaches_output"]:
                problems.append(f"{tag}: nothing is linked into the NodeGroupOutput Geometry input")
            if not t["interface_outputs"]:
                problems.append(f"{tag} has no OUTPUT interface socket")
            if t["invalid_links"]:
                problems.append(f"{tag} invalid links: {t['invalid_links'][:5]}")
        if s["modifier_bound"] == 0:
            problems.append("no object has a NODES modifier bound to a node group")
        if s["verts"] < 32:
            problems.append(f"evaluated geometry has only {s['verts']} verts")

        nodes = sum(t["nodes"] for t in s["trees"])
        links = sum(t["links"] for t in s["trees"])
        summary = (f"trees={len(s['trees'])} nodes={nodes} links={links} verts={s['verts']} "
                   f"materials={len(s['materials'])} lights={s['lights']} camera={s['camera']}")
        if problems:
            print("Error: GN check FAILED - " + summary)
            for p in dict.fromkeys(problems):
                print(f"Error: {p}")
            return 1
        print(f"VERIFIED: valid=True reaches_output=True {summary}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
