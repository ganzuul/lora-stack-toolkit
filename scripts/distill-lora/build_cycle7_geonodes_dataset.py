#!/usr/bin/env python3
"""Cycle 7 Dataset Builder: Zero-Crutch Bare-Prompt, Pi-Agent Tool-Use, & Multi-Part Radial GeoNodes SME."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
import statistics
import sys
import textwrap
from pathlib import Path
from typing import Any

import jinja2

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
import build_cycle6_geonodes_dataset as bc6  # noqa: E402
import train_lora as tlora  # noqa: E402

CHAT_TEMPLATE_PATH = Path(os.environ.get(
    "QWEN_CHAT_TEMPLATE",
    str(SCRIPT_DIR.parent / "qwen_chat_template.jinja"),
))

C7_FORBIDDEN_TERMS = tuple(bc6.EVAL_FORBIDDEN_TERMS) + (
    "chandelier",
    "candelabra",
    "ferris",
    "gazebo",
)

PI_SYSTEM = (
    "You are an expert coding assistant operating inside pi, a coding agent harness. "
    "Use tools to write clean executable files, verify them via bash, make targeted edits on errors, "
    "and unload specialist LoRAs when finished."
)

PI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file on disk.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file by replacing exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "oldText": {"type": "string"},
                    "newText": {"type": "string"},
                },
                "required": ["path", "oldText", "newText"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file, creating parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

_RADIAL_PREAMBLE = """\
import bpy

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
for m in list(bpy.data.materials):
    bpy.data.materials.remove(m)
for g in list(bpy.data.node_groups):
    bpy.data.node_groups.remove(g)

def make_mat(name, color, rough=0.45, metallic=0.0):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    return mat
"""

_RADIAL_FOOTER = """\
obj = bpy.data.objects.new("{obj_name}", bpy.data.meshes.new("{mesh_name}"))
bpy.context.scene.collection.objects.link(obj)
obj.modifiers.new("{mod_name}", "NODES").node_group = ng
bpy.ops.object.light_add(type="SUN", location=(4.0, -4.0, 6.0))
bpy.ops.object.camera_add(location=(3.2, -3.2, 2.2), rotation=(1.1, 0.0, 0.78))
bpy.context.scene.camera = bpy.context.active_object
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 64
bpy.ops.wm.save_as_mainfile(filepath="/workspace/exports/final.blend")
"""


def _wrap_radial_script(body: str, obj_name: str, mod_name: str) -> str:
    clean_body = textwrap.dedent(body).strip()
    footer = _RADIAL_FOOTER.format(obj_name=obj_name, mesh_name=f"{obj_name}Mesh", mod_name=mod_name)
    return f"{_RADIAL_PREAMBLE.strip()}\n\n{clean_body}\n\n{footer.strip()}\n"


def _build_radial_recipes() -> list[dict[str, Any]]:
    """Construct 16 diverse multi-part radial & hierarchical Geometry Nodes assembly recipes."""
    specs = [
        (
            "c7_rad_01_sconce_crown",
            "SconceCrown",
            "Build a wrought-iron circular wall-ring sconce crown with 6 curved radial scroll arms, cream wax tapers, and glowing brass cup collars.",
            """\
            m_iron = make_mat("WroughtIron", (0.14, 0.14, 0.16), rough=0.42, metallic=0.88)
            m_wax = make_mat("CreamWax", (0.94, 0.90, 0.80), rough=0.48, metallic=0.0)
            m_brass = make_mat("WarmBrass", (0.85, 0.65, 0.22), rough=0.22, metallic=0.95)
            ng = bpy.data.node_groups.new("SconceCrown", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ring_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ring_p.inputs["Resolution"].default_value = 48; ring_p.inputs["Radius"].default_value = 0.68
            ring_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ring_prof.inputs["Resolution"].default_value = 10; ring_prof.inputs["Radius"].default_value = 0.032
            ring_m = ng.nodes.new("GeometryNodeCurveToMesh"); ring_m.inputs["Fill Caps"].default_value = True
            t_ring = ng.nodes.new("GeometryNodeTransform"); t_ring.inputs["Translation"].default_value = (0.0, 0.0, 0.48)
            sm_ring = ng.nodes.new("GeometryNodeSetMaterial"); sm_ring.inputs["Material"].default_value = m_iron
            arm_bz = ng.nodes.new("GeometryNodeCurveQuadraticBezier"); arm_bz.inputs["Resolution"].default_value = 14
            arm_bz.inputs["Start"].default_value = (-0.68, 0.0, 0.0); arm_bz.inputs["Middle"].default_value = (-0.34, 0.0, -0.22); arm_bz.inputs["End"].default_value = (0.0, 0.0, 0.0)
            arm_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); arm_prof.inputs["Resolution"].default_value = 8; arm_prof.inputs["Radius"].default_value = 0.018
            arm_m = ng.nodes.new("GeometryNodeCurveToMesh"); arm_m.inputs["Fill Caps"].default_value = True
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 6
            inst_arm = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_arm = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_arm = ng.nodes.new("GeometryNodeSetMaterial"); sm_arm.inputs["Material"].default_value = m_iron
            cup = ng.nodes.new("GeometryNodeMeshCylinder"); cup.inputs["Vertices"].default_value = 16; cup.inputs["Radius"].default_value = 0.055; cup.inputs["Depth"].default_value = 0.04
            t_cup = ng.nodes.new("GeometryNodeTransform"); t_cup.inputs["Translation"].default_value = (0.0, 0.0, 0.02)
            inst_cup = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_cup = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_cup = ng.nodes.new("GeometryNodeSetMaterial"); sm_cup.inputs["Material"].default_value = m_brass
            candle = ng.nodes.new("GeometryNodeMeshCylinder"); candle.inputs["Vertices"].default_value = 14; candle.inputs["Radius"].default_value = 0.028; candle.inputs["Depth"].default_value = 0.22
            t_can = ng.nodes.new("GeometryNodeTransform"); t_can.inputs["Translation"].default_value = (0.0, 0.0, 0.14)
            inst_can = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_can = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_can = ng.nodes.new("GeometryNodeSetMaterial"); sm_can.inputs["Material"].default_value = m_wax
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(ring_p.outputs["Curve"], ring_m.inputs["Curve"]); ng.links.new(ring_prof.outputs["Curve"], ring_m.inputs["Profile Curve"])
            ng.links.new(ring_m.outputs["Mesh"], t_ring.inputs["Geometry"]); ng.links.new(t_ring.outputs["Geometry"], sm_ring.inputs["Geometry"])
            ng.links.new(arm_bz.outputs["Curve"], arm_m.inputs["Curve"]); ng.links.new(arm_prof.outputs["Curve"], arm_m.inputs["Profile Curve"])
            ng.links.new(ring_p.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst_arm.inputs["Points"]); ng.links.new(arm_m.outputs["Mesh"], inst_arm.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst_arm.inputs["Rotation"])
            ng.links.new(inst_arm.outputs["Instances"], real_arm.inputs["Geometry"]); ng.links.new(real_arm.outputs["Geometry"], sm_arm.inputs["Geometry"])
            ng.links.new(cup.outputs["Mesh"], t_cup.inputs["Geometry"])
            ng.links.new(c2p.outputs["Points"], inst_cup.inputs["Points"]); ng.links.new(t_cup.outputs["Geometry"], inst_cup.inputs["Instance"])
            ng.links.new(inst_cup.outputs["Instances"], real_cup.inputs["Geometry"]); ng.links.new(real_cup.outputs["Geometry"], sm_cup.inputs["Geometry"])
            ng.links.new(candle.outputs["Mesh"], t_can.inputs["Geometry"])
            ng.links.new(c2p.outputs["Points"], inst_can.inputs["Points"]); ng.links.new(t_can.outputs["Geometry"], inst_can.inputs["Instance"])
            ng.links.new(inst_can.outputs["Instances"], real_can.inputs["Geometry"]); ng.links.new(real_can.outputs["Geometry"], sm_can.inputs["Geometry"])
            for s in (sm_ring, sm_arm, sm_cup, sm_can):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_02_wagon_wheel",
            "WagonWheel",
            "Build an oak and forged-iron spoked carriage wheel with a swept wooden rim, forged outer iron tire band, central hub, and 12 radial spokes.",
            """\
            m_oak = make_mat("AgedOak", (0.42, 0.26, 0.12), rough=0.62)
            m_iron = make_mat("ForgedIron", (0.16, 0.16, 0.18), rough=0.38, metallic=0.90)
            ng = bpy.data.node_groups.new("WagonWheel", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            rim_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_p.inputs["Resolution"].default_value = 48; rim_p.inputs["Radius"].default_value = 0.85
            rim_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_prof.inputs["Resolution"].default_value = 12; rim_prof.inputs["Radius"].default_value = 0.055
            rim_m = ng.nodes.new("GeometryNodeCurveToMesh"); rim_m.inputs["Fill Caps"].default_value = True
            sm_rim = ng.nodes.new("GeometryNodeSetMaterial"); sm_rim.inputs["Material"].default_value = m_oak
            tire_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); tire_p.inputs["Resolution"].default_value = 48; tire_p.inputs["Radius"].default_value = 0.90
            tire_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); tire_prof.inputs["Resolution"].default_value = 8; tire_prof.inputs["Radius"].default_value = 0.018
            tire_m = ng.nodes.new("GeometryNodeCurveToMesh"); tire_m.inputs["Fill Caps"].default_value = True
            sm_tire = ng.nodes.new("GeometryNodeSetMaterial"); sm_tire.inputs["Material"].default_value = m_iron
            hub = ng.nodes.new("GeometryNodeMeshCylinder"); hub.inputs["Vertices"].default_value = 24; hub.inputs["Radius"].default_value = 0.16; hub.inputs["Depth"].default_value = 0.24
            sm_hub = ng.nodes.new("GeometryNodeSetMaterial"); sm_hub.inputs["Material"].default_value = m_iron
            spk_line = ng.nodes.new("GeometryNodeCurvePrimitiveLine"); spk_line.inputs["Start"].default_value = (0.0, 0.0, 0.0); spk_line.inputs["End"].default_value = (0.0, 0.72, 0.0)
            spk_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); spk_prof.inputs["Resolution"].default_value = 8; spk_prof.inputs["Radius"].default_value = 0.022
            spk_m = ng.nodes.new("GeometryNodeCurveToMesh"); spk_m.inputs["Fill Caps"].default_value = True
            ax_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ax_p.inputs["Resolution"].default_value = 24; ax_p.inputs["Radius"].default_value = 0.10
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 12
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_spk = ng.nodes.new("GeometryNodeSetMaterial"); sm_spk.inputs["Material"].default_value = m_oak
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            tr_all = ng.nodes.new("GeometryNodeTransform"); tr_all.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); tr_all.inputs["Translation"].default_value = (0.0, 0.0, 0.92)
            ng.links.new(rim_p.outputs["Curve"], rim_m.inputs["Curve"]); ng.links.new(rim_prof.outputs["Curve"], rim_m.inputs["Profile Curve"])
            ng.links.new(rim_m.outputs["Mesh"], sm_rim.inputs["Geometry"])
            ng.links.new(tire_p.outputs["Curve"], tire_m.inputs["Curve"]); ng.links.new(tire_prof.outputs["Curve"], tire_m.inputs["Profile Curve"])
            ng.links.new(tire_m.outputs["Mesh"], sm_tire.inputs["Geometry"])
            ng.links.new(hub.outputs["Mesh"], sm_hub.inputs["Geometry"])
            ng.links.new(spk_line.outputs["Curve"], spk_m.inputs["Curve"]); ng.links.new(spk_prof.outputs["Curve"], spk_m.inputs["Profile Curve"])
            ng.links.new(ax_p.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(spk_m.outputs["Mesh"], inst.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm_spk.inputs["Geometry"])
            for s in (sm_rim, sm_tire, sm_hub, sm_spk):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], tr_all.inputs["Geometry"]); ng.links.new(tr_all.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_03_sunburst_mirror",
            "SunburstMirror",
            "Build a gilt brass sunburst wall medallion with a polished glass center disk, swept outer frame ring, and 24 radial tapered cone rays.",
            """\
            m_gold = make_mat("GiltBrass", (0.88, 0.68, 0.22), rough=0.18, metallic=0.96)
            m_glass = make_mat("SilverMirror", (0.82, 0.86, 0.90), rough=0.05, metallic=1.0)
            ng = bpy.data.node_groups.new("SunburstMirror", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            disk = ng.nodes.new("GeometryNodeMeshCylinder"); disk.inputs["Vertices"].default_value = 48; disk.inputs["Radius"].default_value = 0.36; disk.inputs["Depth"].default_value = 0.04
            sm_disk = ng.nodes.new("GeometryNodeSetMaterial"); sm_disk.inputs["Material"].default_value = m_glass
            rim_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_p.inputs["Resolution"].default_value = 48; rim_p.inputs["Radius"].default_value = 0.38
            rim_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_prof.inputs["Resolution"].default_value = 12; rim_prof.inputs["Radius"].default_value = 0.032
            rim_m = ng.nodes.new("GeometryNodeCurveToMesh"); rim_m.inputs["Fill Caps"].default_value = True
            sm_rim = ng.nodes.new("GeometryNodeSetMaterial"); sm_rim.inputs["Material"].default_value = m_gold
            ray = ng.nodes.new("GeometryNodeMeshCone"); ray.inputs["Vertices"].default_value = 6; ray.inputs["Radius Bottom"].default_value = 0.025; ray.inputs["Radius Top"].default_value = 0.002; ray.inputs["Depth"].default_value = 0.55
            t_ray = ng.nodes.new("GeometryNodeTransform"); t_ray.inputs["Rotation"].default_value = (-1.5708, 0.0, 0.0); t_ray.inputs["Translation"].default_value = (0.0, 0.27, 0.0)
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 24
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_ray = ng.nodes.new("GeometryNodeSetMaterial"); sm_ray.inputs["Material"].default_value = m_gold
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            tr_all = ng.nodes.new("GeometryNodeTransform"); tr_all.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); tr_all.inputs["Translation"].default_value = (0.0, 0.0, 0.95)
            ng.links.new(disk.outputs["Mesh"], sm_disk.inputs["Geometry"])
            ng.links.new(rim_p.outputs["Curve"], rim_m.inputs["Curve"]); ng.links.new(rim_prof.outputs["Curve"], rim_m.inputs["Profile Curve"])
            ng.links.new(rim_m.outputs["Mesh"], sm_rim.inputs["Geometry"])
            ng.links.new(ray.outputs["Mesh"], t_ray.inputs["Geometry"]); ng.links.new(rim_p.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(t_ray.outputs["Geometry"], inst.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm_ray.inputs["Geometry"])
            for s in (sm_disk, sm_rim, sm_ray):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], tr_all.inputs["Geometry"]); ng.links.new(tr_all.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_04_clockwork_gear",
            "ClockworkGear",
            "Build a machined brass escapement cogwheel gear with a swept inner hub collar, dark steel web spokes, swept outer rim, and 20 radial teeth.",
            """\
            m_brass = make_mat("MachinedBrass", (0.86, 0.66, 0.24), rough=0.22, metallic=0.95)
            m_steel = make_mat("BluedSteel", (0.16, 0.20, 0.28), rough=0.28, metallic=0.92)
            ng = bpy.data.node_groups.new("ClockworkGear", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            rim_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_p.inputs["Resolution"].default_value = 60; rim_p.inputs["Radius"].default_value = 0.72
            rim_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_prof.inputs["Resolution"].default_value = 8; rim_prof.inputs["Radius"].default_value = 0.055
            rim_m = ng.nodes.new("GeometryNodeCurveToMesh"); rim_m.inputs["Fill Caps"].default_value = True
            sm_rim = ng.nodes.new("GeometryNodeSetMaterial"); sm_rim.inputs["Material"].default_value = m_brass
            hub = ng.nodes.new("GeometryNodeMeshCylinder"); hub.inputs["Vertices"].default_value = 24; hub.inputs["Radius"].default_value = 0.20; hub.inputs["Depth"].default_value = 0.16
            sm_hub = ng.nodes.new("GeometryNodeSetMaterial"); sm_hub.inputs["Material"].default_value = m_steel
            tooth = ng.nodes.new("GeometryNodeMeshCube"); tooth.inputs["Size"].default_value = (0.075, 0.11, 0.10)
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 20
            inst_t = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_t = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_t = ng.nodes.new("GeometryNodeSetMaterial"); sm_t.inputs["Material"].default_value = m_brass
            spoke = ng.nodes.new("GeometryNodeMeshCube"); spoke.inputs["Size"].default_value = (0.06, 0.56, 0.05)
            t_sp = ng.nodes.new("GeometryNodeTransform"); t_sp.inputs["Translation"].default_value = (0.0, 0.36, 0.0)
            sp_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); sp_c.inputs["Resolution"].default_value = 12; sp_c.inputs["Radius"].default_value = 0.08
            c2p_s = ng.nodes.new("GeometryNodeCurveToPoints"); c2p_s.inputs["Count"].default_value = 6
            inst_s = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_s = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_s = ng.nodes.new("GeometryNodeSetMaterial"); sm_s.inputs["Material"].default_value = m_steel
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(rim_p.outputs["Curve"], rim_m.inputs["Curve"]); ng.links.new(rim_prof.outputs["Curve"], rim_m.inputs["Profile Curve"])
            ng.links.new(rim_m.outputs["Mesh"], sm_rim.inputs["Geometry"]); ng.links.new(hub.outputs["Mesh"], sm_hub.inputs["Geometry"])
            ng.links.new(rim_p.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst_t.inputs["Points"]); ng.links.new(tooth.outputs["Mesh"], inst_t.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst_t.inputs["Rotation"])
            ng.links.new(inst_t.outputs["Instances"], real_t.inputs["Geometry"]); ng.links.new(real_t.outputs["Geometry"], sm_t.inputs["Geometry"])
            ng.links.new(spoke.outputs["Mesh"], t_sp.inputs["Geometry"]); ng.links.new(sp_c.outputs["Curve"], c2p_s.inputs["Curve"])
            ng.links.new(c2p_s.outputs["Points"], inst_s.inputs["Points"]); ng.links.new(t_sp.outputs["Geometry"], inst_s.inputs["Instance"]); ng.links.new(c2p_s.outputs["Rotation"], inst_s.inputs["Rotation"])
            ng.links.new(inst_s.outputs["Instances"], real_s.inputs["Geometry"]); ng.links.new(real_s.outputs["Geometry"], sm_s.inputs["Geometry"])
            for s in (sm_rim, sm_hub, sm_t, sm_s):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_05_ship_helm",
            "ShipHelm",
            "Build a varnished teak nautical steering helm with a brass central hub collar, swept mahogany rim ring, and 8 turned outer hand grips.",
            """\
            m_teak = make_mat("VarnishedTeak", (0.48, 0.22, 0.08), rough=0.28)
            m_brass = make_mat("PolishedBrass", (0.86, 0.67, 0.22), rough=0.18, metallic=0.96)
            ng = bpy.data.node_groups.new("ShipHelm", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            rim_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_p.inputs["Resolution"].default_value = 48; rim_p.inputs["Radius"].default_value = 0.64
            rim_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_prof.inputs["Resolution"].default_value = 12; rim_prof.inputs["Radius"].default_value = 0.045
            rim_m = ng.nodes.new("GeometryNodeCurveToMesh"); rim_m.inputs["Fill Caps"].default_value = True
            sm_rim = ng.nodes.new("GeometryNodeSetMaterial"); sm_rim.inputs["Material"].default_value = m_teak
            hub = ng.nodes.new("GeometryNodeMeshCylinder"); hub.inputs["Vertices"].default_value = 24; hub.inputs["Radius"].default_value = 0.14; hub.inputs["Depth"].default_value = 0.18
            sm_hub = ng.nodes.new("GeometryNodeSetMaterial"); sm_hub.inputs["Material"].default_value = m_brass
            grip = ng.nodes.new("GeometryNodeMeshCylinder"); grip.inputs["Vertices"].default_value = 12; grip.inputs["Radius"].default_value = 0.024; grip.inputs["Depth"].default_value = 0.86
            t_grip = ng.nodes.new("GeometryNodeTransform"); t_grip.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); t_grip.inputs["Translation"].default_value = (0.0, 0.43, 0.0)
            ax_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ax_c.inputs["Resolution"].default_value = 16; ax_c.inputs["Radius"].default_value = 0.06
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 8
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_grip = ng.nodes.new("GeometryNodeSetMaterial"); sm_grip.inputs["Material"].default_value = m_teak
            knob = ng.nodes.new("GeometryNodeMeshUVSphere"); knob.inputs["Segments"].default_value = 12; knob.inputs["Rings"].default_value = 8; knob.inputs["Radius"].default_value = 0.036
            kn_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); kn_c.inputs["Resolution"].default_value = 16; kn_c.inputs["Radius"].default_value = 0.92
            c2p_k = ng.nodes.new("GeometryNodeCurveToPoints"); c2p_k.inputs["Count"].default_value = 8
            inst_k = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_k = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_k = ng.nodes.new("GeometryNodeSetMaterial"); sm_k.inputs["Material"].default_value = m_brass
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            tr_all = ng.nodes.new("GeometryNodeTransform"); tr_all.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); tr_all.inputs["Translation"].default_value = (0.0, 0.0, 0.98)
            ng.links.new(rim_p.outputs["Curve"], rim_m.inputs["Curve"]); ng.links.new(rim_prof.outputs["Curve"], rim_m.inputs["Profile Curve"])
            ng.links.new(rim_m.outputs["Mesh"], sm_rim.inputs["Geometry"]); ng.links.new(hub.outputs["Mesh"], sm_hub.inputs["Geometry"])
            ng.links.new(grip.outputs["Mesh"], t_grip.inputs["Geometry"]); ng.links.new(ax_c.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(t_grip.outputs["Geometry"], inst.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm_grip.inputs["Geometry"])
            ng.links.new(kn_c.outputs["Curve"], c2p_k.inputs["Curve"])
            ng.links.new(c2p_k.outputs["Points"], inst_k.inputs["Points"]); ng.links.new(knob.outputs["Mesh"], inst_k.inputs["Instance"])
            ng.links.new(inst_k.outputs["Instances"], real_k.inputs["Geometry"]); ng.links.new(real_k.outputs["Geometry"], sm_k.inputs["Geometry"])
            for s in (sm_rim, sm_hub, sm_grip, sm_k):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], tr_all.inputs["Geometry"]); ng.links.new(tr_all.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_06_carousel_canopy",
            "CarouselCanopy",
            "Build a circular carnival carousel tent canopy with a central brass pole, pitched cupola cone, gold upper cornice ring, and 10 brass drop rods.",
            """\
            m_red = make_mat("CarnivalRed", (0.76, 0.14, 0.14), rough=0.52)
            m_brass = make_mat("PolishedBrass", (0.88, 0.68, 0.22), rough=0.20, metallic=0.95)
            m_ivory = make_mat("PaintedIvory", (0.94, 0.90, 0.80), rough=0.38)
            ng = bpy.data.node_groups.new("CarouselCanopy", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            pole = ng.nodes.new("GeometryNodeMeshCylinder"); pole.inputs["Vertices"].default_value = 20; pole.inputs["Radius"].default_value = 0.11; pole.inputs["Depth"].default_value = 1.65
            t_pole = ng.nodes.new("GeometryNodeTransform"); t_pole.inputs["Translation"].default_value = (0.0, 0.0, 0.82)
            sm_pole = ng.nodes.new("GeometryNodeSetMaterial"); sm_pole.inputs["Material"].default_value = m_brass
            roof = ng.nodes.new("GeometryNodeMeshCone"); roof.inputs["Vertices"].default_value = 20; roof.inputs["Radius Bottom"].default_value = 1.15; roof.inputs["Radius Top"].default_value = 0.04; roof.inputs["Depth"].default_value = 0.62
            t_roof = ng.nodes.new("GeometryNodeTransform"); t_roof.inputs["Translation"].default_value = (0.0, 0.0, 1.56)
            sm_roof = ng.nodes.new("GeometryNodeSetMaterial"); sm_roof.inputs["Material"].default_value = m_red
            rim_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_p.inputs["Resolution"].default_value = 40; rim_p.inputs["Radius"].default_value = 1.08
            rim_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_prof.inputs["Resolution"].default_value = 10; rim_prof.inputs["Radius"].default_value = 0.045
            rim_m = ng.nodes.new("GeometryNodeCurveToMesh"); rim_m.inputs["Fill Caps"].default_value = True
            t_rim = ng.nodes.new("GeometryNodeTransform"); t_rim.inputs["Translation"].default_value = (0.0, 0.0, 1.24)
            sm_rim = ng.nodes.new("GeometryNodeSetMaterial"); sm_rim.inputs["Material"].default_value = m_ivory
            rod = ng.nodes.new("GeometryNodeMeshCylinder"); rod.inputs["Vertices"].default_value = 10; rod.inputs["Radius"].default_value = 0.018; rod.inputs["Depth"].default_value = 1.24
            t_rod = ng.nodes.new("GeometryNodeTransform"); t_rod.inputs["Translation"].default_value = (0.0, 0.0, 0.62)
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 10
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_rod = ng.nodes.new("GeometryNodeSetMaterial"); sm_rod.inputs["Material"].default_value = m_brass
            base = ng.nodes.new("GeometryNodeMeshCylinder"); base.inputs["Vertices"].default_value = 40; base.inputs["Radius"].default_value = 1.22; base.inputs["Depth"].default_value = 0.12
            t_base = ng.nodes.new("GeometryNodeTransform"); t_base.inputs["Translation"].default_value = (0.0, 0.0, 0.06)
            sm_base = ng.nodes.new("GeometryNodeSetMaterial"); sm_base.inputs["Material"].default_value = m_ivory
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(pole.outputs["Mesh"], t_pole.inputs["Geometry"]); ng.links.new(t_pole.outputs["Geometry"], sm_pole.inputs["Geometry"])
            ng.links.new(roof.outputs["Mesh"], t_roof.inputs["Geometry"]); ng.links.new(t_roof.outputs["Geometry"], sm_roof.inputs["Geometry"])
            ng.links.new(rim_p.outputs["Curve"], rim_m.inputs["Curve"]); ng.links.new(rim_prof.outputs["Curve"], rim_m.inputs["Profile Curve"])
            ng.links.new(rim_m.outputs["Mesh"], t_rim.inputs["Geometry"]); ng.links.new(t_rim.outputs["Geometry"], sm_rim.inputs["Geometry"])
            ng.links.new(rod.outputs["Mesh"], t_rod.inputs["Geometry"]); ng.links.new(rim_p.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(t_rod.outputs["Geometry"], inst.inputs["Instance"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm_rod.inputs["Geometry"])
            ng.links.new(base.outputs["Mesh"], t_base.inputs["Geometry"]); ng.links.new(t_base.outputs["Geometry"], sm_base.inputs["Geometry"])
            for s in (sm_pole, sm_roof, sm_rim, sm_rod, sm_base):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_07_rotunda_colonnade",
            "RotundaColonnade",
            "Build a neoclassical white marble circular portico rotunda colonnade with a dark granite stepped plinth, 12 perimeter columns, cornice ring, and bronze dome.",
            """\
            m_marble = make_mat("CarraraMarble", (0.92, 0.90, 0.87), rough=0.35)
            m_granite = make_mat("DarkGranite", (0.24, 0.25, 0.27), rough=0.60)
            m_bronze = make_mat("PatinaDome", (0.25, 0.42, 0.36), rough=0.36, metallic=0.82)
            ng = bpy.data.node_groups.new("RotundaColonnade", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            plinth = ng.nodes.new("GeometryNodeMeshCylinder"); plinth.inputs["Vertices"].default_value = 48; plinth.inputs["Radius"].default_value = 1.35; plinth.inputs["Depth"].default_value = 0.16
            t_pl = ng.nodes.new("GeometryNodeTransform"); t_pl.inputs["Translation"].default_value = (0.0, 0.0, 0.08)
            sm_pl = ng.nodes.new("GeometryNodeSetMaterial"); sm_pl.inputs["Material"].default_value = m_granite
            col_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); col_c.inputs["Resolution"].default_value = 36; col_c.inputs["Radius"].default_value = 1.05
            col = ng.nodes.new("GeometryNodeMeshCylinder"); col.inputs["Vertices"].default_value = 18; col.inputs["Radius"].default_value = 0.075; col.inputs["Depth"].default_value = 1.25
            t_col = ng.nodes.new("GeometryNodeTransform"); t_col.inputs["Translation"].default_value = (0.0, 0.0, 0.78)
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 12
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_col = ng.nodes.new("GeometryNodeSetMaterial"); sm_col.inputs["Material"].default_value = m_marble
            ent_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ent_prof.inputs["Resolution"].default_value = 8; ent_prof.inputs["Radius"].default_value = 0.095
            ent_m = ng.nodes.new("GeometryNodeCurveToMesh"); ent_m.inputs["Fill Caps"].default_value = True
            t_ent = ng.nodes.new("GeometryNodeTransform"); t_ent.inputs["Translation"].default_value = (0.0, 0.0, 1.44)
            sm_ent = ng.nodes.new("GeometryNodeSetMaterial"); sm_ent.inputs["Material"].default_value = m_marble
            dome = ng.nodes.new("GeometryNodeMeshUVSphere"); dome.inputs["Segments"].default_value = 32; dome.inputs["Rings"].default_value = 16; dome.inputs["Radius"].default_value = 1.04
            t_dom = ng.nodes.new("GeometryNodeTransform"); t_dom.inputs["Translation"].default_value = (0.0, 0.0, 1.48); t_dom.inputs["Scale"].default_value = (1.0, 1.0, 0.55)
            sm_dom = ng.nodes.new("GeometryNodeSetMaterial"); sm_dom.inputs["Material"].default_value = m_bronze
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(plinth.outputs["Mesh"], t_pl.inputs["Geometry"]); ng.links.new(t_pl.outputs["Geometry"], sm_pl.inputs["Geometry"])
            ng.links.new(col.outputs["Mesh"], t_col.inputs["Geometry"]); ng.links.new(col_c.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(t_col.outputs["Geometry"], inst.inputs["Instance"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm_col.inputs["Geometry"])
            ng.links.new(col_c.outputs["Curve"], ent_m.inputs["Curve"]); ng.links.new(ent_prof.outputs["Curve"], ent_m.inputs["Profile Curve"])
            ng.links.new(ent_m.outputs["Mesh"], t_ent.inputs["Geometry"]); ng.links.new(t_ent.outputs["Geometry"], sm_ent.inputs["Geometry"])
            ng.links.new(dome.outputs["Mesh"], t_dom.inputs["Geometry"]); ng.links.new(t_dom.outputs["Geometry"], sm_dom.inputs["Geometry"])
            for s in (sm_pl, sm_col, sm_ent, sm_dom):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_08_compass_rose",
            "CompassRose",
            "Build an engraved brass marine compass rose dial with a brushed bronze outer bezel ring, dark slate dial plate, cardinal star needles, and 16 radial degree ticks.",
            """\
            m_brass = make_mat("AntiqueBrass", (0.82, 0.62, 0.22), rough=0.25, metallic=0.94)
            m_slate = make_mat("DialSlate", (0.10, 0.12, 0.15), rough=0.45)
            m_red = make_mat("NorthCrimson", (0.78, 0.12, 0.12), rough=0.28)
            ng = bpy.data.node_groups.new("CompassRose", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            dial = ng.nodes.new("GeometryNodeMeshCylinder"); dial.inputs["Vertices"].default_value = 48; dial.inputs["Radius"].default_value = 0.80; dial.inputs["Depth"].default_value = 0.04
            sm_dial = ng.nodes.new("GeometryNodeSetMaterial"); sm_dial.inputs["Material"].default_value = m_slate
            bezel_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); bezel_p.inputs["Resolution"].default_value = 48; bezel_p.inputs["Radius"].default_value = 0.84
            bezel_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); bezel_prof.inputs["Resolution"].default_value = 12; bezel_prof.inputs["Radius"].default_value = 0.055
            bezel_m = ng.nodes.new("GeometryNodeCurveToMesh"); bezel_m.inputs["Fill Caps"].default_value = True
            sm_bez = ng.nodes.new("GeometryNodeSetMaterial"); sm_bez.inputs["Material"].default_value = m_brass
            star = ng.nodes.new("GeometryNodeCurveStar"); star.inputs["Points"].default_value = 8; star.inputs["Inner Radius"].default_value = 0.16; star.inputs["Outer Radius"].default_value = 0.66
            star_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); star_prof.inputs["Resolution"].default_value = 8; star_prof.inputs["Radius"].default_value = 0.018
            star_m = ng.nodes.new("GeometryNodeCurveToMesh"); star_m.inputs["Fill Caps"].default_value = True
            t_star = ng.nodes.new("GeometryNodeTransform"); t_star.inputs["Translation"].default_value = (0.0, 0.0, 0.04)
            sm_star = ng.nodes.new("GeometryNodeSetMaterial"); sm_star.inputs["Material"].default_value = m_red
            tick = ng.nodes.new("GeometryNodeMeshCube"); tick.inputs["Size"].default_value = (0.018, 0.075, 0.03)
            tk_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); tk_c.inputs["Resolution"].default_value = 32; tk_c.inputs["Radius"].default_value = 0.72
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 16
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            t_tk = ng.nodes.new("GeometryNodeTransform"); t_tk.inputs["Translation"].default_value = (0.0, 0.0, 0.03)
            sm_tk = ng.nodes.new("GeometryNodeSetMaterial"); sm_tk.inputs["Material"].default_value = m_brass
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(dial.outputs["Mesh"], sm_dial.inputs["Geometry"])
            ng.links.new(bezel_p.outputs["Curve"], bezel_m.inputs["Curve"]); ng.links.new(bezel_prof.outputs["Curve"], bezel_m.inputs["Profile Curve"])
            ng.links.new(bezel_m.outputs["Mesh"], sm_bez.inputs["Geometry"])
            ng.links.new(star.outputs["Curve"], star_m.inputs["Curve"]); ng.links.new(star_prof.outputs["Curve"], star_m.inputs["Profile Curve"])
            ng.links.new(star_m.outputs["Mesh"], t_star.inputs["Geometry"]); ng.links.new(t_star.outputs["Geometry"], sm_star.inputs["Geometry"])
            ng.links.new(tk_c.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(tick.outputs["Mesh"], inst.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], t_tk.inputs["Geometry"]); ng.links.new(t_tk.outputs["Geometry"], sm_tk.inputs["Geometry"])
            for s in (sm_dial, sm_bez, sm_star, sm_tk):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_09_lotus_fountain",
            "LotusFountain",
            "Build a carved limestone garden lotus fountain basin with a swept coping rim ring, water pool disk, central pedestal column, and 12 radial curved bronze lotus petals.",
            """\
            m_stone = make_mat("Limestone", (0.76, 0.72, 0.65), rough=0.72)
            m_water = make_mat("PoolWater", (0.14, 0.42, 0.48), rough=0.06)
            m_bronze = make_mat("DarkBronze", (0.45, 0.34, 0.20), rough=0.32, metallic=0.88)
            ng = bpy.data.node_groups.new("LotusFountain", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            rim_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_p.inputs["Resolution"].default_value = 48; rim_p.inputs["Radius"].default_value = 1.15
            rim_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_prof.inputs["Resolution"].default_value = 12; rim_prof.inputs["Radius"].default_value = 0.09
            rim_m = ng.nodes.new("GeometryNodeCurveToMesh"); rim_m.inputs["Fill Caps"].default_value = True
            t_rim = ng.nodes.new("GeometryNodeTransform"); t_rim.inputs["Translation"].default_value = (0.0, 0.0, 0.22)
            sm_rim = ng.nodes.new("GeometryNodeSetMaterial"); sm_rim.inputs["Material"].default_value = m_stone
            pool = ng.nodes.new("GeometryNodeMeshCylinder"); pool.inputs["Vertices"].default_value = 48; pool.inputs["Radius"].default_value = 1.10; pool.inputs["Depth"].default_value = 0.16
            t_pool = ng.nodes.new("GeometryNodeTransform"); t_pool.inputs["Translation"].default_value = (0.0, 0.0, 0.09)
            sm_pool = ng.nodes.new("GeometryNodeSetMaterial"); sm_pool.inputs["Material"].default_value = m_water
            ped = ng.nodes.new("GeometryNodeMeshCylinder"); ped.inputs["Vertices"].default_value = 24; ped.inputs["Radius"].default_value = 0.16; ped.inputs["Depth"].default_value = 0.72
            t_ped = ng.nodes.new("GeometryNodeTransform"); t_ped.inputs["Translation"].default_value = (0.0, 0.0, 0.38)
            sm_ped = ng.nodes.new("GeometryNodeSetMaterial"); sm_ped.inputs["Material"].default_value = m_stone
            petal_bz = ng.nodes.new("GeometryNodeCurveQuadraticBezier"); petal_bz.inputs["Resolution"].default_value = 12
            petal_bz.inputs["Start"].default_value = (0.0, 0.0, 0.0); petal_bz.inputs["Middle"].default_value = (0.0, 0.22, 0.18); petal_bz.inputs["End"].default_value = (0.0, 0.36, 0.32)
            petal_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); petal_prof.inputs["Resolution"].default_value = 8; petal_prof.inputs["Radius"].default_value = 0.045
            petal_m = ng.nodes.new("GeometryNodeCurveToMesh"); petal_m.inputs["Fill Caps"].default_value = True
            t_pet = ng.nodes.new("GeometryNodeTransform"); t_pet.inputs["Translation"].default_value = (0.0, 0.0, 0.72); t_pet.inputs["Scale"].default_value = (1.6, 1.0, 0.35)
            pet_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); pet_c.inputs["Resolution"].default_value = 24; pet_c.inputs["Radius"].default_value = 0.12
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 12
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_pet = ng.nodes.new("GeometryNodeSetMaterial"); sm_pet.inputs["Material"].default_value = m_bronze
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(rim_p.outputs["Curve"], rim_m.inputs["Curve"]); ng.links.new(rim_prof.outputs["Curve"], rim_m.inputs["Profile Curve"])
            ng.links.new(rim_m.outputs["Mesh"], t_rim.inputs["Geometry"]); ng.links.new(t_rim.outputs["Geometry"], sm_rim.inputs["Geometry"])
            ng.links.new(pool.outputs["Mesh"], t_pool.inputs["Geometry"]); ng.links.new(t_pool.outputs["Geometry"], sm_pool.inputs["Geometry"])
            ng.links.new(ped.outputs["Mesh"], t_ped.inputs["Geometry"]); ng.links.new(t_ped.outputs["Geometry"], sm_ped.inputs["Geometry"])
            ng.links.new(petal_bz.outputs["Curve"], petal_m.inputs["Curve"]); ng.links.new(petal_prof.outputs["Curve"], petal_m.inputs["Profile Curve"])
            ng.links.new(petal_m.outputs["Mesh"], t_pet.inputs["Geometry"]); ng.links.new(pet_c.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(t_pet.outputs["Geometry"], inst.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm_pet.inputs["Geometry"])
            for s in (sm_rim, sm_pool, sm_ped, sm_pet):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_10_turbine_rotor",
            "TurbineRotor",
            "Build a brushed titanium axial jet turbofan assembly with an aerodynamic nose cone spinner, outer intake shroud ring, and 16 pitched radial fan blades.",
            """\
            m_ti = make_mat("BrushedTitanium", (0.64, 0.67, 0.72), rough=0.26, metallic=0.94)
            m_shroud = make_mat("DarkCowling", (0.14, 0.16, 0.20), rough=0.36, metallic=0.80)
            ng = bpy.data.node_groups.new("TurbineRotor", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            spinner = ng.nodes.new("GeometryNodeMeshCone"); spinner.inputs["Vertices"].default_value = 28; spinner.inputs["Radius Bottom"].default_value = 0.22; spinner.inputs["Radius Top"].default_value = 0.01; spinner.inputs["Depth"].default_value = 0.34
            t_sp = ng.nodes.new("GeometryNodeTransform"); t_sp.inputs["Translation"].default_value = (0.0, 0.0, 0.18)
            sm_sp = ng.nodes.new("GeometryNodeSetMaterial"); sm_sp.inputs["Material"].default_value = m_shroud
            shroud_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); shroud_p.inputs["Resolution"].default_value = 48; shroud_p.inputs["Radius"].default_value = 0.92
            shroud_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); shroud_prof.inputs["Resolution"].default_value = 12; shroud_prof.inputs["Radius"].default_value = 0.065
            shroud_m = ng.nodes.new("GeometryNodeCurveToMesh"); shroud_m.inputs["Fill Caps"].default_value = True
            sm_sh = ng.nodes.new("GeometryNodeSetMaterial"); sm_sh.inputs["Material"].default_value = m_shroud
            blade = ng.nodes.new("GeometryNodeMeshCube"); blade.inputs["Size"].default_value = (0.16, 0.66, 0.024)
            t_bl = ng.nodes.new("GeometryNodeTransform"); t_bl.inputs["Rotation"].default_value = (0.0, 0.42, 0.0); t_bl.inputs["Translation"].default_value = (0.0, 0.36, 0.0)
            hub_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); hub_c.inputs["Resolution"].default_value = 24; hub_c.inputs["Radius"].default_value = 0.18
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 16
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_bl = ng.nodes.new("GeometryNodeSetMaterial"); sm_bl.inputs["Material"].default_value = m_ti
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            tr_all = ng.nodes.new("GeometryNodeTransform"); tr_all.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); tr_all.inputs["Translation"].default_value = (0.0, 0.0, 1.05)
            ng.links.new(spinner.outputs["Mesh"], t_sp.inputs["Geometry"]); ng.links.new(t_sp.outputs["Geometry"], sm_sp.inputs["Geometry"])
            ng.links.new(shroud_p.outputs["Curve"], shroud_m.inputs["Curve"]); ng.links.new(shroud_prof.outputs["Curve"], shroud_m.inputs["Profile Curve"])
            ng.links.new(shroud_m.outputs["Mesh"], sm_sh.inputs["Geometry"])
            ng.links.new(blade.outputs["Mesh"], t_bl.inputs["Geometry"]); ng.links.new(hub_c.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(t_bl.outputs["Geometry"], inst.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm_bl.inputs["Geometry"])
            for s in (sm_sp, sm_sh, sm_bl):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], tr_all.inputs["Geometry"]); ng.links.new(tr_all.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_11_bicycle_spokes",
            "BicycleWheel",
            "Build an anodized dark alloy bicycle wheel assembly with a swept rubber tire torus, alloy rim ring, steel center hub, and 18 thin stainless radial spokes.",
            """\
            m_rubber = make_mat("TreadRubber", (0.08, 0.08, 0.09), rough=0.78)
            m_alloy = make_mat("DarkAlloy", (0.25, 0.27, 0.30), rough=0.30, metallic=0.88)
            m_steel = make_mat("StainlessSpoke", (0.82, 0.84, 0.86), rough=0.16, metallic=0.98)
            ng = bpy.data.node_groups.new("BicycleWheel", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            tire_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); tire_p.inputs["Resolution"].default_value = 56; tire_p.inputs["Radius"].default_value = 0.82
            tire_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); tire_prof.inputs["Resolution"].default_value = 12; tire_prof.inputs["Radius"].default_value = 0.048
            tire_m = ng.nodes.new("GeometryNodeCurveToMesh"); tire_m.inputs["Fill Caps"].default_value = True
            sm_tire = ng.nodes.new("GeometryNodeSetMaterial"); sm_tire.inputs["Material"].default_value = m_rubber
            rim_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_p.inputs["Resolution"].default_value = 56; rim_p.inputs["Radius"].default_value = 0.76
            rim_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_prof.inputs["Resolution"].default_value = 10; rim_prof.inputs["Radius"].default_value = 0.025
            rim_m = ng.nodes.new("GeometryNodeCurveToMesh"); rim_m.inputs["Fill Caps"].default_value = True
            sm_rim = ng.nodes.new("GeometryNodeSetMaterial"); sm_rim.inputs["Material"].default_value = m_alloy
            hub = ng.nodes.new("GeometryNodeMeshCylinder"); hub.inputs["Vertices"].default_value = 24; hub.inputs["Radius"].default_value = 0.09; hub.inputs["Depth"].default_value = 0.16
            sm_hub = ng.nodes.new("GeometryNodeSetMaterial"); sm_hub.inputs["Material"].default_value = m_steel
            spk_l = ng.nodes.new("GeometryNodeCurvePrimitiveLine"); spk_l.inputs["Start"].default_value = (0.0, 0.0, 0.0); spk_l.inputs["End"].default_value = (0.0, 0.69, 0.0)
            spk_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); spk_prof.inputs["Resolution"].default_value = 6; spk_prof.inputs["Radius"].default_value = 0.008
            spk_m = ng.nodes.new("GeometryNodeCurveToMesh"); spk_m.inputs["Fill Caps"].default_value = True
            ax_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ax_c.inputs["Resolution"].default_value = 24; ax_c.inputs["Radius"].default_value = 0.07
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 18
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_spk = ng.nodes.new("GeometryNodeSetMaterial"); sm_spk.inputs["Material"].default_value = m_steel
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            tr_all = ng.nodes.new("GeometryNodeTransform"); tr_all.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); tr_all.inputs["Translation"].default_value = (0.0, 0.0, 0.90)
            ng.links.new(tire_p.outputs["Curve"], tire_m.inputs["Curve"]); ng.links.new(tire_prof.outputs["Curve"], tire_m.inputs["Profile Curve"])
            ng.links.new(tire_m.outputs["Mesh"], sm_tire.inputs["Geometry"])
            ng.links.new(rim_p.outputs["Curve"], rim_m.inputs["Curve"]); ng.links.new(rim_prof.outputs["Curve"], rim_m.inputs["Profile Curve"])
            ng.links.new(rim_m.outputs["Mesh"], sm_rim.inputs["Geometry"]); ng.links.new(hub.outputs["Mesh"], sm_hub.inputs["Geometry"])
            ng.links.new(spk_l.outputs["Curve"], spk_m.inputs["Curve"]); ng.links.new(spk_prof.outputs["Curve"], spk_m.inputs["Profile Curve"])
            ng.links.new(ax_c.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(spk_m.outputs["Mesh"], inst.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm_spk.inputs["Geometry"])
            for s in (sm_tire, sm_rim, sm_hub, sm_spk):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], tr_all.inputs["Geometry"]); ng.links.new(tr_all.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_12_floral_wreath",
            "FloralWreath",
            "Build a decorative seasonal vine ring wreath with a braided woody vine core, 16 green leaf clusters, and 8 bright crimson berry spheres instanced around the ring.",
            """\
            m_vine = make_mat("WoodyVine", (0.34, 0.22, 0.12), rough=0.72)
            m_leaf = make_mat("CedarGreen", (0.12, 0.38, 0.14), rough=0.45)
            m_berry = make_mat("CrimsonBerry", (0.82, 0.08, 0.10), rough=0.22)
            ng = bpy.data.node_groups.new("FloralWreath", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ring_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ring_p.inputs["Resolution"].default_value = 48; ring_p.inputs["Radius"].default_value = 0.68
            ring_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rim_prof = ring_prof
            rim_prof.inputs["Resolution"].default_value = 10; rim_prof.inputs["Radius"].default_value = 0.065
            ring_m = ng.nodes.new("GeometryNodeCurveToMesh"); ring_m.inputs["Fill Caps"].default_value = True
            sm_ring = ng.nodes.new("GeometryNodeSetMaterial"); sm_ring.inputs["Material"].default_value = m_vine
            leaf = ng.nodes.new("GeometryNodeMeshIcoSphere"); leaf.inputs["Radius"].default_value = 0.11; leaf.inputs["Subdivisions"].default_value = 2
            t_lf = ng.nodes.new("GeometryNodeTransform"); t_lf.inputs["Scale"].default_value = (1.6, 0.75, 0.30); t_lf.inputs["Rotation"].default_value = (0.35, 0.25, 0.4)
            c2p_l = ng.nodes.new("GeometryNodeCurveToPoints"); c2p_l.inputs["Count"].default_value = 16
            inst_l = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_l = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_l = ng.nodes.new("GeometryNodeSetMaterial"); sm_l.inputs["Material"].default_value = m_leaf
            berry = ng.nodes.new("GeometryNodeMeshUVSphere"); berry.inputs["Segments"].default_value = 12; berry.inputs["Rings"].default_value = 8; berry.inputs["Radius"].default_value = 0.055
            c2p_b = ng.nodes.new("GeometryNodeCurveToPoints"); c2p_b.inputs["Count"].default_value = 8
            inst_b = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_b = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_b = ng.nodes.new("GeometryNodeSetMaterial"); sm_b.inputs["Material"].default_value = m_berry
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            tr_all = ng.nodes.new("GeometryNodeTransform"); tr_all.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); tr_all.inputs["Translation"].default_value = (0.0, 0.0, 0.90)
            ng.links.new(ring_p.outputs["Curve"], ring_m.inputs["Curve"]); ng.links.new(rim_prof.outputs["Curve"], ring_m.inputs["Profile Curve"])
            ng.links.new(ring_m.outputs["Mesh"], sm_ring.inputs["Geometry"])
            ng.links.new(leaf.outputs["Mesh"], t_lf.inputs["Geometry"]); ng.links.new(ring_p.outputs["Curve"], c2p_l.inputs["Curve"])
            ng.links.new(c2p_l.outputs["Points"], inst_l.inputs["Points"]); ng.links.new(t_lf.outputs["Geometry"], inst_l.inputs["Instance"]); ng.links.new(c2p_l.outputs["Rotation"], inst_l.inputs["Rotation"])
            ng.links.new(inst_l.outputs["Instances"], real_l.inputs["Geometry"]); ng.links.new(real_l.outputs["Geometry"], sm_l.inputs["Geometry"])
            ng.links.new(ring_p.outputs["Curve"], c2p_b.inputs["Curve"])
            ng.links.new(c2p_b.outputs["Points"], inst_b.inputs["Points"]); ng.links.new(berry.outputs["Mesh"], inst_b.inputs["Instance"])
            ng.links.new(inst_b.outputs["Instances"], real_b.inputs["Geometry"]); ng.links.new(real_b.outputs["Geometry"], sm_b.inputs["Geometry"])
            for s in (sm_ring, sm_l, sm_b):
                ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], tr_all.inputs["Geometry"]); ng.links.new(tr_all.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_13_armillary_sphere",
            "ArmillarySphere",
            "Build an antique astronomical brass armillary sphere with nested meridian and tilted ecliptic rings, a cobalt central globe, and a turned walnut pedestal stand.",
            """\
            m_brass = make_mat("EngravedBrass", (0.85, 0.66, 0.24), rough=0.22, metallic=0.95)
            m_globe = make_mat("CobaltGlobe", (0.12, 0.28, 0.56), rough=0.30)
            m_wood = make_mat("DarkWalnut", (0.26, 0.16, 0.08), rough=0.52)
            ng = bpy.data.node_groups.new("ArmillarySphere", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            r_outer = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); r_outer.inputs["Resolution"].default_value = 48; r_outer.inputs["Radius"].default_value = 0.68
            r_inner = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); r_inner.inputs["Resolution"].default_value = 48; r_inner.inputs["Radius"].default_value = 0.55
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); prof.inputs["Resolution"].default_value = 10; prof.inputs["Radius"].default_value = 0.025
            m_out = ng.nodes.new("GeometryNodeCurveToMesh"); m_out.inputs["Fill Caps"].default_value = True
            m_in = ng.nodes.new("GeometryNodeCurveToMesh"); m_in.inputs["Fill Caps"].default_value = True
            stud = ng.nodes.new("GeometryNodeMeshIcoSphere"); stud.inputs["Radius"].default_value = 0.032; stud.inputs["Subdivisions"].default_value = 1
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 12
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            t_r1 = ng.nodes.new("GeometryNodeTransform"); t_r1.inputs["Translation"].default_value = (0.0, 0.0, 1.02)
            t_r2 = ng.nodes.new("GeometryNodeTransform"); t_r2.inputs["Rotation"].default_value = (1.5708, 0.35, 0.0); t_r2.inputs["Translation"].default_value = (0.0, 0.0, 1.02)
            t_r3 = ng.nodes.new("GeometryNodeTransform"); t_r3.inputs["Rotation"].default_value = (0.58, 1.15, 0.3); t_r3.inputs["Translation"].default_value = (0.0, 0.0, 1.02)
            t_st_ring = ng.nodes.new("GeometryNodeTransform"); t_st_ring.inputs["Translation"].default_value = (0.0, 0.0, 1.02)
            jn_b = ng.nodes.new("GeometryNodeJoinGeometry")
            sm_b = ng.nodes.new("GeometryNodeSetMaterial"); sm_b.inputs["Material"].default_value = m_brass
            globe = ng.nodes.new("GeometryNodeMeshUVSphere"); globe.inputs["Segments"].default_value = 28; globe.inputs["Rings"].default_value = 16; globe.inputs["Radius"].default_value = 0.20
            t_gl = ng.nodes.new("GeometryNodeTransform"); t_gl.inputs["Translation"].default_value = (0.0, 0.0, 1.02)
            sm_gl = ng.nodes.new("GeometryNodeSetMaterial"); sm_gl.inputs["Material"].default_value = m_globe
            stand = ng.nodes.new("GeometryNodeMeshCylinder"); stand.inputs["Vertices"].default_value = 28; stand.inputs["Radius"].default_value = 0.32; stand.inputs["Depth"].default_value = 0.14
            t_st = ng.nodes.new("GeometryNodeTransform"); t_st.inputs["Translation"].default_value = (0.0, 0.0, 0.07)
            col = ng.nodes.new("GeometryNodeMeshCylinder"); col.inputs["Vertices"].default_value = 16; col.inputs["Radius"].default_value = 0.045; col.inputs["Depth"].default_value = 0.42
            t_cl = ng.nodes.new("GeometryNodeTransform"); t_cl.inputs["Translation"].default_value = (0.0, 0.0, 0.28)
            jn_w = ng.nodes.new("GeometryNodeJoinGeometry")
            sm_w = ng.nodes.new("GeometryNodeSetMaterial"); sm_w.inputs["Material"].default_value = m_wood
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(r_outer.outputs["Curve"], m_out.inputs["Curve"]); ng.links.new(prof.outputs["Curve"], m_out.inputs["Profile Curve"])
            ng.links.new(r_inner.outputs["Curve"], m_in.inputs["Curve"]); ng.links.new(prof.outputs["Curve"], m_in.inputs["Profile Curve"])
            ng.links.new(r_outer.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(stud.outputs["Mesh"], inst.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], t_st_ring.inputs["Geometry"])
            ng.links.new(m_out.outputs["Mesh"], t_r1.inputs["Geometry"]); ng.links.new(m_out.outputs["Mesh"], t_r2.inputs["Geometry"]); ng.links.new(m_in.outputs["Mesh"], t_r3.inputs["Geometry"])
            for s in (t_r1, t_r2, t_r3, t_st_ring): ng.links.new(s.outputs["Geometry"], jn_b.inputs["Geometry"])
            ng.links.new(jn_b.outputs["Geometry"], sm_b.inputs["Geometry"])
            ng.links.new(globe.outputs["Mesh"], t_gl.inputs["Geometry"]); ng.links.new(t_gl.outputs["Geometry"], sm_gl.inputs["Geometry"])
            ng.links.new(stand.outputs["Mesh"], t_st.inputs["Geometry"]); ng.links.new(col.outputs["Mesh"], t_cl.inputs["Geometry"])
            ng.links.new(t_st.outputs["Geometry"], jn_w.inputs["Geometry"]); ng.links.new(t_cl.outputs["Geometry"], jn_w.inputs["Geometry"]); ng.links.new(jn_w.outputs["Geometry"], sm_w.inputs["Geometry"])
            for s in (sm_b, sm_gl, sm_w): ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_14_windmill_sails",
            "WindmillSails",
            "Build a four-blade timber and white sailcloth Dutch windmill rotor cross with a central oak axle hub, 4 radial spar poles, and angled canvas lattice blades.",
            """\
            m_oak = make_mat("TimberOak", (0.38, 0.24, 0.12), rough=0.64)
            m_canvas = make_mat("SailCanvas", (0.94, 0.92, 0.86), rough=0.72)
            ng = bpy.data.node_groups.new("WindmillSails", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            hub = ng.nodes.new("GeometryNodeMeshCylinder"); hub.inputs["Vertices"].default_value = 24; hub.inputs["Radius"].default_value = 0.16; hub.inputs["Depth"].default_value = 0.28
            sm_hub = ng.nodes.new("GeometryNodeSetMaterial"); sm_hub.inputs["Material"].default_value = m_oak
            spar_l = ng.nodes.new("GeometryNodeCurvePrimitiveLine"); spar_l.inputs["Start"].default_value = (0.0, 0.0, 0.0); spar_l.inputs["End"].default_value = (0.0, 1.35, 0.0)
            spar_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); spar_prof.inputs["Resolution"].default_value = 8; spar_prof.inputs["Radius"].default_value = 0.032
            spar_m = ng.nodes.new("GeometryNodeCurveToMesh"); spar_m.inputs["Fill Caps"].default_value = True
            ax_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ax_c.inputs["Resolution"].default_value = 16; ax_c.inputs["Radius"].default_value = 0.08
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 4
            inst_sp = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_sp = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_sp = ng.nodes.new("GeometryNodeSetMaterial"); sm_sp.inputs["Material"].default_value = m_oak
            sail = ng.nodes.new("GeometryNodeMeshCube"); sail.inputs["Size"].default_value = (0.28, 1.02, 0.018)
            t_sail = ng.nodes.new("GeometryNodeTransform"); t_sail.inputs["Translation"].default_value = (0.16, 0.82, 0.0); t_sail.inputs["Rotation"].default_value = (0.0, 0.26, 0.0)
            inst_sl = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real_sl = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_sl = ng.nodes.new("GeometryNodeSetMaterial"); sm_sl.inputs["Material"].default_value = m_canvas
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            tr_all = ng.nodes.new("GeometryNodeTransform"); tr_all.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); tr_all.inputs["Translation"].default_value = (0.0, 0.0, 1.45)
            ng.links.new(hub.outputs["Mesh"], sm_hub.inputs["Geometry"])
            ng.links.new(spar_l.outputs["Curve"], spar_m.inputs["Curve"]); ng.links.new(spar_prof.outputs["Curve"], spar_m.inputs["Profile Curve"])
            ng.links.new(ax_c.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst_sp.inputs["Points"]); ng.links.new(spar_m.outputs["Mesh"], inst_sp.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst_sp.inputs["Rotation"])
            ng.links.new(inst_sp.outputs["Instances"], real_sp.inputs["Geometry"]); ng.links.new(real_sp.outputs["Geometry"], sm_sp.inputs["Geometry"])
            ng.links.new(sail.outputs["Mesh"], t_sail.inputs["Geometry"])
            ng.links.new(c2p.outputs["Points"], inst_sl.inputs["Points"]); ng.links.new(t_sail.outputs["Geometry"], inst_sl.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst_sl.inputs["Rotation"])
            ng.links.new(inst_sl.outputs["Instances"], real_sl.inputs["Geometry"]); ng.links.new(real_sl.outputs["Geometry"], sm_sl.inputs["Geometry"])
            for s in (sm_hub, sm_sp, sm_sl): ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], tr_all.inputs["Geometry"]); ng.links.new(tr_all.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_15_drum_pendant",
            "DrumPendant",
            "Build a pleated cream linen circular hanging drum lamp shade with a ceiling brass drop rod, 4 radial support struts, top/bottom brass trim rings, and inner diffuser.",
            """\
            m_linen = make_mat("CreamLinen", (0.95, 0.91, 0.82), rough=0.72)
            m_brass = make_mat("BrushedBrass", (0.85, 0.66, 0.24), rough=0.22, metallic=0.95)
            m_bulb = make_mat("WarmDiffuser", (0.98, 0.88, 0.65), rough=0.18)
            ng = bpy.data.node_groups.new("DrumPendant", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            rod = ng.nodes.new("GeometryNodeMeshCylinder"); rod.inputs["Vertices"].default_value = 16; rod.inputs["Radius"].default_value = 0.025; rod.inputs["Depth"].default_value = 0.90
            t_rod = ng.nodes.new("GeometryNodeTransform"); t_rod.inputs["Translation"].default_value = (0.0, 0.0, 1.35)
            sm_rod = ng.nodes.new("GeometryNodeSetMaterial"); sm_rod.inputs["Material"].default_value = m_brass
            ring_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ring_p.inputs["Resolution"].default_value = 48; ring_p.inputs["Radius"].default_value = 0.72
            ring_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ring_prof.inputs["Resolution"].default_value = 10; ring_prof.inputs["Radius"].default_value = 0.022
            ring_m = ng.nodes.new("GeometryNodeCurveToMesh"); ring_m.inputs["Fill Caps"].default_value = True
            t_top = ng.nodes.new("GeometryNodeTransform"); t_top.inputs["Translation"].default_value = (0.0, 0.0, 1.12)
            t_bot = ng.nodes.new("GeometryNodeTransform"); t_bot.inputs["Translation"].default_value = (0.0, 0.0, 0.62)
            strut = ng.nodes.new("GeometryNodeCurvePrimitiveLine"); strut.inputs["Start"].default_value = (0.0, 0.0, 0.0); strut.inputs["End"].default_value = (0.0, 0.70, 0.0)
            st_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); st_prof.inputs["Resolution"].default_value = 6; st_prof.inputs["Radius"].default_value = 0.010
            st_m = ng.nodes.new("GeometryNodeCurveToMesh"); st_m.inputs["Fill Caps"].default_value = True
            t_st_m = ng.nodes.new("GeometryNodeTransform"); t_st_m.inputs["Translation"].default_value = (0.0, 0.0, 0.92)
            hub_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); hub_c.inputs["Resolution"].default_value = 12; hub_c.inputs["Radius"].default_value = 0.02
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 4
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            jn_b = ng.nodes.new("GeometryNodeJoinGeometry")
            sm_b = ng.nodes.new("GeometryNodeSetMaterial"); sm_b.inputs["Material"].default_value = m_brass
            shade = ng.nodes.new("GeometryNodeMeshCylinder"); shade.inputs["Vertices"].default_value = 48; shade.inputs["Radius"].default_value = 0.71; shade.inputs["Depth"].default_value = 0.50
            t_sh = ng.nodes.new("GeometryNodeTransform"); t_sh.inputs["Translation"].default_value = (0.0, 0.0, 0.87)
            sm_sh = ng.nodes.new("GeometryNodeSetMaterial"); sm_sh.inputs["Material"].default_value = m_linen
            diff = ng.nodes.new("GeometryNodeMeshUVSphere"); diff.inputs["Segments"].default_value = 24; diff.inputs["Rings"].default_value = 12; diff.inputs["Radius"].default_value = 0.22
            t_df = ng.nodes.new("GeometryNodeTransform"); t_df.inputs["Translation"].default_value = (0.0, 0.0, 0.84)
            sm_df = ng.nodes.new("GeometryNodeSetMaterial"); sm_df.inputs["Material"].default_value = m_bulb
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(rod.outputs["Mesh"], t_rod.inputs["Geometry"])
            ng.links.new(ring_p.outputs["Curve"], ring_m.inputs["Curve"]); ng.links.new(ring_prof.outputs["Curve"], ring_m.inputs["Profile Curve"])
            ng.links.new(ring_m.outputs["Mesh"], t_top.inputs["Geometry"]); ng.links.new(ring_m.outputs["Mesh"], t_bot.inputs["Geometry"])
            ng.links.new(strut.outputs["Curve"], st_m.inputs["Curve"]); ng.links.new(st_prof.outputs["Curve"], st_m.inputs["Profile Curve"])
            ng.links.new(st_m.outputs["Mesh"], t_st_m.inputs["Geometry"]); ng.links.new(hub_c.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(t_st_m.outputs["Geometry"], inst.inputs["Instance"]); ng.links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
            for s in (t_rod, t_top, t_bot, real): ng.links.new(s.outputs["Geometry"], jn_b.inputs["Geometry"])
            ng.links.new(jn_b.outputs["Geometry"], sm_b.inputs["Geometry"])
            ng.links.new(shade.outputs["Mesh"], t_sh.inputs["Geometry"]); ng.links.new(t_sh.outputs["Geometry"], sm_sh.inputs["Geometry"])
            ng.links.new(diff.outputs["Mesh"], t_df.inputs["Geometry"]); ng.links.new(t_df.outputs["Geometry"], sm_df.inputs["Geometry"])
            for s in (sm_b, sm_sh, sm_df): ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
        (
            "c7_rad_16_temple_bell_frame",
            "TempleBellFrame",
            "Build a ceremonial cast-bronze temple bell suspended under a swept granite ring lintel supported by 6 cedar timber frame pillars around a stone dais.",
            """\
            m_bronze = make_mat("CastBronze", (0.58, 0.44, 0.24), rough=0.34, metallic=0.88)
            m_cedar = make_mat("RedCedar", (0.42, 0.18, 0.10), rough=0.58)
            m_stone = make_mat("MossStone", (0.40, 0.42, 0.38), rough=0.78)
            ng = bpy.data.node_groups.new("TempleBellFrame", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            dais = ng.nodes.new("GeometryNodeMeshCylinder"); dais.inputs["Vertices"].default_value = 36; dais.inputs["Radius"].default_value = 1.15; dais.inputs["Depth"].default_value = 0.18
            t_d = ng.nodes.new("GeometryNodeTransform"); t_d.inputs["Translation"].default_value = (0.0, 0.0, 0.09)
            sm_d = ng.nodes.new("GeometryNodeSetMaterial"); sm_d.inputs["Material"].default_value = m_stone
            pil_c = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); pil_c.inputs["Resolution"].default_value = 24; pil_c.inputs["Radius"].default_value = 0.88
            pil = ng.nodes.new("GeometryNodeMeshCylinder"); pil.inputs["Vertices"].default_value = 12; pil.inputs["Radius"].default_value = 0.065; pil.inputs["Depth"].default_value = 1.45
            t_pil = ng.nodes.new("GeometryNodeTransform"); t_pil.inputs["Translation"].default_value = (0.0, 0.0, 0.86)
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 6
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm_pil = ng.nodes.new("GeometryNodeSetMaterial"); sm_pil.inputs["Material"].default_value = m_cedar
            lint_prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); lint_prof.inputs["Resolution"].default_value = 8; lint_prof.inputs["Radius"].default_value = 0.085
            lint_m = ng.nodes.new("GeometryNodeCurveToMesh"); lint_m.inputs["Fill Caps"].default_value = True
            t_lint = ng.nodes.new("GeometryNodeTransform"); t_lint.inputs["Translation"].default_value = (0.0, 0.0, 1.56)
            sm_lint = ng.nodes.new("GeometryNodeSetMaterial"); sm_lint.inputs["Material"].default_value = m_cedar
            bell = ng.nodes.new("GeometryNodeMeshCone"); bell.inputs["Vertices"].default_value = 28; bell.inputs["Radius Bottom"].default_value = 0.36; bell.inputs["Radius Top"].default_value = 0.18; bell.inputs["Depth"].default_value = 0.62
            t_bell = ng.nodes.new("GeometryNodeTransform"); t_bell.inputs["Translation"].default_value = (0.0, 0.0, 1.02)
            sm_bell = ng.nodes.new("GeometryNodeSetMaterial"); sm_bell.inputs["Material"].default_value = m_bronze
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(dais.outputs["Mesh"], t_d.inputs["Geometry"]); ng.links.new(t_d.outputs["Geometry"], sm_d.inputs["Geometry"])
            ng.links.new(pil.outputs["Mesh"], t_pil.inputs["Geometry"]); ng.links.new(pil_c.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(t_pil.outputs["Geometry"], inst.inputs["Instance"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm_pil.inputs["Geometry"])
            ng.links.new(pil_c.outputs["Curve"], lint_m.inputs["Curve"]); ng.links.new(lint_prof.outputs["Curve"], lint_m.inputs["Profile Curve"])
            ng.links.new(lint_m.outputs["Mesh"], t_lint.inputs["Geometry"]); ng.links.new(t_lint.outputs["Geometry"], sm_lint.inputs["Geometry"])
            ng.links.new(bell.outputs["Mesh"], t_bell.inputs["Geometry"]); ng.links.new(t_bell.outputs["Geometry"], sm_bell.inputs["Geometry"])
            for s in (sm_d, sm_pil, sm_lint, sm_bell): ng.links.new(s.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], go.inputs["Geometry"])
            """,
        ),
    ]
    out: list[dict[str, Any]] = []
    for rid, name, prompt, body in specs:
        code = _wrap_radial_script(body, name, f"{name}GN")
        out.append({
            "id": rid,
            "task_id": rid,
            "source": "c7_radial_assembly",
            "prompt": prompt + bgn._SUFFIX,
            "response": f"```python\n{code.strip()}\n```",
        })
    return out


def format_pin_flow_ir_c7(dag_ir: list[dict[str, Any]]) -> str:
    """Serialize live Blender 5.4 GeometryNodeTree DAG telemetry with full exact bl_idnames."""
    lines = ["[Pin-Flow IR (Blender 5.4)]"]
    for tree_info in dag_ir[:2]:
        tname = str(tree_info.get("tree") or "GeoTree")
        nodes = list(dict.fromkeys(str(n) for n in (tree_info.get("nodes") or [])))
        edges = tree_info.get("edges") or []
        if len(dag_ir) > 1:
            lines.append(f"Tree: {tname}")
        lines.append("Nodes: " + " -> ".join(nodes[:14]))
        if edges:
            lines.append("Wires:")
            for src_n, src_p, src_t, dst_n, dst_p, dst_t in edges[:16]:
                lines.append(f"  {src_n}[{src_p}:{src_t}] -> {dst_n}[{dst_p}:{dst_t}]")
    lines.append(
        "Boundary: attach NODES modifier, keep per-branch SetMaterial before JoinGeometry, "
        "SUN light, active CAMERA, Cycles 64."
    )
    return "\n".join(lines)


def _clean_subject_line(full_prompt: str) -> str:
    """Extract and scrub any API class hints from the first descriptive line of a prompt."""
    first_line = full_prompt.strip().splitlines()[0].strip()
    first_line = re.sub(
        r"\s+(?:using|with|via)\s+(?:GeometryNode|Set\s*Position|Curve\s*Spiral|Curve\s*Circle|Curve\s*to\s*Mesh|Distribute\s*Points)[^.]*\.",
        ".",
        first_line,
        flags=re.IGNORECASE,
    )
    for pat in (
        r"\bGeometryNodeTree\b",
        r"\bGeometryNode[A-Za-z0-9_]*\b",
        r"\bShaderNodeBsdfPrincipled\b",
        r"\bNodeGroupOutput\b",
        r"\bng\.interface\.new_socket\b",
        r"\bbpy\.data\.node_groups\b",
        r"\bNODES\b",
    ):
        first_line = re.sub(pat, "", first_line)
    first_line = re.sub(r"\s+procedurally\.", ".", first_line, flags=re.IGNORECASE)
    first_line = re.sub(r"\s{2,}", " ", first_line).replace(" .", ".").strip()
    return first_line


def extract_bare_subject(full_prompt: str) -> str:
    """Strip Blender API crutches and return a short natural bare Geometry Nodes prompt."""
    first_line = _clean_subject_line(full_prompt)
    templates = [
        "{s} Use Blender Geometry Nodes and save `/workspace/exports/final.blend`.",
        "Create this in Blender Geometry Nodes: {s} Save `/workspace/exports/final.blend`.",
        "Make a `/workspace/exports/final.blend` scene using Blender Geometry Nodes: {s}",
    ]
    idx = sum(ord(ch) for ch in first_line) % len(templates)
    bare = templates[idx].format(s=first_line)
    for banned in (
        "GeometryNodeTree", "ng.interface", "NodeGroupOutput", "NODES",
        "bpy.data.node_groups", "ShaderNodeBsdfPrincipled", "4.2",
    ):
        assert banned not in bare, f"Bare prompt leaked API hint '{banned}': {bare}"
    return bare


def check_c7_leakage(row_id: str, texts: list[str]) -> None:
    eval_ids = {str(t["id"]).lower() for t in bgn.HELD_OUT_EVAL_TASKS}
    assert row_id.lower() not in eval_ids, f"Eval ID collision: {row_id}"
    hay = f"{row_id} " + " ".join(texts)
    low = hay.lower()
    for term in C7_FORBIDDEN_TERMS:
        assert term not in low, f"Forbidden Cycle 7 held-out term '{term}' found in '{row_id}'"


def make_jinja_renderer() -> Any:
    env = jinja2.Environment()
    return env.from_string(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))


def render_pi_prefix(tpl: Any, messages: list[dict[str, Any]]) -> str:
    rendered = tpl.render(
        messages=messages,
        tools=PI_TOOLS,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="medium",
        preserve_reasoning=True,
        tool_call_format="xml",
    )
    assert rendered.endswith("<|im_start|>assistant\n<think>\n"), f"Unexpected tail: {rendered[-80:]!r}"
    return rendered


def build_pi_write_completion(reasoning: str, path: str, content: str) -> str:
    return (
        f"{reasoning.strip()}\n</think>\n\n"
        f"<tool_call>\n<function=write>\n"
        f"<parameter=path>\n{path}\n</parameter>\n"
        f"<parameter=content>\n{content.strip()}\n</parameter>\n"
        f"</function>\n</tool_call><|im_end|>\n"
    )


def build_repair_rows(tpl: Any) -> list[dict[str, Any]]:
    """Build 8 Pi-agent 1-step targeted `edit` traceback-repair examples for Blender 5.4 GN."""
    cases = [
        (
            "c7_rep_01_mesh_node_tree",
            "/home/nos/scripts/ring_gn.py",
            "Build `/home/nos/scripts/ring.blend` using Blender Geometry Nodes.",
            "import bpy\nobj = bpy.data.objects.new('Ring', bpy.data.meshes.new('RingM'))\n"
            "bpy.context.scene.collection.objects.link(obj)\ngt = obj.data.node_tree\n",
            "AttributeError: 'Mesh' object has no attribute 'node_tree'",
            "A Mesh has no `.node_tree`. Create `bpy.data.node_groups.new(..., 'GeometryNodeTree')`, "
            "expose an OUTPUT geometry socket, and attach a `NODES` modifier.",
            "gt = obj.data.node_tree",
            "ng = bpy.data.node_groups.new('RingGN', 'GeometryNodeTree')\n"
            "ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')\n"
            "obj.modifiers.new('RingGN', 'NODES').node_group = ng",
        ),
        (
            "c7_rep_02_modifier_node_tree",
            "/home/nos/scripts/wheel_gn.py",
            "Create a spoked wheel `.blend` with Geometry Nodes.",
            "ng = bpy.data.node_groups.new('WheelGN', 'GeometryNodeTree')\n"
            "obj = bpy.data.objects.new('Wheel', bpy.data.meshes.new('WheelM'))\n"
            "bpy.context.scene.collection.objects.link(obj)\n"
            "mod = obj.modifiers.new('WheelGN', 'NODES')\nmod.node_tree = ng\n",
            "AttributeError: 'NodesModifier' object has no attribute 'node_tree'",
            "In Blender 5.4 `NodesModifier` binds the tree via `.node_group`, not `.node_tree`.",
            "mod.node_tree = ng",
            "mod.node_group = ng",
        ),
        (
            "c7_rep_03_nodegroupoutput",
            "/home/nos/scripts/arch_gn.py",
            "Write `/home/nos/scripts/arch.blend` using Geometry Nodes.",
            "ng = bpy.data.node_groups.new('ArchGN', 'GeometryNodeTree')\n"
            "go = ng.nodes.new('GeometryNodeOutput')\n",
            "RuntimeError: Error: Node type GeometryNodeOutput undefined",
            "`GeometryNodeOutput` does not exist; use `ng.interface.new_socket('Geometry', in_out='OUTPUT', "
            "socket_type='NodeSocketGeometry')` with `ng.nodes.new('NodeGroupOutput')`.",
            "ng = bpy.data.node_groups.new('ArchGN', 'GeometryNodeTree')\ngo = ng.nodes.new('GeometryNodeOutput')",
            "ng = bpy.data.node_groups.new('ArchGN', 'GeometryNodeTree')\n"
            "ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')\n"
            "go = ng.nodes.new('NodeGroupOutput')",
        ),
        (
            "c7_rep_04_curveline",
            "/home/nos/scripts/mast_gn.py",
            "Build a rigged mast `.blend` in Geometry Nodes.",
            "ng = bpy.data.node_groups.new('MastGN', 'GeometryNodeTree')\n"
            "ln = ng.nodes.new('GeometryNodeCurveLine')\n"
            "ln.inputs['End'].default_value = (0.0, 0.0, 2.0)\n",
            "RuntimeError: Error: Node type GeometryNodeCurveLine undefined",
            "The Blender 5.4 `bl_idname` is `GeometryNodeCurvePrimitiveLine`, not `GeometryNodeCurveLine`.",
            "ln = ng.nodes.new('GeometryNodeCurveLine')",
            "ln = ng.nodes.new('GeometryNodeCurvePrimitiveLine')",
        ),
        (
            "c7_rep_05_meshtorus",
            "/home/nos/scripts/hoop_gn.py",
            "Build a brass hoop `.blend` using Geometry Nodes.",
            "tor = ng.nodes.new('GeometryNodeMeshTorus')\n"
            "tor.inputs['Major Radius'].default_value = 0.7\n",
            "RuntimeError: Error: Node type GeometryNodeMeshTorus undefined",
            "Blender 5.4 has no `GeometryNodeMeshTorus`; sweep `GeometryNodeCurvePrimitiveCircle` along a "
            "profile `GeometryNodeCurvePrimitiveCircle` with `GeometryNodeCurveToMesh`.",
            "tor = ng.nodes.new('GeometryNodeMeshTorus')\ntor.inputs['Major Radius'].default_value = 0.7",
            "rp = ng.nodes.new('GeometryNodeCurvePrimitiveCircle'); rp.inputs['Radius'].default_value = 0.7\n"
            "rpf = ng.nodes.new('GeometryNodeCurvePrimitiveCircle'); rpf.inputs['Radius'].default_value = 0.04\n"
            "tor = ng.nodes.new('GeometryNodeCurveToMesh'); tor.inputs['Fill Caps'].default_value = True\n"
            "ng.links.new(rp.outputs['Curve'], tor.inputs['Curve']); ng.links.new(rpf.outputs['Curve'], tor.inputs['Profile Curve'])",
        ),
        (
            "c7_rep_06_setmaterial",
            "/home/nos/scripts/lamp_gn.py",
            "Assign gold material cleanly inside Geometry Nodes.",
            "sm = ng.nodes.new('GeometryNodeReplaceMaterial')\n"
            "sm.inputs['Material'].default_value = m_gold\n",
            "KeyError: 'bpy_prop_collection[key]: key \"Material\" not found'",
            "`GeometryNodeReplaceMaterial` has `Old`/`New` sockets; use `GeometryNodeSetMaterial` to assign `m_gold`.",
            "sm = ng.nodes.new('GeometryNodeReplaceMaterial')\nsm.inputs['Material'].default_value = m_gold",
            "sm = ng.nodes.new('GeometryNodeSetMaterial')\nsm.inputs['Material'].default_value = m_gold",
        ),
        (
            "c7_rep_07_curvecircle",
            "/home/nos/scripts/collar_gn.py",
            "Sweep a metallic collar ring in Geometry Nodes.",
            "rp = ng.nodes.new('GeometryNodeMeshCircle')\n"
            "c2m = ng.nodes.new('GeometryNodeCurveToMesh')\n"
            "ng.links.new(rp.outputs['Curve'], c2m.inputs['Curve'])\n",
            "KeyError: 'bpy_prop_collection[key]: key \"Curve\" not found'",
            "`GeometryNodeMeshCircle` outputs `Mesh`; `GeometryNodeCurveToMesh` requires `GeometryNodeCurvePrimitiveCircle`.",
            "rp = ng.nodes.new('GeometryNodeMeshCircle')",
            "rp = ng.nodes.new('GeometryNodeCurvePrimitiveCircle')",
        ),
        (
            "c7_rep_08_light_enum_vec3",
            "/home/nos/scripts/pedestal_gn.py",
            "Fix transform vector and light type in `/home/nos/scripts/pedestal_gn.py`.",
            "t1.inputs['Translation'].default_value = (0.0, 0.0, 0.5, 1.0)\n"
            "bpy.ops.object.light_add(type='Point', location=(0, 0, 2.5))\n",
            "ValueError: bpy_struct: item.attr = val: sequences of dimension 0 should contain 3 items, not 4",
            "`Translation` takes a 3-tuple `(x, y, z)` and Blender light types must be uppercase (`'POINT'` / `'SUN'`).",
            "t1.inputs['Translation'].default_value = (0.0, 0.0, 0.5, 1.0)\n"
            "bpy.ops.object.light_add(type='Point', location=(0, 0, 2.5))",
            "t1.inputs['Translation'].default_value = (0.0, 0.0, 0.5)\n"
            "bpy.ops.object.light_add(type='POINT', location=(0, 0, 2.5))",
        ),
    ]
    out: list[dict[str, Any]] = []
    for rid, fpath, u_prompt, bad_snippet, err_line, think_txt, old_t, new_t in cases:
        msgs = [
            {"role": "system", "content": PI_SYSTEM},
            {"role": "user", "content": u_prompt},
            {
                "role": "assistant",
                "reasoning_content": f"I will write `{fpath}` and run headless Blender 5.4.",
                "content": "",
                "tool_calls": [
                    {"id": "c_w", "type": "function", "function": {"name": "write", "arguments": {"path": fpath, "content": bad_snippet}}},
                    {"id": "c_b", "type": "function", "function": {"name": "bash", "arguments": {"command": f"/home/nos/blender-5.4.0/blender --background --factory-startup --python {fpath}"}}},
                ],
            },
            {"role": "tool", "tool_call_id": "c_w", "content": f"Successfully wrote to {fpath}"},
            {"role": "tool", "tool_call_id": "c_b", "content": f"Traceback (most recent call last):\n  File \"{fpath}\", line 4\n{err_line}\nEXIT: 1"},
        ]
        prefix = render_pi_prefix(tpl, msgs)
        comp = (
            f"{think_txt}\n</think>\n\n"
            f"<tool_call>\n<function=edit>\n"
            f"<parameter=path>\n{fpath}\n</parameter>\n"
            f"<parameter=oldText>\n{old_t}\n</parameter>\n"
            f"<parameter=newText>\n{new_t}\n</parameter>\n"
            f"</function>\n</tool_call><|im_end|>\n"
        )
        out.append({
            "id": rid,
            "task_id": rid,
            "domain": "geonodes",
            "source": "c7_pi_repair",
            "prefix_text": prefix,
            "completion_text": comp,
            "prompt": u_prompt,
            "reasoning": think_txt,
            "response": comp,
        })
    return out


def build_unload_rows(tpl: Any) -> list[dict[str, Any]]:
    """Build 4 Pi-agent post-verification `lora_ctl.py unload` discipline examples."""
    cases = [
        ("c7_unl_01_crown", "/home/nos/scripts/crown_gn.py", "/home/nos/scripts/crown.blend", 1840),
        ("c7_unl_02_wheel", "/home/nos/scripts/wheel_gn.py", "/home/nos/scripts/wheel.blend", 2160),
        ("c7_unl_03_lantern", "/home/nos/scripts/lantern_gn.py", "/home/nos/scripts/lantern.blend", 1592),
        ("c7_unl_04_rotor", "/home/nos/scripts/rotor_gn.py", "/home/nos/scripts/rotor.blend", 1928),
    ]
    out: list[dict[str, Any]] = []
    for rid, fpath, bpath, verts in cases:
        u_prompt = f"Use the lora-router skill and Blender Geometry Nodes to create `{bpath}`."
        msgs = [
            {"role": "system", "content": PI_SYSTEM},
            {"role": "user", "content": u_prompt},
            {
                "role": "assistant",
                "reasoning_content": "Loading `geonodes` SME before writing the Blender 5.4 script.",
                "content": "",
                "tool_calls": [
                    {"id": "c_l", "type": "function", "function": {"name": "bash", "arguments": {"command": "/home/nos/.pi/agent/skills/lora-router/scripts/lora_ctl.py load geonodes"}}},
                ],
            },
            {"role": "tool", "tool_call_id": "c_l", "content": "[lora_ctl] load(Bonsai-2-27B-blender-geonodes:1) applied & verified in 0.58 ms"},
            {
                "role": "assistant",
                "reasoning_content": f"Writing `{fpath}` and verifying `{bpath}` in headless Blender 5.4.",
                "content": "",
                "tool_calls": [
                    {"id": "c_r", "type": "function", "function": {"name": "bash", "arguments": {"command": f"/home/nos/blender-5.4.0/blender --background --factory-startup --python {fpath}"}}},
                ],
            },
            {"role": "tool", "tool_call_id": "c_r", "content": f"Info: Saved as \"{bpath}\"\nVERIFIED: valid=True total_verts={verts} reaches_output=True\nBlender quit"},
        ]
        prefix = render_pi_prefix(tpl, msgs)
        think_txt = f"`{bpath}` is saved and verified ({verts} verts). I must unload the `geonodes` specialist now before replying."
        comp = (
            f"{think_txt}\n</think>\n\n"
            f"<tool_call>\n<function=bash>\n"
            f"<parameter=command>\n/home/nos/.pi/agent/skills/lora-router/scripts/lora_ctl.py unload\n</parameter>\n"
            f"</function>\n</tool_call><|im_end|>\n"
        )
        out.append({
            "id": rid,
            "task_id": rid,
            "domain": "geonodes",
            "source": "c7_pi_unload",
            "prefix_text": prefix,
            "completion_text": comp,
            "prompt": u_prompt,
            "reasoning": think_txt,
            "response": comp,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "work" / "curriculum_c7_gn_sme.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2500)
    args = ap.parse_args()

    schema = bs54.extract_schema(out_json=bs54.DEFAULT_SCHEMA_JSON)
    tok = tlora.load_qwen35_tokenizer()
    tpl = make_jinja_renderer()

    # 1. Gather raw candidate pools
    c4_all = [json.loads(l) for l in (SCRIPT_DIR / "work" / "curriculum_c4_compact_geonodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    c4_rows = [r for r in c4_all if r.get("source") == "c4_compact_geonodes"]
    assert len(c4_rows) == 60

    c5_rows = bc5._build_c5_spiral_recipes()
    assert len(c5_rows) == 10

    rl_raw = [json.loads(l) for l in (SCRIPT_DIR / "work" / "rl_gn_harvested_winners.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rad_rows = _build_radial_recipes()
    assert len(rad_rows) == 16

    # 2. Canonicalize, schema-lint, and anti-leakage filter
    staged: list[dict[str, Any]] = []
    seen_rl_tids: set[str] = set()
    skipped_rl: list[str] = []

    raw_items: list[tuple[str, dict[str, Any]]] = (
        [("c7_radial_gn", r) for r in rad_rows]
        + [("c4_gn", r) for r in c4_rows]
        + [("c5_spiral_gn", r) for r in c5_rows]
        + [("rl_gn_winner", r) for r in rl_raw]
    )
    for src_group, raw in raw_items:
        rid = str(raw["id"])
        tid = str(raw.get("task_id") or rid)
        if src_group == "rl_gn_winner":
            if tid in seen_rl_tids:
                continue
            seen_rl_tids.add(tid)
        canon_code = bc6.canonicalize_blender54_code(str(raw["response"]))
        prompt = str(raw["prompt"]).strip()

        if src_group == "rl_gn_winner":
            lint_errs = bs54.lint_geonodes_script(canon_code, schema, enforce_canonical_boundary=True)
            if lint_errs:
                skipped_rl.append(f"{rid}: {lint_errs[0]}")
                continue
        else:
            lint_errs = bs54.lint_geonodes_script(canon_code, schema, enforce_canonical_boundary=True)
            assert not lint_errs, f"Lint errors in {rid}:\n" + "\n".join(lint_errs)

        # Skip any legacy C4/C5/RL row overlapping new C7 held-out eval terms (keeps 0 eval leakage),
        # while asserting hard on all C6 eval terms & all new C7 radial recipes.
        try:
            check_c7_leakage(tid, [prompt, canon_code])
        except AssertionError as exc:
            if src_group != "c7_radial_gn":
                skipped_rl.append(f"{rid}: {exc}")
                continue
            raise

        # All 16 new radial recipes must have >=2 SetMaterial calls before JoinGeometry
        if src_group == "c7_radial_gn":
            assert canon_code.count("GeometryNodeSetMaterial") >= 2, f"{rid} missing >=2 SetMaterial"
            assert "GeometryNodeInstanceOnPoints" in canon_code, f"{rid} missing InstanceOnPoints"

        staged.append({
            "id": rid,
            "task_id": tid,
            "source": src_group,
            "prompt": prompt,
            "code": canon_code.strip(),
        })

    print(
        f"[c7_data] linted {len(staged)} recipes (radial={sum(1 for r in staged if r['source']=='c7_radial_gn')}, "
        f"c4=60, c5=10, rl={sum(1 for r in staged if r['source']=='rl_gn_winner')}, skipped_rl={skipped_rl})",
        flush=True,
    )

    # 3. Headless Blender 5.4 Structural Verification across all staged recipes
    print(f"[c7_data] running Blender 5.4 structural verifier on {len(staged)} recipes (workers={args.workers})...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        vrs = list(pool.map(bv.verify_script, [f"```python\n{r['code']}\n```" for r in staged]))

    verified_pool: list[dict[str, Any]] = []
    for row, vr in zip(staged, vrs):
        ss = vr.scene_summary or {}
        gp = bv.is_geonodes_pass(vr)
        verts = int(ss.get("total_verts", 0))
        min_v = 100 if row["source"] == "c7_radial_gn" else 32
        ok = (
            vr.syntax_ok
            and vr.exec_ok
            and gp
            and ss.get("geonode_invalid_links", 1) == 0
            and ss.get("geonode_reaches_output") is True
            and verts >= min_v
            and vr.reward >= 0.85
        )
        if not ok:
            if row["source"] == "rl_gn_winner":
                skipped_rl.append(f"{row['id']}: blender verify fail R={vr.reward} err={vr.error[:120]}")
                continue
            raise SystemExit(
                f"Blender 5.4 verify FAILED for {row['id']}: syn={vr.syntax_ok} exec={vr.exec_ok} gp={gp} "
                f"inv={ss.get('geonode_invalid_links')} reach={ss.get('geonode_reaches_output')} verts={verts} err={vr.error[:240]}"
            )
        row["reward"] = vr.reward
        row["verts"] = verts
        row["gn_nodes"] = ss.get("geonode_nodes_total", 0)
        row["gn_links"] = ss.get("geonode_valid_links", 0)
        row["_dag_ir"] = ss.get("geonode_dag_ir") or []
        verified_pool.append(row)

    print(f"[c7_data] {len(verified_pool)}/{len(staged)} PASSED Blender 5.4 structural verification (0 invalid links)", flush=True)

    # 4. Partition verified_pool into Group C (16 Pi `write` rows), Group B (bare single-turn), Group A (full-prompt single-turn)
    radial_pool = [r for r in verified_pool if r["source"] == "c7_radial_gn"]
    other_pool = [r for r in verified_pool if r["source"] != "c7_radial_gn"]

    # Choose 10 radial + 6 diverse non-radial for Pi `write` tool-call Group C (16 rows total)
    pi_write_src = radial_pool[:10] + other_pool[:: max(1, len(other_pool) // 6)][:6]
    pi_write_ids = {r["id"] for r in pi_write_src}
    single_turn_pool = [r for r in verified_pool if r["id"] not in pi_write_ids]

    # Among single_turn_pool, split ~half (and all remaining radial) into Group B bare prompts, rest Group A full prompts
    # Put all remaining 6 radial in Group B bare prompts + every other single_turn item!
    group_b_src: list[dict[str, Any]] = []
    group_a_src: list[dict[str, Any]] = []
    toggle = 0
    for r in single_turn_pool:
        if r["source"] == "c7_radial_gn":
            group_b_src.append(r)
        else:
            if toggle % 2 == 0:
                group_b_src.append(r)
            else:
                group_a_src.append(r)
            toggle += 1

    cutoff_msg = bbw.DEFAULT_REASONING_BUDGET_MESSAGE.strip()
    final_rows: list[dict[str, Any]] = []

    def add_single_turn_rows(src_list: list[dict[str, Any]], bare: bool, reason_every: int, max_cutoff: int) -> None:
        cutoff_done = 0
        for idx, r in enumerate(src_list):
            p_txt = extract_bare_subject(r["prompt"]) if bare else r["prompt"]
            ir_txt = ""
            if idx % reason_every == 0:
                cand_ir = format_pin_flow_ir_c7(r["_dag_ir"])
                if cutoff_done < max_cutoff and idx % (reason_every * 3) == 0:
                    cand_ir = cand_ir + "\n" + cutoff_msg
                probe = {
                    "system": bcg.GN_SYSTEM_PROMPT,
                    "prompt": p_txt,
                    "preserve_prompt": True,
                    "reasoning": cand_ir,
                    "response": f"```python\n{r['code']}\n```",
                }
                if len(tlora.format_conversation(probe, tok, max_seq_len=99999, truncate=False)[0]) <= args.max_tokens:
                    ir_txt = cand_ir
                    if cutoff_msg in cand_ir:
                        cutoff_done += 1
            row = {
                "id": f"{r['id']}_{'bare' if bare else 'full'}",
                "task_id": r["task_id"],
                "domain": "geonodes",
                "source": f"{r['source']}_{'bare' if bare else 'full'}",
                "system": bcg.GN_SYSTEM_PROMPT,
                "prompt": p_txt,
                "preserve_prompt": True,
                "reasoning": ir_txt,
                "response": f"```python\n{r['code']}\n```",
                "reward": r["reward"],
            }
            final_rows.append(row)

    add_single_turn_rows(group_a_src, bare=False, reason_every=3, max_cutoff=3)
    add_single_turn_rows(group_b_src, bare=True, reason_every=2, max_cutoff=3)

    # Group C: 16 Pi-agent `write` tool-call rows
    for idx, r in enumerate(pi_write_src):
        slug = re.sub(r"[^a-z0-9]+", "_", r["id"].lower()).strip("_")[:28]
        py_path = f"/home/nos/scripts/{slug}.py"
        blend_path = f"/home/nos/scripts/{slug}.blend"
        bare_desc = _clean_subject_line(r["prompt"])
        u_prompt = f"Use Blender Geometry Nodes to write `{py_path}` and build `{blend_path}`: {bare_desc}"
        code_pi = r["code"].replace("/workspace/exports/final.blend", blend_path)
        prefix = render_pi_prefix(tpl, [
            {"role": "system", "content": PI_SYSTEM},
            {"role": "user", "content": u_prompt},
        ])
        short_ir = (
            f"Pin-Flow plan: GeometryNodeTree -> ng.interface.new_socket -> NodeGroupOutput + NODES modifier -> write `{py_path}`."
        )
        ir_txt = short_ir
        if idx % 2 == 0:
            full_ir = format_pin_flow_ir_c7(r["_dag_ir"])
            comp_probe = build_pi_write_completion(full_ir, py_path, code_pi)
            if len(tlora.format_conversation({"prefix_text": prefix, "completion_text": comp_probe}, tok, max_seq_len=99999, truncate=False)[0]) <= args.max_tokens:
                ir_txt = full_ir
        comp = build_pi_write_completion(ir_txt, py_path, code_pi)
        final_rows.append({
            "id": f"{r['id']}_pi_write",
            "task_id": r["task_id"],
            "domain": "geonodes",
            "source": "c7_pi_write",
            "prefix_text": prefix,
            "completion_text": comp,
            "prompt": u_prompt,
            "reasoning": ir_txt,
            "response": comp,
            "reward": r["reward"],
        })

    # Groups D & E: Pi repair & unload traces
    final_rows.extend(build_repair_rows(tpl))
    final_rows.extend(build_unload_rows(tpl))

    # 5. Strict global anti-leakage + tokenization verification
    tok_lens: list[int] = []
    over_budget: list[tuple[str, int]] = []
    for row in final_rows:
        check_c7_leakage(
            str(row["id"]),
            [
                str(row.get("prompt", "")),
                str(row.get("reasoning", "")),
                str(row.get("response", "")),
                str(row.get("prefix_text", "")),
                str(row.get("completion_text", "")),
            ],
        )
        ids, plen = tlora.format_conversation(row, tok, max_seq_len=99999, truncate=False)
        n_tok = len(ids)
        row["tokens"] = n_tok
        row["prefix_tokens"] = plen
        tok_lens.append(n_tok)
        if n_tok > args.max_tokens:
            over_budget.append((row["id"], n_tok))

    if over_budget:
        raise SystemExit(f"Rows exceed max_tokens={args.max_tokens}: {over_budget}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in final_rows) + "\n", encoding="utf-8")

    by_src: dict[str, int] = {}
    for r in final_rows:
        s = r["source"]
        by_src[s] = by_src.get(s, 0) + 1
    n_bare = sum(1 for r in final_rows if "_bare" in r["source"] or r["source"] == "c7_pi_write")
    print(f"\n=== CYCLE 7 GEONODES SME DATASET READY: {args.out} ===", flush=True)
    print(f"  Total rows       : {len(final_rows)} (0 leaked, 0 over {args.max_tokens} tok)", flush=True)
    print(f"  Zero-crutch bare : {n_bare}/{len(final_rows)} ({100*n_bare/len(final_rows):.1f}%)", flush=True)
    print(f"  By source        : {json.dumps(by_src)}", flush=True)
    print(f"  Token lengths    : min={min(tok_lens)}, med={int(statistics.median(tok_lens))}, max={max(tok_lens)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
