#!/usr/bin/env python3
"""Build, anti-leakage check, and Blender-5.4 verify the Cycle 5 Unified Blender + GeoNodes SFT dataset."""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import json
import os
import statistics
import sys
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_blenderworld as bbw  # noqa: E402
import benchmark_geonodes as bgn  # noqa: E402
import blender_verifier as bv  # noqa: E402
import build_compact_geonodes_dataset as bcg  # noqa: E402
import train_lora as tlora  # noqa: E402

BLENDERWORLD_SYS_PROMPT = (
    "You are an expert Blender 4.2 Python programmer. Generate clean, self-contained, executable `bpy` code "
    "compatible with Blender 4.2.10."
)


def _check_all_text_forbidden(id_str: str, prompt: str, reasoning: str, code: str) -> None:
    """Assert zero leakage against benchmark/RL forbidden substrings across all text fields."""
    hay = f"{id_str} {prompt} {reasoning} {code}".lower()
    for bad in bcg.FORBIDDEN_TERMS:
        if bad in hay:
            raise AssertionError(f"Forbidden term '{bad}' found in recipe '{id_str}'")


def _build_c5_spiral_recipes() -> list[dict[str, Any]]:
    """Build 10 compact Blender 5.4 spiral/curve sweep Geometry Nodes recipes (c5_sp_01..10)."""
    specs: list[tuple[str, str, str, str]] = [
        (
            "c5_sp_01_corkscrew_auger",
            "Build a steel helical wine corkscrew auger bit with a tapered spiral tip mounted into a turned walnut T-handle.",
            """\
            m_steel = make_mat("AugerSteel", (0.78, 0.80, 0.84), rough=0.22, metallic=0.92)
            m_wood = make_mat("WalnutHandle", (0.28, 0.14, 0.06), rough=0.45, metallic=0.0)
            ng = bpy.data.node_groups.new("CorkscrewAuger", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 48
            sp.inputs["Rotations"].default_value = 5.5
            sp.inputs["Start Radius"].default_value = 0.075
            sp.inputs["End Radius"].default_value = 0.055
            sp.inputs["Height"].default_value = 0.95
            spar = ng.nodes.new("GeometryNodeSplineParameter")
            mr = ng.nodes.new("ShaderNodeMapRange")
            mr.inputs["From Min"].default_value = 0.0
            mr.inputs["From Max"].default_value = 1.0
            mr.inputs["To Min"].default_value = 0.022
            mr.inputs["To Max"].default_value = 0.006
            sr = ng.nodes.new("GeometryNodeSetCurveRadius")
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
            prof.inputs["Resolution"].default_value = 12
            prof.inputs["Radius"].default_value = 1.0
            c2m = ng.nodes.new("GeometryNodeCurveToMesh")
            c2m.inputs["Fill Caps"].default_value = True
            sh = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_steel
            ng.links.new(spar.outputs["Factor"], mr.inputs["Value"])
            ng.links.new(sp.outputs["Curve"], sr.inputs["Curve"])
            ng.links.new(mr.outputs["Result"], sr.inputs["Radius"])
            ng.links.new(sr.outputs["Curve"], c2m.inputs["Curve"])
            ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sh.inputs["Geometry"])
            ng.links.new(sh.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            auger = bpy.data.objects.new("Auger", bpy.data.meshes.new("AugerM"))
            auger.location = (0, 0, 0.25)
            bpy.context.scene.collection.objects.link(auger)
            auger.modifiers.new("AugerGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.055, depth=0.35, location=(0, 0, 1.38))
            bpy.context.active_object.data.materials.append(m_steel)
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.09, depth=1.15, location=(0, 0, 1.58), rotation=(0, 1.5708, 0))
            bpy.context.active_object.data.materials.append(m_wood)
            """,
            (
                "Plan: Create GeometryNodeCurveSpiral with Start Radius=0.075, End Radius=0.055, Rotations=5.5, Height=0.95, "
                "Resolution=48. Modulate radius along SplineParameter Factor via MapRange -> GeometryNodeSetCurveRadius, sweep with "
                "GeometryNodeCurvePrimitiveCircle (Radius=1.0) + GeometryNodeCurveToMesh (Fill Caps=True), apply SetShadeSmooth "
                "and Steel SetMaterial, then add shaft and Walnut wood T-handle."
            ),
        ),
        (
            "c5_sp_02_nautilus_shell",
            "Build a pearlescent conical nautilus shell swept along an expanding spiral path with gradually widening circular cross-section on a dark slate display cradle.",
            """\
            m_shell = pbr_mat("PearlShell", (0.90, 0.83, 0.72), roughness=0.28, metal=0.0)
            m_slate = pbr_mat("SlateBase", (0.14, 0.15, 0.17), roughness=0.62, metal=0.1)
            ng = bpy.data.node_groups.new("NautilusShell", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 48
            sp.inputs["Rotations"].default_value = 3.5
            sp.inputs["Start Radius"].default_value = 0.04
            sp.inputs["End Radius"].default_value = 0.72
            sp.inputs["Height"].default_value = 0.55
            spar = ng.nodes.new("GeometryNodeSplineParameter")
            mr = ng.nodes.new("ShaderNodeMapRange")
            mr.inputs["To Min"].default_value = 0.02
            mr.inputs["To Max"].default_value = 0.23
            sr = ng.nodes.new("GeometryNodeSetCurveRadius")
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
            prof.inputs["Resolution"].default_value = 20
            prof.inputs["Radius"].default_value = 1.0
            c2m = ng.nodes.new("GeometryNodeCurveToMesh")
            c2m.inputs["Fill Caps"].default_value = True
            sh = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_shell
            ng.links.new(spar.outputs["Factor"], mr.inputs["Value"])
            ng.links.new(sp.outputs["Curve"], sr.inputs["Curve"])
            ng.links.new(mr.outputs["Result"], sr.inputs["Radius"])
            ng.links.new(sr.outputs["Curve"], c2m.inputs["Curve"])
            ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sh.inputs["Geometry"])
            ng.links.new(sh.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            sh_obj = bpy.data.objects.new("Nautilus", bpy.data.meshes.new("NautilusM"))
            sh_obj.location = (0, 0, 0.35)
            bpy.context.scene.collection.objects.link(sh_obj)
            sh_obj.modifiers.new("ShellGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.95, depth=0.14, location=(0, 0, 0.07))
            bpy.context.active_object.data.materials.append(m_slate)
            """,
            (
                "Plan: Use GeometryNodeCurveSpiral (Start Radius=0.04, End Radius=0.72, Rotations=3.5, Height=0.55, "
                "Resolution=48) paired with SplineParameter -> MapRange (0.02..0.23) -> GeometryNodeSetCurveRadius. Sweep using "
                "GeometryNodeCurvePrimitiveCircle (Resolution=20, Radius=1.0) into GeometryNodeCurveToMesh (Fill Caps=True), "
                "smooth and shade PearlShell, resting on a dark slate display cylinder."
            ),
        ),
        (
            "c5_sp_03_brass_horn",
            "Build a coiled polished brass hunting horn with a narrow mouthpiece tube flaring continuously into a wide bell mouth resting on a dark velvet block.",
            """\
            m_brass = make_mat("PolishedBrass", (0.88, 0.68, 0.24), rough=0.18, metallic=0.95)
            m_velvet = make_mat("DarkVelvet", (0.16, 0.04, 0.06), rough=0.82, metallic=0.0)
            ng = bpy.data.node_groups.new("BrassHorn", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 40
            sp.inputs["Rotations"].default_value = 2.2
            sp.inputs["Start Radius"].default_value = 0.44
            sp.inputs["End Radius"].default_value = 0.50
            sp.inputs["Height"].default_value = 0.22
            spar = ng.nodes.new("GeometryNodeSplineParameter")
            pw = ng.nodes.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 3.0
            mr = ng.nodes.new("ShaderNodeMapRange")
            mr.inputs["To Min"].default_value = 0.022
            mr.inputs["To Max"].default_value = 0.17
            sr = ng.nodes.new("GeometryNodeSetCurveRadius")
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
            prof.inputs["Resolution"].default_value = 20
            prof.inputs["Radius"].default_value = 1.0
            c2m = ng.nodes.new("GeometryNodeCurveToMesh")
            c2m.inputs["Fill Caps"].default_value = False
            sh = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_brass
            ng.links.new(spar.outputs["Factor"], pw.inputs[0])
            ng.links.new(pw.outputs["Value"], mr.inputs["Value"])
            ng.links.new(sp.outputs["Curve"], sr.inputs["Curve"])
            ng.links.new(mr.outputs["Result"], sr.inputs["Radius"])
            ng.links.new(sr.outputs["Curve"], c2m.inputs["Curve"])
            ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sh.inputs["Geometry"])
            ng.links.new(sh.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            horn = bpy.data.objects.new("Horn", bpy.data.meshes.new("HornM"))
            horn.location = (0, 0, 0.34)
            bpy.context.scene.collection.objects.link(horn)
            horn.modifiers.new("HornGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.11))
            bpy.context.active_object.scale = (1.35, 1.35, 0.22)
            bpy.context.active_object.data.materials.append(m_velvet)
            """,
            (
                "Plan: Build GeometryNodeCurveSpiral (Start Radius=0.44, End Radius=0.50, Rotations=2.2, Height=0.22, "
                "Resolution=40), apply cubic bell flare with SplineParameter -> POWER(3.0) -> MapRange (0.022..0.17) -> "
                "GeometryNodeSetCurveRadius, sweep with GeometryNodeCurvePrimitiveCircle (Radius=1.0) into CurveToMesh "
                "(Fill Caps=False), smooth and shade PolishedBrass above a dark velvet display block."
            ),
        ),
        (
            "c5_sp_04_wrought_trellis",
            "Build a dark wrought-iron twin-helix garden trellis with two intertwined vertical spiral bars framing a slim central post in a terracotta planter cube.",
            """\
            m_iron = pbr_mat("WroughtIron", (0.14, 0.14, 0.16), roughness=0.52, metal=0.88)
            m_clay = pbr_mat("Terracotta", (0.68, 0.30, 0.18), roughness=0.68, metal=0.0)
            ng = bpy.data.node_groups.new("TwinTrellis", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 36
            sp.inputs["Rotations"].default_value = 4.0
            sp.inputs["Start Radius"].default_value = 0.24
            sp.inputs["End Radius"].default_value = 0.24
            sp.inputs["Height"].default_value = 2.3
            tr = ng.nodes.new("GeometryNodeTransform")
            tr.inputs["Rotation"].default_value = (0.0, 0.0, 3.14159)
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
            prof.inputs["Resolution"].default_value = 10
            prof.inputs["Radius"].default_value = 0.024
            c2m = ng.nodes.new("GeometryNodeCurveToMesh")
            c2m.inputs["Fill Caps"].default_value = True
            sh = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_iron
            ng.links.new(sp.outputs["Curve"], tr.inputs["Geometry"])
            ng.links.new(sp.outputs["Curve"], jn.inputs["Geometry"])
            ng.links.new(tr.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], c2m.inputs["Curve"])
            ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sh.inputs["Geometry"])
            ng.links.new(sh.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            trellis = bpy.data.objects.new("Trellis", bpy.data.meshes.new("TrellisM"))
            trellis.location = (0, 0, 0.45)
            bpy.context.scene.collection.objects.link(trellis)
            trellis.modifiers.new("TrellisGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.032, depth=2.35, location=(0, 0, 1.6))
            bpy.context.active_object.data.materials.append(m_iron)
            bpy.ops.mesh.primitive_cube_add(size=0.68, location=(0, 0, 0.34))
            bpy.context.active_object.data.materials.append(m_clay)
            """,
            (
                "Plan: Configure GeometryNodeCurveSpiral (Start Radius=0.24, End Radius=0.24, Rotations=4.0, Height=2.3, "
                "Resolution=36), duplicate rotated pi rad via GeometryNodeTransform -> JoinGeometry, sweep twin helixes with "
                "GeometryNodeCurvePrimitiveCircle (Radius=0.024) + CurveToMesh (Fill Caps=True), and add central post + planter cube."
            ),
        ),
        (
            "c5_sp_05_striped_pillar",
            "Build a limestone architectural column wrapped with twin offset bronze spiral relief bands anchored between square stone plinth and capital blocks.",
            """\
            m_stone = make_mat("Limestone", (0.80, 0.77, 0.70), rough=0.45, metallic=0.0)
            m_bronze = make_mat("ReliefBronze", (0.72, 0.48, 0.22), rough=0.28, metallic=0.90)
            ng = bpy.data.node_groups.new("SpiralPillar", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            cyl = ng.nodes.new("GeometryNodeMeshCylinder")
            cyl.inputs["Vertices"].default_value = 36
            cyl.inputs["Radius"].default_value = 0.34
            cyl.inputs["Depth"].default_value = 2.3
            tr_c = ng.nodes.new("GeometryNodeTransform"); tr_c.inputs["Translation"].default_value = (0.0, 0.0, 1.15)
            sm_c = ng.nodes.new("GeometryNodeSetMaterial"); sm_c.inputs["Material"].default_value = m_stone
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 36
            sp.inputs["Rotations"].default_value = 3.0
            sp.inputs["Start Radius"].default_value = 0.35
            sp.inputs["End Radius"].default_value = 0.35
            sp.inputs["Height"].default_value = 2.25
            tr_s = ng.nodes.new("GeometryNodeTransform"); tr_s.inputs["Rotation"].default_value = (0.0, 0.0, 3.14159)
            jn_s = ng.nodes.new("GeometryNodeJoinGeometry")
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
            prof.inputs["Resolution"].default_value = 10
            prof.inputs["Radius"].default_value = 0.032
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm_b = ng.nodes.new("GeometryNodeSetMaterial"); sm_b.inputs["Material"].default_value = m_bronze
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(cyl.outputs["Mesh"], tr_c.inputs["Geometry"])
            ng.links.new(tr_c.outputs["Geometry"], sm_c.inputs["Geometry"])
            ng.links.new(sp.outputs["Curve"], tr_s.inputs["Geometry"])
            ng.links.new(sp.outputs["Curve"], jn_s.inputs["Geometry"])
            ng.links.new(tr_s.outputs["Geometry"], jn_s.inputs["Geometry"])
            ng.links.new(jn_s.outputs["Geometry"], c2m.inputs["Curve"])
            ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm_b.inputs["Geometry"])
            ng.links.new(sm_c.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(sm_b.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], out.inputs["Geometry"])
            col = bpy.data.objects.new("Pillar", bpy.data.meshes.new("PillarM"))
            col.location = (0, 0, 0.16)
            bpy.context.scene.collection.objects.link(col)
            col.modifiers.new("PillarGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.08)); bpy.context.active_object.scale = (0.95, 0.95, 0.16); bpy.context.active_object.data.materials.append(m_stone)
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 2.54)); bpy.context.active_object.scale = (0.95, 0.95, 0.16); bpy.context.active_object.data.materials.append(m_stone)
            """,
            (
                "Plan: Combine Limestone MeshCylinder (Radius=0.34, Depth=2.3 shifted to +1.15 Z) with twin phase-shifted "
                "GeometryNodeCurveSpiral bands (Start Radius=0.35, End Radius=0.35, Rotations=3.0, Height=2.25) swept by "
                "GeometryNodeCurvePrimitiveCircle (Radius=0.032) via CurveToMesh (Fill Caps=True) and shaded in ReliefBronze."
            ),
        ),
        (
            "c5_sp_06_lamp_gooseneck",
            "Build a modern matte-black studio desk lamp featuring a circular weighted base, curved spiral gooseneck arm, and angled conical metal shade.",
            """\
            m_black = pbr_mat("MatteCharcoal", (0.10, 0.11, 0.12), roughness=0.42, metal=0.65)
            m_bulb = pbr_mat("WarmBulb", (1.0, 0.88, 0.65), roughness=0.2, metal=0.0)
            ng = bpy.data.node_groups.new("StudioLamp", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            base = ng.nodes.new("GeometryNodeMeshCylinder")
            base.inputs["Vertices"].default_value = 32
            base.inputs["Radius"].default_value = 0.36
            base.inputs["Depth"].default_value = 0.08
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 40
            sp.inputs["Rotations"].default_value = 1.6
            sp.inputs["Start Radius"].default_value = 0.10
            sp.inputs["End Radius"].default_value = 0.18
            sp.inputs["Height"].default_value = 1.05
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
            prof.inputs["Resolution"].default_value = 12
            prof.inputs["Radius"].default_value = 0.028
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            cone = ng.nodes.new("GeometryNodeMeshCone")
            cone.inputs["Vertices"].default_value = 28
            cone.inputs["Radius Top"].default_value = 0.07
            cone.inputs["Radius Bottom"].default_value = 0.24
            cone.inputs["Depth"].default_value = 0.34
            tr_h = ng.nodes.new("GeometryNodeTransform")
            tr_h.inputs["Translation"].default_value = (-0.16, -0.08, 1.05)
            tr_h.inputs["Rotation"].default_value = (0.45, 0.35, 0.0)
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            sh = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_black
            ng.links.new(sp.outputs["Curve"], c2m.inputs["Curve"])
            ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(cone.outputs["Mesh"], tr_h.inputs["Geometry"])
            ng.links.new(base.outputs["Mesh"], jn.inputs["Geometry"])
            ng.links.new(c2m.outputs["Mesh"], jn.inputs["Geometry"])
            ng.links.new(tr_h.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], sh.inputs["Geometry"])
            ng.links.new(sh.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            lamp = bpy.data.objects.new("DeskLamp", bpy.data.meshes.new("LampM"))
            lamp.location = (0, 0, 0.04)
            bpy.context.scene.collection.objects.link(lamp)
            lamp.modifiers.new("LampGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08, location=(-0.18, -0.14, 0.96))
            bpy.context.active_object.data.materials.append(m_bulb)
            """,
            (
                "Plan: Assemble StudioLamp GeometryNodeTree joining MeshCylinder weighted base (Radius=0.36), helical neck via "
                "GeometryNodeCurveSpiral (Start Radius=0.10, End Radius=0.18, Rotations=1.6, Height=1.05) + GeometryNodeCurvePrimitiveCircle "
                "(Radius=0.028) -> CurveToMesh (Fill Caps=True), tilted MeshCone shade, and WarmBulb inner sphere."
            ),
        ),
        (
            "c5_sp_07_archimedes_screw",
            "Build an industrial steel Archimedes screw conveyor featuring a central cylindrical axle wrapped in a wide rectangular-blade helix between pillow-block bearings.",
            """\
            m_steel = make_mat("MachinedSteel", (0.72, 0.74, 0.78), rough=0.28, metallic=0.90)
            m_cast = make_mat("CastIronBlock", (0.20, 0.22, 0.24), rough=0.58, metallic=0.75)
            ng = bpy.data.node_groups.new("ArchimedesScrew", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            axle = ng.nodes.new("GeometryNodeMeshCylinder")
            axle.inputs["Vertices"].default_value = 28
            axle.inputs["Radius"].default_value = 0.11
            axle.inputs["Depth"].default_value = 2.2
            tr_a = ng.nodes.new("GeometryNodeTransform"); tr_a.inputs["Translation"].default_value = (0.0, 0.0, 1.1)
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 40
            sp.inputs["Rotations"].default_value = 6.0
            sp.inputs["Start Radius"].default_value = 0.21
            sp.inputs["End Radius"].default_value = 0.21
            sp.inputs["Height"].default_value = 1.95
            tr_s = ng.nodes.new("GeometryNodeTransform"); tr_s.inputs["Translation"].default_value = (0.0, 0.0, 0.12)
            quad = ng.nodes.new("GeometryNodeCurvePrimitiveQuadrilateral")
            quad.inputs["Width"].default_value = 0.22
            quad.inputs["Height"].default_value = 0.032
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_steel
            ng.links.new(axle.outputs["Mesh"], tr_a.inputs["Geometry"])
            ng.links.new(sp.outputs["Curve"], tr_s.inputs["Geometry"])
            ng.links.new(tr_s.outputs["Geometry"], c2m.inputs["Curve"])
            ng.links.new(quad.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(tr_a.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(c2m.outputs["Mesh"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            scr = bpy.data.objects.new("Screw", bpy.data.meshes.new("ScrewM"))
            scr.rotation_euler = (0.0, 1.5708, 0.0)
            scr.location = (-1.1, 0.0, 0.45)
            bpy.context.scene.collection.objects.link(scr)
            scr.modifiers.new("ScrewGN", "NODES").node_group = ng
            for x in (-1.12, 1.12):
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, 0, 0.28))
                bpy.context.active_object.scale = (0.18, 0.48, 0.56)
                bpy.context.active_object.data.materials.append(m_cast)
            """,
            (
                "Plan: Wrap central MeshCylinder axle (Radius=0.11, Depth=2.2) with helical blade from GeometryNodeCurveSpiral "
                "(Start Radius=0.21, End Radius=0.21, Rotations=6.0, Height=1.95) swept with rectangular "
                "GeometryNodeCurvePrimitiveQuadrilateral (Width=0.22, Height=0.032) -> CurveToMesh (Fill Caps=True), "
                "oriented horizontally between two CastIronBlock pillow bearings."
            ),
        ),
        (
            "c5_sp_08_wire_lantern",
            "Build an ornate hanging copper spiral wire lantern cage composed of interlocking helical ribs surrounding a softly glowing warm glass globe.",
            """\
            m_copper = pbr_mat("AgedCopper", (0.76, 0.38, 0.22), roughness=0.34, metal=0.92)
            m_glow = pbr_mat("AmberGlass", (1.0, 0.72, 0.30), roughness=0.18, metal=0.0)
            m_glow.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (1.0, 0.65, 0.2, 1.0)
            m_glow.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 2.0
            ng = bpy.data.node_groups.new("WireLantern", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 32
            sp.inputs["Rotations"].default_value = 1.8
            sp.inputs["Start Radius"].default_value = 0.28
            sp.inputs["End Radius"].default_value = 0.16
            sp.inputs["Height"].default_value = 0.85
            tr1 = ng.nodes.new("GeometryNodeTransform"); tr1.inputs["Rotation"].default_value = (0.0, 0.0, 2.0944)
            tr2 = ng.nodes.new("GeometryNodeTransform"); tr2.inputs["Rotation"].default_value = (0.0, 0.0, 4.1888)
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
            prof.inputs["Resolution"].default_value = 8
            prof.inputs["Radius"].default_value = 0.015
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_copper
            ng.links.new(sp.outputs["Curve"], tr1.inputs["Geometry"])
            ng.links.new(sp.outputs["Curve"], tr2.inputs["Geometry"])
            ng.links.new(sp.outputs["Curve"], jn.inputs["Geometry"])
            ng.links.new(tr1.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(tr2.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], c2m.inputs["Curve"])
            ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            cage = bpy.data.objects.new("Lantern", bpy.data.meshes.new("LanternM"))
            cage.location = (0, 0, 0.4)
            bpy.context.scene.collection.objects.link(cage)
            cage.modifiers.new("LanternGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.14, location=(0, 0, 0.78))
            bpy.context.active_object.data.materials.append(m_glow)
            """,
            (
                "Plan: Generate tapered GeometryNodeCurveSpiral rib (Start Radius=0.28, End Radius=0.16, Rotations=1.8, Height=0.85), "
                "clone at 120 deg and 240 deg via TransformGeometry -> JoinGeometry, sweep with GeometryNodeCurvePrimitiveCircle "
                "(Radius=0.015) + CurveToMesh (Fill Caps=True), shade AgedCopper around an emissive AmberGlass UV sphere."
            ),
        ),
        (
            "c5_sp_09_bronze_obelisk",
            "Build a sculptural tapered bronze twisted-ribbon obelisk monument composed of three phase-shifted rectangular spirals atop a stepped stone pedestal.",
            """\
            m_bronze = make_mat("PatinaBronze", (0.64, 0.45, 0.24), rough=0.30, metallic=0.88)
            m_granite = make_mat("GrayGranite", (0.36, 0.37, 0.39), rough=0.62, metallic=0.0)
            ng = bpy.data.node_groups.new("BronzeObelisk", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 42
            sp.inputs["Rotations"].default_value = 2.5
            sp.inputs["Start Radius"].default_value = 0.36
            sp.inputs["End Radius"].default_value = 0.06
            sp.inputs["Height"].default_value = 2.2
            quad = ng.nodes.new("GeometryNodeCurvePrimitiveQuadrilateral")
            quad.inputs["Width"].default_value = 0.14
            quad.inputs["Height"].default_value = 0.045
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            t1 = ng.nodes.new("GeometryNodeTransform"); t1.inputs["Rotation"].default_value = (0.0, 0.0, 2.0944)
            t2 = ng.nodes.new("GeometryNodeTransform"); t2.inputs["Rotation"].default_value = (0.0, 0.0, 4.1888)
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_bronze
            ng.links.new(sp.outputs["Curve"], c2m.inputs["Curve"])
            ng.links.new(quad.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], t1.inputs["Geometry"])
            ng.links.new(c2m.outputs["Mesh"], t2.inputs["Geometry"])
            ng.links.new(c2m.outputs["Mesh"], jn.inputs["Geometry"])
            ng.links.new(t1.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(t2.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            ob = bpy.data.objects.new("Obelisk", bpy.data.meshes.new("ObeliskM"))
            ob.location = (0, 0, 0.42)
            bpy.context.scene.collection.objects.link(ob)
            ob.modifiers.new("ObeliskGN", "NODES").node_group = ng
            for z, s in ((0.10, 1.25), (0.30, 0.92)):
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, z))
                bpy.context.active_object.scale = (s, s, 0.22)
                bpy.context.active_object.data.materials.append(m_granite)
            """,
            (
                "Plan: Taper GeometryNodeCurveSpiral (Start Radius=0.36, End Radius=0.06, Rotations=2.5, Height=2.2), sweep "
                "using GeometryNodeCurvePrimitiveQuadrilateral (Width=0.14, Height=0.045) + CurveToMesh (Fill Caps=True), "
                "duplicate 3-way symmetry at +2pi/3 and +4pi/3 via JoinGeometry, shade PatinaBronze above two stepped granite cubes."
            ),
        ),
        (
            "c5_sp_10_grain_chute",
            "Build an industrial galvanized-steel helical grain chute trough swept with an open semicircular arc profile around a vertical support column on a floor flange.",
            """\
            m_galv = pbr_mat("GalvanizedSteel", (0.68, 0.70, 0.72), roughness=0.36, metal=0.84)
            m_dark = pbr_mat("FlangeSteel", (0.22, 0.24, 0.26), roughness=0.50, metal=0.85)
            ng = bpy.data.node_groups.new("GrainChute", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 48
            sp.inputs["Rotations"].default_value = 2.5
            sp.inputs["Start Radius"].default_value = 0.52
            sp.inputs["End Radius"].default_value = 0.52
            sp.inputs["Height"].default_value = 2.0
            arc = ng.nodes.new("GeometryNodeCurveArc")
            arc.inputs["Resolution"].default_value = 12
            arc.inputs["Radius"].default_value = 0.18
            arc.inputs["Start Angle"].default_value = -3.14159
            arc.inputs["Sweep Angle"].default_value = 3.14159
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = False
            col = ng.nodes.new("GeometryNodeMeshCylinder")
            col.inputs["Vertices"].default_value = 24
            col.inputs["Radius"].default_value = 0.15
            col.inputs["Depth"].default_value = 2.2
            tr_c = ng.nodes.new("GeometryNodeTransform"); tr_c.inputs["Translation"].default_value = (0.0, 0.0, 1.05)
            jn = ng.nodes.new("GeometryNodeJoinGeometry")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_galv
            ng.links.new(sp.outputs["Curve"], c2m.inputs["Curve"])
            ng.links.new(arc.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(col.outputs["Mesh"], tr_c.inputs["Geometry"])
            ng.links.new(c2m.outputs["Mesh"], jn.inputs["Geometry"])
            ng.links.new(tr_c.outputs["Geometry"], jn.inputs["Geometry"])
            ng.links.new(jn.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            chute = bpy.data.objects.new("Chute", bpy.data.meshes.new("ChuteM"))
            chute.location = (0, 0, 0.08)
            bpy.context.scene.collection.objects.link(chute)
            chute.modifiers.new("ChuteGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.45, depth=0.08, location=(0, 0, 0.04))
            bpy.context.active_object.data.materials.append(m_dark)
            """,
            (
                "Plan: Sweep GeometryNodeCurveSpiral (Start Radius=0.52, End Radius=0.52, Rotations=2.5, Height=2.0, "
                "Resolution=48) with semicircular GeometryNodeCurveArc (Radius=0.18, Start Angle=-pi, Sweep Angle=pi) "
                "via CurveToMesh (Fill Caps=False), join central support MeshCylinder (Radius=0.15), shade GalvanizedSteel."
            ),
        ),
    ]

    items: list[dict[str, Any]] = []
    for i, (rid, prompt_core, body, reasoning) in enumerate(specs):
        code = bcg._wrap_script(i, bcg._head(i), body, bcg._tail(cam_loc=(4.0, -4.6, 3.0)), fn_name=None, apply_mod_obj=None)
        suffix = bcg.EVAL_SUFFIX if i % 2 == 0 else bcg.ALT_SUFFIXES[(i // 2) % len(bcg.ALT_SUFFIXES)]
        prompt_full = prompt_core + suffix
        _check_all_text_forbidden(rid, prompt_full, reasoning, code)
        items.append({
            "id": rid,
            "task_id": rid,
            "domain": "geonodes",
            "system": bcg.GN_SYSTEM_PROMPT,
            "prompt": prompt_full,
            "reasoning": reasoning,
            "response": f"```python\n{code.strip()}\n```",
            "source": "c5_spiral_geonodes",
        })
    return items


def _summarize_blenderworld_reasoning(code: str) -> str:
    """Build an accurate, AST-grounded concise planning trace for a verified Blender 4.2 script."""
    clean = bv.extract_python_code(code)
    fn_names: list[str] = []
    ops_calls: list[str] = []
    mat_names: list[str] = []
    uses_bmesh = "bmesh" in clean
    uses_gn = "GeometryNodeTree" in clean

    try:
        tree = ast.parse(clean)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                if node.name not in fn_names:
                    fn_names.append(node.name)
            elif isinstance(node, ast.Call):
                # Capture bpy.ops.mesh.primitive_* and bpy.data.materials.new("...")
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
                    if node.func.attr.startswith("primitive_") and len(ops_calls) < 4:
                        short = node.func.attr.replace("primitive_", "").replace("_add", "")
                        if short not in ops_calls:
                            ops_calls.append(short)
                    elif (
                        node.func.attr == "new"
                        and node.func.value.attr == "materials"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and len(mat_names) < 3
                    ):
                        if node.args[0].value not in mat_names:
                            mat_names.append(node.args[0].value)
    except SyntaxError:
        pass

    parts = ["Plan: Clear startup scene objects"]
    if mat_names:
        parts.append(f"configure Blender 4.2 Principled BSDF materials ({', '.join(mat_names[:3])})")
    else:
        parts.append("define Blender 4.2 Principled BSDF materials")
    geo_bits = []
    if fn_names:
        geo_bits.append(f"helpers ({', '.join(fn_names[:3])})")
    if ops_calls:
        geo_bits.append(f"primitives ({', '.join(ops_calls[:4])})")
    if uses_bmesh:
        geo_bits.append("bmesh geometry")
    if uses_gn:
        geo_bits.append("GeometryNodeTree modifiers")
    if geo_bits:
        parts.append("build 3D parts via " + " + ".join(geo_bits))
    else:
        parts.append("construct multi-part object hierarchy")
    parts.append("setup lighting, active camera, Cycles (64 samples), and save /workspace/exports/final.blend.")
    return "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "work" / "curriculum_c5_unified.jsonl",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tok = tlora.load_qwen35_tokenizer()
    eval_gn_ids = {str(t["id"]) for t in bgn.HELD_OUT_EVAL_TASKS}
    eval_bw_ids = set(bv.load_blenderworld_text_tasks()["split"]["eval"])

    # 1. GeoNodes Core (60 rows from curriculum_c4_compact_geonodes.jsonl)
    c4_path = SCRIPT_DIR / "work" / "curriculum_c4_compact_geonodes.jsonl"
    c4_all = [json.loads(l) for l in c4_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    gn_core_rows: list[dict[str, Any]] = []
    for r in c4_all:
        if r.get("source") != "c4_compact_geonodes":
            continue
        row = copy.deepcopy(r)
        row["domain"] = "geonodes"
        row["system"] = str(row.get("system") or bcg.GN_SYSTEM_PROMPT)
        assert str(row.get("id")) not in eval_gn_ids
        gn_core_rows.append(row)
    assert len(gn_core_rows) == 60, f"Expected 60 c4_compact_geonodes rows, got {len(gn_core_rows)}"

    # 2. Harvested Cycle 4B RL GeoNodes Winners (4 rows)
    gn_win_path = SCRIPT_DIR / "work" / "rl_gn_harvested_winners.jsonl"
    gn_win_raw = [json.loads(l) for l in gn_win_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    gn_win_rows: list[dict[str, Any]] = []
    for r in gn_win_raw:
        tid = str(r.get("task_id") or r.get("id"))
        assert tid not in eval_gn_ids and str(r.get("id")) not in eval_gn_ids, f"Eval GN leak: {tid}"
        row = {
            "id": str(r["id"]),
            "task_id": tid,
            "domain": "geonodes",
            "source": "c4_rl_gn_winner",
            "system": bcg.GN_SYSTEM_PROMPT,
            "prompt": str(r["prompt"]),
            "reasoning": str(r.get("reasoning") or ""),
            "response": str(r["response"]),
            "reward": float(r.get("reward", 1.0)),
        }
        ids, _ = tlora.format_conversation(row, tok, max_seq_len=99999, truncate=False)
        row["tokens"] = len(ids)
        gn_win_rows.append(row)
    assert len(gn_win_rows) == 4, f"Expected 4 rl_gn_harvested_winners rows, got {len(gn_win_rows)}"

    # 3. 10 New Compact Blender 5.4 Curve/Spiral/Sweep GeoNodes Recipes (c5_sp_01..10)
    c5_sp_rows = _build_c5_spiral_recipes()
    print(f"[c5_data] verifying {len(c5_sp_rows)} new spiral GeoNodes scripts in headless Blender 5.4...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        c5_results = list(pool.map(bv.verify_script, [r["response"] for r in c5_sp_rows]))

    sp_failures: list[str] = []
    for row, vr in zip(c5_sp_rows, c5_results):
        gp = bv.is_geonodes_pass(vr)
        ss = vr.scene_summary or {}
        verts = int(ss.get("total_verts", 0))
        ids, _ = tlora.format_conversation(row, tok, max_seq_len=99999, truncate=False)
        n_tok = len(ids)
        row["tokens"] = n_tok
        row["reward"] = float(vr.reward)
        row["verts"] = verts
        row["gn_trees"] = ss.get("geonode_trees_count", 0)
        row["gn_nodes"] = ss.get("geonode_nodes_total", 0)
        row["gn_links"] = ss.get("geonode_links_total", 0)
        ok = vr.syntax_ok and vr.exec_ok and gp and verts >= 100 and vr.reward >= 0.85 and n_tok <= 1550
        tag = "OK  " if ok else "FAIL"
        print(
            f"  [{tag}] {row['id']:26s} toks={n_tok:4d} verts={verts:5d} "
            f"gn=({row['gn_trees']}/{row['gn_nodes']}/{row['gn_links']}) R={vr.reward:.2f}",
            flush=True,
        )
        if not ok:
            sp_failures.append(
                f"{row['id']}: syntax={vr.syntax_ok} exec={vr.exec_ok} gp={gp} verts={verts} R={vr.reward} toks={n_tok} err={vr.error[:240]}"
            )
    if sp_failures:
        raise RuntimeError("Failed c5_sp recipes:\n" + "\n".join(sp_failures))

    # 4. BlenderWorld Corpus (54 rows: 5 RL winners + 49 Strat A diverse verified)
    bw_win_path = SCRIPT_DIR / "work" / "rl_harvested_winners.jsonl"
    bw_win_raw = [json.loads(l) for l in bw_win_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    bw_win_rows: list[dict[str, Any]] = []
    seen_bw_ids: set[str] = set()
    for r in bw_win_raw:
        tid = str(r.get("task_id") or r.get("id"))
        rid = str(r["id"])
        assert tid not in eval_bw_ids and rid not in eval_bw_ids, f"Eval BW leak: {tid}"
        seen_bw_ids.add(tid)
        seen_bw_ids.add(rid)
        row = {
            "id": rid,
            "task_id": tid,
            "domain": "blenderworld",
            "source": "c2_rl_bw_winner",
            "system": BLENDERWORLD_SYS_PROMPT,
            "prompt": str(r["prompt"]),
            "reasoning": str(r.get("reasoning") or ""),
            "response": str(r["response"]),
            "reward": float(r.get("reward", 0.5)),
        }
        ids, _ = tlora.format_conversation(row, tok, max_seq_len=99999, truncate=False)
        row["tokens"] = len(ids)
        bw_win_rows.append(row)
    assert len(bw_win_rows) == 5, f"Expected 5 rl_harvested_winners rows, got {len(bw_win_rows)}"

    strat_path = SCRIPT_DIR / "work" / "strat_a_verified.jsonl"
    strat_raw = [json.loads(l) for l in strat_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_src: dict[str, list[tuple[tuple[int, float, int], dict[str, Any]]]] = {}
    for r in strat_raw:
        tid = str(r.get("task_id") or r.get("id"))
        rid = str(r["id"])
        if tid in eval_bw_ids or rid in eval_bw_ids or tid in seen_bw_ids or rid in seen_bw_ids:
            continue
        row = {
            "id": rid,
            "task_id": tid,
            "domain": "blenderworld",
            "source": "strat_a_bw",
            "orig_source": str(r.get("source") or "other"),
            "system": BLENDERWORLD_SYS_PROMPT,
            "prompt": str(r["prompt"]),
            "reasoning": str(r.get("reasoning") or ""),
            "response": str(r["response"]),
            "reward": float(r.get("reward", 0.8)),
        }
        ids, _ = tlora.format_conversation(row, tok, max_seq_len=99999, truncate=False)
        n_tok = len(ids)
        if n_tok <= 2150:
            row["tokens"] = n_tok
            is_cam = 1 if ("camera" in rid.lower() or "level1" in rid.lower()) else 0
            sort_key = (is_cam, -row["reward"], n_tok)
            by_src.setdefault(row["orig_source"], []).append((sort_key, row))

    for lst in by_src.values():
        lst.sort(key=lambda x: x[0])

    # Round-robin pick 49 rows across source buckets, preferring rich scene/object sources before simple camera tests
    src_order = sorted(by_src.keys(), key=lambda s: (1 if "BlenderBench" in s else 0, s))
    strat_selected: list[dict[str, Any]] = []
    idx = 0
    while len(strat_selected) < 49:
        progressed = False
        for src_name in src_order:
            if len(strat_selected) >= 49:
                break
            if idx < len(by_src[src_name]):
                _, cand = by_src[src_name][idx]
                if cand["id"] not in seen_bw_ids and cand["task_id"] not in seen_bw_ids:
                    seen_bw_ids.add(cand["id"])
                    seen_bw_ids.add(cand["task_id"])
                    strat_selected.append(cand)
                    progressed = True
        if not progressed:
            break
        idx += 1
    assert len(strat_selected) == 49, f"Expected 49 strat_a_bw rows <=2150 tok, got {len(strat_selected)}"

    bw_rows = bw_win_rows + strat_selected

    # Add concise AST-grounded reasoning traces (~60-140 tok) to 18 BlenderWorld rows with base tokens <= 1980
    bw_reason_candidates = sorted(
        [r for r in bw_rows if not r.get("reasoning") and r["tokens"] <= 1980],
        key=lambda r: (0 if r["source"] == "c2_rl_bw_winner" else 1, -r["reward"], r["tokens"]),
    )
    assert len(bw_reason_candidates) >= 18, f"Need >=18 BW rows <=1980 tok for reasoning, got {len(bw_reason_candidates)}"
    bw_reasoned = 0
    for r in bw_reason_candidates:
        if bw_reasoned >= 18:
            break
        plan_txt = _summarize_blenderworld_reasoning(r["response"])
        test_r = copy.deepcopy(r)
        test_r["reasoning"] = plan_txt
        new_tok = len(tlora.format_conversation(test_r, tok, max_seq_len=99999, truncate=False)[0])
        if new_tok <= 2250:
            r["reasoning"] = plan_txt
            r["tokens"] = new_tok
            bw_reasoned += 1
    assert bw_reasoned == 18, f"Expected 18 reasoned BlenderWorld rows <=2250 tok, got {bw_reasoned}"

    # 5. Add explicit budget-cutoff transition marker to 4 GeoNodes + 4 BlenderWorld reasoning rows
    cutoff_msg = bbw.DEFAULT_REASONING_BUDGET_MESSAGE.strip()
    gn_rows = gn_core_rows + gn_win_rows + c5_sp_rows

    gn_cut_count = 0
    # Spread across c5_sp + c4_compact_geonodes rows that have reasoning and remain <= 2250
    for r in [x for x in c5_sp_rows[::3]] + [x for x in gn_core_rows if (x.get("reasoning") or "").strip()]:
        if gn_cut_count >= 4:
            break
        if cutoff_msg in (r.get("reasoning") or ""):
            continue
        new_reason = r["reasoning"].rstrip() + "\n" + cutoff_msg
        test_r = copy.deepcopy(r)
        test_r["reasoning"] = new_reason
        new_tok = len(tlora.format_conversation(test_r, tok, max_seq_len=99999, truncate=False)[0])
        if new_tok <= 2250:
            r["reasoning"] = new_reason
            r["tokens"] = new_tok
            r["cutoff_example"] = True
            gn_cut_count += 1
    assert gn_cut_count == 4, f"Expected 4 GeoNodes cutoff rows, got {gn_cut_count}"

    bw_cut_count = 0
    for r in [x for x in bw_rows if (x.get("reasoning") or "").strip()]:
        if bw_cut_count >= 4:
            break
        new_reason = r["reasoning"].rstrip() + "\n" + cutoff_msg
        test_r = copy.deepcopy(r)
        test_r["reasoning"] = new_reason
        new_tok = len(tlora.format_conversation(test_r, tok, max_seq_len=99999, truncate=False)[0])
        if new_tok <= 2250:
            r["reasoning"] = new_reason
            r["tokens"] = new_tok
            r["cutoff_example"] = True
            bw_cut_count += 1
    assert bw_cut_count == 4, f"Expected 4 BlenderWorld cutoff rows, got {bw_cut_count}"

    all_rows = gn_rows + bw_rows

    # 6. Final Dataset Invariants (Option 2)
    assert len(all_rows) == 128, f"Expected 128 total rows, got {len(all_rows)}"
    for r in all_rows:
        ids, _ = tlora.format_conversation(r, tok, max_seq_len=99999, truncate=False)
        r["tokens"] = len(ids)
        tid = str(r.get("task_id") or r.get("id"))
        assert tid not in eval_gn_ids and str(r["id"]) not in eval_gn_ids, f"Eval GN leak: {tid}"
        assert tid not in eval_bw_ids and str(r["id"]) not in eval_bw_ids, f"Eval BW leak: {tid}"
        if r["source"] == "c2_rl_bw_winner":
            assert r["tokens"] <= 2850, f"Winner {r['id']} exceeded 2850 tok: {r['tokens']}"
        else:
            assert r["tokens"] <= 2250, f"Row {r['id']} ({r['source']}) exceeded 2250 tok: {r['tokens']}"

    n_reason = sum(1 for r in all_rows if (r.get("reasoning") or "").strip())
    n_cutoff = sum(1 for r in all_rows if cutoff_msg in (r.get("reasoning") or ""))
    assert 45 <= n_reason <= 52, f"Expected 45..52 reasoned rows, got {n_reason}"
    assert n_cutoff == 8, f"Expected 8 cutoff examples, got {n_cutoff}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n",
        encoding="utf-8",
    )

    all_toks = [r["tokens"] for r in all_rows]
    gn_toks = [r["tokens"] for r in gn_rows]
    bw_toks = [r["tokens"] for r in bw_rows]
    print(f"\n=== CYCLE 5 UNIFIED DATASET READY: {args.out} ===", flush=True)
    print(f"  Total rows       : {len(all_rows)} (GeoNodes={len(gn_rows)} [{100*len(gn_rows)/len(all_rows):.1f}%], BlenderWorld={len(bw_rows)} [{100*len(bw_rows)/len(all_rows):.1f}%])")
    print(f"  With reasoning   : {n_reason}/{len(all_rows)} ({100*n_reason/len(all_rows):.1f}%), budget-cutoff examples={n_cutoff} (gn={gn_cut_count}, bw={bw_cut_count})")
    print(f"  All tokens       : min={min(all_toks)}, med={int(statistics.median(all_toks))}, max={max(all_toks)} (<=3072 max_seq_len)")
    print(f"  GeoNodes tokens  : min={min(gn_toks)}, med={int(statistics.median(gn_toks))}, max={max(gn_toks)}")
    print(f"  BW tokens        : min={min(bw_toks)}, med={int(statistics.median(bw_toks))}, max={max(bw_toks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
