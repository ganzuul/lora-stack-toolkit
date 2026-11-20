#!/usr/bin/env python3
"""Build, anti-leakage check, and Blender-5.4 verify 60 compact Geometry Nodes + 12 replay SFT rows."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import textwrap
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import blender_verifier as bv  # noqa: E402
import train_lora as tlora  # noqa: E402

GN_SYSTEM_PROMPT = (
    "You are an expert Blender 5.4 Python programmer. Generate clean, self-contained, executable `bpy` code."
)

EVAL_SUFFIX = (
    "\nWrite a clean, standalone Blender 5.4 Python (`bpy`) script using Geometry Nodes "
    "(`GeometryNodeTree` with `ng.interface.new_socket`, `ng.nodes.new`, `ng.links.new`, and `NodeGroupOutput`) "
    "to build this procedural object from scratch, attach a `NODES` modifier or apply cleanly while keeping "
    "the node group in `bpy.data.node_groups`, assign `ShaderNodeBsdfPrincipled` materials, add a SUN light, "
    "active CAMERA, Cycles (64 samples), and save `/workspace/exports/final.blend`."
)

ALT_SUFFIXES = [
    (
        "\nGenerate a concise, standalone Blender 5.4 `bpy` script building the asset via `GeometryNodeTree` "
        "(`ng.interface.new_socket`, `ng.nodes.new`, `ng.links.new`, `NodeGroupOutput`). Attach a `NODES` modifier "
        "(keeping the node group in `bpy.data.node_groups`), set `ShaderNodeBsdfPrincipled` materials, add a SUN "
        "light and active CAMERA, configure Cycles (`64` samples), and save `/workspace/exports/final.blend`."
    ),
    (
        "\nCreate a clean Blender 5.4 Python (`bpy`) script using a `GeometryNodeTree` (`ng.interface.new_socket`, "
        "`ng.nodes.new`, `ng.links.new`, `NodeGroupOutput`), linked to an object via a `NODES` modifier while "
        "keeping the group in `bpy.data.node_groups`. Use `ShaderNodeBsdfPrincipled` materials, a SUN light, an "
        "active CAMERA, Cycles 64 samples, and write `/workspace/exports/final.blend`."
    ),
    (
        "\nWrite a compact self-contained Blender 5.4 script with Geometry Nodes (`bpy.data.node_groups.new(..., "
        "'GeometryNodeTree')`, `interface.new_socket`, `NodeGroupOutput`), a `NODES` modifier kept in `bpy.data.node_groups`, "
        "`ShaderNodeBsdfPrincipled` shading, a SUN light, active CAMERA, Cycles (`64` samples), and export `/workspace/exports/final.blend`."
    ),
]

FORBIDDEN_TERMS = [
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
    "moss",
    "twisted torus",
    "scalloped bowl",
    "sea-urchin",
    "urchin",
    "bamboo",
    "corrugated",
    "coiled rope",
    "pine bough",
    "pine needle",
    "torus",
    "bowl",
    "rope",
    "pine",
]


def _head(style: int) -> str:
    if style % 2 == 0:
        return textwrap.dedent("""\
            import bpy

            for o in list(bpy.data.objects):
                bpy.data.objects.remove(o, do_unlink=True)
            for m in list(bpy.data.materials):
                bpy.data.materials.remove(m)
            for g in list(bpy.data.node_groups):
                bpy.data.node_groups.remove(g)

            def make_mat(name, color, rough=0.45, metallic=0.0):
                mat = bpy.data.materials.new(name)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs["Base Color"].default_value = (*color, 1.0)
                bsdf.inputs["Roughness"].default_value = rough
                bsdf.inputs["Metallic"].default_value = metallic
                return mat
        """)
    return textwrap.dedent("""\
        import bpy

        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mat in list(bpy.data.materials):
            bpy.data.materials.remove(mat)
        for grp in list(bpy.data.node_groups):
            bpy.data.node_groups.remove(grp)

        def pbr_mat(mat_name, rgb, roughness=0.4, metal=0.0):
            m = bpy.data.materials.new(mat_name)
            m.use_nodes = True
            node = m.node_tree.nodes["Principled BSDF"]
            node.inputs["Base Color"].default_value = (*rgb, 1.0)
            node.inputs["Roughness"].default_value = roughness
            node.inputs["Metallic"].default_value = metal
            return m
    """)


def _tail(cam_loc: tuple[float, float, float] = (4.2, -4.8, 3.0), energy: float = 3.2, lens: float = 42.0) -> str:
    return textwrap.dedent(f"""\
        bpy.ops.object.light_add(type="SUN", location=(4, -4, 6.5))
        bpy.context.active_object.data.energy = {energy}
        bpy.ops.object.camera_add(location={cam_loc}, rotation=(1.12, 0.0, 0.72))
        sc = bpy.context.scene
        sc.camera = bpy.context.active_object
        sc.render.engine = "CYCLES"
        sc.cycles.samples = 64
        bpy.ops.wm.save_as_mainfile(filepath="/workspace/exports/final.blend")
    """)


def _wrap_script(
    idx: int,
    head: str,
    body: str,
    tail: str,
    fn_name: str | None = None,
    apply_mod_obj: str | None = None,
) -> str:
    clean_head = textwrap.dedent(head).strip("\n")
    clean_body = textwrap.dedent(body).strip("\n")
    clean_tail = textwrap.dedent(tail).strip("\n")
    if fn_name:
        indented = textwrap.indent(clean_body, "    ")
        code = f"{clean_head}\n\ndef {fn_name}():\n{indented}\n\n{fn_name}()\n"
    else:
        code = f"{clean_head}\n\n{clean_body}\n"
    if apply_mod_obj:
        code += textwrap.dedent(f"""\
            bpy.context.view_layer.objects.active = {apply_mod_obj}
            bpy.ops.object.modifier_apply(modifier={apply_mod_obj}.modifiers[0].name)
        """)
    return code + "\n" + clean_tail + "\n"


# ==============================================================================
# Family 1 (10 recipes): Radial / Azimuthal & Axial SetPosition Sculpting
# ==============================================================================
def _build_family1() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    specs = [
        (
            "f1_01_pumpkin_gourd",
            "Build a ribbed autumn pumpkin gourd with subtle vertical lobe creases, a flat wooden pedestal disk, and a short curved green stem procedurally.",
            """\
            m_rind = make_mat("PumpkinRind", (0.82, 0.34, 0.05), rough=0.48)
            m_stem = make_mat("PumpkinStem", (0.20, 0.32, 0.10), rough=0.68)
            ng = bpy.data.node_groups.new("PumpkinGourd", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeMeshUVSphere")
            sp.inputs["Segments"].default_value = 64
            sp.inputs["Rings"].default_value = 32
            sp.inputs["Radius"].default_value = 0.65
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            ang = ng.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
            mul_f = ng.nodes.new("ShaderNodeMath"); mul_f.operation = "MULTIPLY"; mul_f.inputs[1].default_value = 12.0
            cos_n = ng.nodes.new("ShaderNodeMath"); cos_n.operation = "COSINE"
            abs_n = ng.nodes.new("ShaderNodeMath"); abs_n.operation = "ABSOLUTE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.085
            vec = ng.nodes.new("ShaderNodeCombineXYZ")
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Y"], ang.inputs[0]); ng.links.new(sep.outputs["X"], ang.inputs[1])
            ng.links.new(ang.outputs["Value"], mul_f.inputs[0])
            ng.links.new(mul_f.outputs["Value"], cos_n.inputs[0])
            ng.links.new(cos_n.outputs["Value"], abs_n.inputs[0])
            ng.links.new(abs_n.outputs["Value"], amp.inputs[0])
            scl = ng.nodes.new("ShaderNodeVectorMath"); scl.operation = "SCALE"
            ng.links.new(sep.outputs["X"], vec.inputs["X"]); ng.links.new(sep.outputs["Y"], vec.inputs["Y"])
            mul_z = ng.nodes.new("ShaderNodeMath"); mul_z.operation = "MULTIPLY"; mul_z.inputs[1].default_value = -0.32
            ng.links.new(sep.outputs["Z"], mul_z.inputs[0]); ng.links.new(mul_z.outputs["Value"], vec.inputs["Z"])
            ng.links.new(vec.outputs["Vector"], scl.inputs["Vector"]); ng.links.new(amp.outputs["Value"], scl.inputs["Scale"])
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_rind
            ng.links.new(sp.outputs["Mesh"], set_p.inputs["Geometry"])
            ng.links.new(scl.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            obj = bpy.data.objects.new("Pumpkin", bpy.data.meshes.new("PumpkinM"))
            obj.location = (0, 0, 0.45)
            bpy.context.scene.collection.objects.link(obj)
            obj.modifiers.new("PumpkinGeo", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.07, depth=0.35, location=(0, 0, 0.92))
            bpy.context.active_object.rotation_euler = (0.18, 0.12, 0.0)
            bpy.context.active_object.data.materials.append(m_stem)
            """,
            "Plan: Use GeometryNodeMeshUVSphere (64x32), compute azimuth ARCTAN2(Y,X)*12 -> ABS(COS) radial lobes scaled via VectorMath SCALE, compress poles in Z via SetPosition, shade PumpkinRind + separate stem mesh.",
        ),
        (
            "f1_02_doric_column",
            "Build a fluted white marble Doric architectural column with 20 concave vertical channels tapering gently toward a square stone capital and plinth.",
            """\
            m_marble = pbr_mat("DoricMarble", (0.86, 0.84, 0.79), roughness=0.32)
            m_plinth = pbr_mat("DarkPlinth", (0.24, 0.24, 0.26), roughness=0.58)
            ng = bpy.data.node_groups.new("DoricColumn", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            out = ng.nodes.new("NodeGroupOutput")
            cyl = ng.nodes.new("GeometryNodeMeshCylinder")
            cyl.inputs["Vertices"].default_value = 80
            cyl.inputs["Side Segments"].default_value = 24
            cyl.inputs["Radius"].default_value = 0.42
            cyl.inputs["Depth"].default_value = 2.6
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            ang = ng.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
            freq = ng.nodes.new("ShaderNodeMath"); freq.operation = "MULTIPLY"; freq.inputs[1].default_value = 20.0
            cos_n = ng.nodes.new("ShaderNodeMath"); cos_n.operation = "COSINE"
            flute = ng.nodes.new("ShaderNodeMath"); flute.operation = "MULTIPLY"; flute.inputs[1].default_value = -0.025
            rad = ng.nodes.new("ShaderNodeVectorMath"); rad.operation = "SCALE"
            xy = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_marble
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Y"], ang.inputs[0]); ng.links.new(sep.outputs["X"], ang.inputs[1])
            ng.links.new(ang.outputs["Value"], freq.inputs[0]); ng.links.new(freq.outputs["Value"], cos_n.inputs[0])
            ng.links.new(cos_n.outputs["Value"], flute.inputs[0])
            ng.links.new(sep.outputs["X"], xy.inputs["X"]); ng.links.new(sep.outputs["Y"], xy.inputs["Y"])
            ng.links.new(xy.outputs["Vector"], rad.inputs["Vector"]); ng.links.new(flute.outputs["Value"], rad.inputs["Scale"])
            ng.links.new(cyl.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(rad.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"])
            ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], out.inputs["Geometry"])
            col = bpy.data.objects.new("Column", bpy.data.meshes.new("ColM"))
            col.location = (0, 0, 1.4)
            bpy.context.scene.collection.objects.link(col)
            col.modifiers.new("ColumnGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.08))
            bpy.context.active_object.scale = (1.1, 1.1, 0.16)
            bpy.context.active_object.data.materials.append(m_plinth)
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 2.76))
            bpy.context.active_object.scale = (1.05, 1.05, 0.14)
            bpy.context.active_object.data.materials.append(m_marble)
            """,
            None,
        ),
        (
            "f1_03_chalice_goblet",
            "Build a turned sterling silver ceremonial chalice cup sculpted from a segmented cylinder with radial flutes and a flared rim sitting on a dark slate plinth.",
            """\
            m_silver = make_mat("SilverChalice", (0.85, 0.86, 0.89), rough=0.18, metallic=0.96)
            m_slate = make_mat("SlateBase", (0.14, 0.15, 0.17), rough=0.72)
            ng = bpy.data.node_groups.new("SilverChalice", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            cyl = ng.nodes.new("GeometryNodeMeshCylinder")
            cyl.inputs["Vertices"].default_value = 64
            cyl.inputs["Side Segments"].default_value = 28
            cyl.inputs["Radius"].default_value = 0.38
            cyl.inputs["Depth"].default_value = 1.1
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            ang = ng.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
            m_a = ng.nodes.new("ShaderNodeMath"); m_a.operation = "MULTIPLY"; m_a.inputs[1].default_value = 16.0
            s_a = ng.nodes.new("ShaderNodeMath"); s_a.operation = "SINE"
            fl = ng.nodes.new("ShaderNodeMath"); fl.operation = "MULTIPLY"; fl.inputs[1].default_value = 0.028
            z_sq = ng.nodes.new("ShaderNodeMath"); z_sq.operation = "POWER"; z_sq.inputs[1].default_value = 2.0
            flare = ng.nodes.new("ShaderNodeMath"); flare.operation = "MULTIPLY"; flare.inputs[1].default_value = 0.9
            tot = ng.nodes.new("ShaderNodeMath"); tot.operation = "ADD"
            xy = ng.nodes.new("ShaderNodeCombineXYZ")
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_silver
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Y"], ang.inputs[0]); ng.links.new(sep.outputs["X"], ang.inputs[1])
            ng.links.new(ang.outputs["Value"], m_a.inputs[0]); ng.links.new(m_a.outputs["Value"], s_a.inputs[0])
            ng.links.new(s_a.outputs["Value"], fl.inputs[0])
            ng.links.new(sep.outputs["Z"], z_sq.inputs[0]); ng.links.new(z_sq.outputs["Value"], flare.inputs[0])
            ng.links.new(fl.outputs["Value"], tot.inputs[0]); ng.links.new(flare.outputs["Value"], tot.inputs[1])
            ng.links.new(sep.outputs["X"], xy.inputs["X"]); ng.links.new(sep.outputs["Y"], xy.inputs["Y"])
            ng.links.new(xy.outputs["Vector"], vsc.inputs["Vector"]); ng.links.new(tot.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(cyl.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            obj = bpy.data.objects.new("Chalice", bpy.data.meshes.new("ChaliceM"))
            obj.location = (0, 0, 0.85)
            bpy.context.scene.collection.objects.link(obj)
            obj.modifiers.new("ChaliceGeo", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.45, depth=0.14, location=(0, 0, 0.07))
            bpy.context.active_object.data.materials.append(m_slate)
            """,
            None,
        ),
        (
            "f1_04_paper_lantern",
            "Build a pleated silk oval hanging paper lantern with crisp vertical ribs and brushed brass end collars.",
            """\
            m_paper = pbr_mat("WarmPaper", (0.94, 0.87, 0.72), roughness=0.62)
            m_brass = pbr_mat("LanternBrass", (0.76, 0.56, 0.20), roughness=0.25, metal=0.9)
            ng = bpy.data.node_groups.new("LanternGlobe", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeMeshUVSphere")
            sp.inputs["Segments"].default_value = 72
            sp.inputs["Rings"].default_value = 32
            sp.inputs["Radius"].default_value = 0.58
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            ang = ng.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
            mf = ng.nodes.new("ShaderNodeMath"); mf.operation = "MULTIPLY"; mf.inputs[1].default_value = 18.0
            sn = ng.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.04
            mz = ng.nodes.new("ShaderNodeMath"); mz.operation = "MULTIPLY"; mz.inputs[1].default_value = 0.35
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            addv = ng.nodes.new("ShaderNodeVectorMath"); addv.operation = "ADD"
            zoff = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_paper
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Y"], ang.inputs[0]); ng.links.new(sep.outputs["X"], ang.inputs[1])
            ng.links.new(ang.outputs["Value"], mf.inputs[0]); ng.links.new(mf.outputs["Value"], sn.inputs[0])
            ng.links.new(sn.outputs["Value"], amp.inputs[0])
            ng.links.new(sep.outputs["X"], comb.inputs["X"]); ng.links.new(sep.outputs["Y"], comb.inputs["Y"])
            ng.links.new(comb.outputs["Vector"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(sep.outputs["Z"], mz.inputs[0]); ng.links.new(mz.outputs["Value"], zoff.inputs["Z"])
            ng.links.new(vsc.outputs["Vector"], addv.inputs[0]); ng.links.new(zoff.outputs["Vector"], addv.inputs[1])
            ng.links.new(sp.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(addv.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            lobj = bpy.data.objects.new("Lantern", bpy.data.meshes.new("LantM"))
            lobj.location = (0, 0, 1.0)
            bpy.context.scene.collection.objects.link(lobj)
            lobj.modifiers.new("LanternNodes", "NODES").node_group = ng
            for zz in (0.22, 1.78):
                bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.22, depth=0.07, location=(0, 0, zz))
                bpy.context.active_object.data.materials.append(m_brass)
            """,
            "Plan: Build GeometryNodeMeshUVSphere stretched along Z with CombineXYZ, modulate radial X/Y via SINE(ARCTAN2(Y,X)*18), and place two brass collars.",
        ),
        (
            "f1_05_acorn_nut",
            "Build a sculpted oak acorn nut using Geometry Nodes with a pointed tapered kernel shell and a textured cap dome.",
            """\
            m_shell = make_mat("AcornShell", (0.56, 0.34, 0.14), rough=0.36)
            m_cap = make_mat("AcornCap", (0.24, 0.15, 0.08), rough=0.78)
            ng = bpy.data.node_groups.new("AcornKernel", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeMeshUVSphere")
            sp.inputs["Segments"].default_value = 48
            sp.inputs["Rings"].default_value = 28
            sp.inputs["Radius"].default_value = 0.45
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            zt = ng.nodes.new("ShaderNodeMath"); zt.operation = "MULTIPLY"; zt.inputs[1].default_value = 0.45
            taper = ng.nodes.new("ShaderNodeMath"); taper.operation = "MULTIPLY"; taper.inputs[1].default_value = -0.35
            xy = ng.nodes.new("ShaderNodeCombineXYZ")
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            zcomb = ng.nodes.new("ShaderNodeCombineXYZ")
            addv = ng.nodes.new("ShaderNodeVectorMath"); addv.operation = "ADD"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_shell
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Z"], taper.inputs[0]); ng.links.new(sep.outputs["Z"], zt.inputs[0])
            ng.links.new(sep.outputs["X"], xy.inputs["X"]); ng.links.new(sep.outputs["Y"], xy.inputs["Y"])
            ng.links.new(xy.outputs["Vector"], vsc.inputs["Vector"]); ng.links.new(taper.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(zt.outputs["Value"], zcomb.inputs["Z"])
            ng.links.new(vsc.outputs["Vector"], addv.inputs[0]); ng.links.new(zcomb.outputs["Vector"], addv.inputs[1])
            ng.links.new(sp.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(addv.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"]); ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            aobj = bpy.data.objects.new("Acorn", bpy.data.meshes.new("AcornM"))
            aobj.location = (0, 0, 0.55)
            bpy.context.scene.collection.objects.link(aobj)
            aobj.modifiers.new("AcornGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_uv_sphere_add(segments=36, ring_count=16, radius=0.50, location=(0, 0, 0.92))
            cap = bpy.context.active_object
            cap.scale = (1.0, 1.0, 0.55)
            cap.data.materials.append(m_cap)
            """,
            None,
        ),
        (
            "f1_06_pagoda_pillar",
            "Build a helically twisted carved timber pagoda pillar using Geometry Nodes by twisting azimuthal flute ridges smoothly along Z.",
            """\
            m_lacquer = pbr_mat("VermilionWood", (0.58, 0.11, 0.08), roughness=0.34)
            m_granite = pbr_mat("GranitePad", (0.36, 0.36, 0.38), roughness=0.72)
            ng = bpy.data.node_groups.new("PagodaPillar", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            cyl = ng.nodes.new("GeometryNodeMeshCylinder")
            cyl.inputs["Vertices"].default_value = 64
            cyl.inputs["Side Segments"].default_value = 48
            cyl.inputs["Radius"].default_value = 0.38
            cyl.inputs["Depth"].default_value = 2.4
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            ang = ng.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
            ma = ng.nodes.new("ShaderNodeMath"); ma.operation = "MULTIPLY"; ma.inputs[1].default_value = 8.0
            mz = ng.nodes.new("ShaderNodeMath"); mz.operation = "MULTIPLY"; mz.inputs[1].default_value = 3.2
            phase = ng.nodes.new("ShaderNodeMath"); phase.operation = "ADD"
            sn = ng.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.055
            xy = ng.nodes.new("ShaderNodeCombineXYZ")
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_lacquer
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Y"], ang.inputs[0]); ng.links.new(sep.outputs["X"], ang.inputs[1])
            ng.links.new(ang.outputs["Value"], ma.inputs[0]); ng.links.new(sep.outputs["Z"], mz.inputs[0])
            ng.links.new(ma.outputs["Value"], phase.inputs[0]); ng.links.new(mz.outputs["Value"], phase.inputs[1])
            ng.links.new(phase.outputs["Value"], sn.inputs[0]); ng.links.new(sn.outputs["Value"], amp.inputs[0])
            ng.links.new(sep.outputs["X"], xy.inputs["X"]); ng.links.new(sep.outputs["Y"], xy.inputs["Y"])
            ng.links.new(xy.outputs["Vector"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(cyl.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            pobj = bpy.data.objects.new("Pillar", bpy.data.meshes.new("PillarM"))
            pobj.location = (0, 0, 1.32)
            bpy.context.scene.collection.objects.link(pobj)
            pobj.modifiers.new("TwistGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.62, depth=0.22, location=(0, 0, 0.11))
            bpy.context.active_object.data.materials.append(m_granite)
            """,
            None,
        ),
        (
            "f1_07_teapot_body",
            "Build a dimpled porcelain ceramic teapot body with geometric radial facets, lid finial, and wooden tea tray.",
            """\
            m_porc = make_mat("CeladonPorcelain", (0.44, 0.66, 0.58), rough=0.18)
            m_tray = make_mat("TeakTray", (0.28, 0.15, 0.07), rough=0.62)
            ng = bpy.data.node_groups.new("TeapotBody", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeMeshUVSphere")
            sp.inputs["Segments"].default_value = 64
            sp.inputs["Rings"].default_value = 32
            sp.inputs["Radius"].default_value = 0.52
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            ang = ng.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
            ma = ng.nodes.new("ShaderNodeMath"); ma.operation = "MULTIPLY"; ma.inputs[1].default_value = 10.0
            mz = ng.nodes.new("ShaderNodeMath"); mz.operation = "MULTIPLY"; mz.inputs[1].default_value = 8.0
            ca = ng.nodes.new("ShaderNodeMath"); ca.operation = "COSINE"
            cz = ng.nodes.new("ShaderNodeMath"); cz.operation = "COSINE"
            prod = ng.nodes.new("ShaderNodeMath"); prod.operation = "MULTIPLY"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.042
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_porc
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Y"], ang.inputs[0]); ng.links.new(sep.outputs["X"], ang.inputs[1])
            ng.links.new(ang.outputs["Value"], ma.inputs[0]); ng.links.new(sep.outputs["Z"], mz.inputs[0])
            ng.links.new(ma.outputs["Value"], ca.inputs[0]); ng.links.new(mz.outputs["Value"], cz.inputs[0])
            ng.links.new(ca.outputs["Value"], prod.inputs[0]); ng.links.new(cz.outputs["Value"], prod.inputs[1])
            ng.links.new(prod.outputs["Value"], amp.inputs[0])
            ng.links.new(pos.outputs["Position"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(sp.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            obj = bpy.data.objects.new("Teapot", bpy.data.meshes.new("TeaM"))
            obj.location = (0, 0, 0.56)
            bpy.context.scene.collection.objects.link(obj)
            obj.modifiers.new("TeapotGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.85, depth=0.08, location=(0, 0, 0.04))
            bpy.context.active_object.data.materials.append(m_tray)
            """,
            None,
        ),
        (
            "f1_08_notched_flywheel",
            "Build a heavy cast-iron industrial notched flywheel drum with perimeter timing lugs and a polished steel center axle.",
            """\
            m_iron = pbr_mat("CastIron", (0.16, 0.17, 0.19), roughness=0.48, metal=0.88)
            m_steel = pbr_mat("PolishedSteel", (0.78, 0.80, 0.84), roughness=0.20, metal=0.98)
            ng = bpy.data.node_groups.new("NotchedFlywheel", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            cyl = ng.nodes.new("GeometryNodeMeshCylinder")
            cyl.inputs["Vertices"].default_value = 96
            cyl.inputs["Side Segments"].default_value = 6
            cyl.inputs["Radius"].default_value = 0.78
            cyl.inputs["Depth"].default_value = 0.34
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            ang = ng.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
            mul = ng.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 16.0
            sn = ng.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.075
            xy = ng.nodes.new("ShaderNodeCombineXYZ")
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_iron
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Y"], ang.inputs[0]); ng.links.new(sep.outputs["X"], ang.inputs[1])
            ng.links.new(ang.outputs["Value"], mul.inputs[0]); ng.links.new(mul.outputs["Value"], sn.inputs[0])
            ng.links.new(sn.outputs["Value"], amp.inputs[0])
            ng.links.new(sep.outputs["X"], xy.inputs["X"]); ng.links.new(sep.outputs["Y"], xy.inputs["Y"])
            ng.links.new(xy.outputs["Vector"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(cyl.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            fobj = bpy.data.objects.new("Flywheel", bpy.data.meshes.new("FlyM"))
            fobj.location = (0, 0, 0.25)
            bpy.context.scene.collection.objects.link(fobj)
            fobj.modifiers.new("WheelGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.14, depth=0.95, location=(0, 0, 0.45))
            bpy.context.active_object.data.materials.append(m_steel)
            """,
            "Plan: High-segment MeshCylinder (96 verts) extruded radially by SINE(ARCTAN2*16)*0.075 via SetPosition for flywheel lugs + polished steel shaft.",
        ),
        (
            "f1_09_stone_bollard",
            "Build a carved granite park bollard post with triple annular rings sculpted along Z and a dark bronze cap plate.",
            """\
            m_stone = make_mat("CarvedGranite", (0.52, 0.50, 0.48), rough=0.72)
            m_cap = make_mat("DarkBronze", (0.28, 0.22, 0.14), rough=0.35, metallic=0.85)
            ng = bpy.data.node_groups.new("StoneBollard", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            cyl = ng.nodes.new("GeometryNodeMeshCylinder")
            cyl.inputs["Vertices"].default_value = 48
            cyl.inputs["Side Segments"].default_value = 40
            cyl.inputs["Radius"].default_value = 0.28
            cyl.inputs["Depth"].default_value = 1.6
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            mz = ng.nodes.new("ShaderNodeMath"); mz.operation = "MULTIPLY"; mz.inputs[1].default_value = 12.0
            cz = ng.nodes.new("ShaderNodeMath"); cz.operation = "COSINE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.14
            xy = ng.nodes.new("ShaderNodeCombineXYZ")
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_stone
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Z"], mz.inputs[0]); ng.links.new(mz.outputs["Value"], cz.inputs[0])
            ng.links.new(cz.outputs["Value"], amp.inputs[0])
            ng.links.new(sep.outputs["X"], xy.inputs["X"]); ng.links.new(sep.outputs["Y"], xy.inputs["Y"])
            ng.links.new(xy.outputs["Vector"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(cyl.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            bobj = bpy.data.objects.new("Bollard", bpy.data.meshes.new("BolM"))
            bobj.location = (0, 0, 0.8)
            bpy.context.scene.collection.objects.link(bobj)
            bobj.modifiers.new("BollardGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.31, location=(0, 0, 1.62))
            bpy.context.active_object.data.materials.append(m_cap)
            """,
            None,
        ),
        (
            "f1_10_pleated_lampshade",
            "Build a pleated cream linen conical table lampshade with sharp radial accordion folds on a walnut-and-brass lamp base.",
            """\
            m_linen = pbr_mat("CreamLinen", (0.91, 0.86, 0.76), roughness=0.72)
            m_walnut = pbr_mat("LampWalnut", (0.26, 0.14, 0.06), roughness=0.42)
            ng = bpy.data.node_groups.new("PleatedShade", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            cone = ng.nodes.new("GeometryNodeMeshCone")
            cone.inputs["Vertices"].default_value = 96
            cone.inputs["Side Segments"].default_value = 12
            cone.inputs["Radius Top"].default_value = 0.28
            cone.inputs["Radius Bottom"].default_value = 0.58
            cone.inputs["Depth"].default_value = 0.62
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            ang = ng.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
            mf = ng.nodes.new("ShaderNodeMath"); mf.operation = "MULTIPLY"; mf.inputs[1].default_value = 24.0
            sn = ng.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
            ab = ng.nodes.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.075
            xy = ng.nodes.new("ShaderNodeCombineXYZ")
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_linen
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Y"], ang.inputs[0]); ng.links.new(sep.outputs["X"], ang.inputs[1])
            ng.links.new(ang.outputs["Value"], mf.inputs[0]); ng.links.new(mf.outputs["Value"], sn.inputs[0])
            ng.links.new(sn.outputs["Value"], ab.inputs[0]); ng.links.new(ab.outputs["Value"], amp.inputs[0])
            ng.links.new(sep.outputs["X"], xy.inputs["X"]); ng.links.new(sep.outputs["Y"], xy.inputs["Y"])
            ng.links.new(xy.outputs["Vector"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(cone.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            sobj = bpy.data.objects.new("LampShade", bpy.data.meshes.new("ShadeM"))
            sobj.location = (0, 0, 0.95)
            bpy.context.scene.collection.objects.link(sobj)
            sobj.modifiers.new("ShadeGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=28, radius=0.14, depth=0.8, location=(0, 0, 0.4))
            bpy.context.active_object.data.materials.append(m_walnut)
            """,
            None,
        ),
    ]
    for i, (rid, prompt, body, reason) in enumerate(specs):
        fn = f"create_{rid[6:]}" if i % 3 == 1 else None
        apply_obj = "obj" if (i == 2) else ("lobj" if i == 3 else None)
        if apply_obj and fn:
            apply_obj = None
        code = _wrap_script(i, _head(i), body, _tail(), fn_name=fn, apply_mod_obj=apply_obj)
        items.append({"id": rid, "prompt": prompt, "code": code, "reasoning": reason})
    return items


# ==============================================================================
# Family 2 (9 recipes): 2D Relief / Surface Math Displacement (MeshGrid)
# ==============================================================================
def _build_family2() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    specs = [
        (
            "f2_01_koi_pond_disc",
            "Build a rippled tranquil stone-rimmed water pond square with concentric surface wave rings displaced via radial SQRT math.",
            """\
            m_water = make_mat("PondWater", (0.12, 0.42, 0.48), rough=0.06)
            m_rim = make_mat("CopingStone", (0.42, 0.40, 0.36), rough=0.74)
            ng = bpy.data.node_groups.new("PondRipples", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 2.4; grid.inputs["Size Y"].default_value = 2.4
            grid.inputs["Vertices X"].default_value = 64; grid.inputs["Vertices Y"].default_value = 64
            pos = ng.nodes.new("GeometryNodeInputPosition")
            vlen = ng.nodes.new("ShaderNodeVectorMath"); vlen.operation = "LENGTH"
            mf = ng.nodes.new("ShaderNodeMath"); mf.operation = "MULTIPLY"; mf.inputs[1].default_value = 18.0
            sn = ng.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.045
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_water
            ng.links.new(pos.outputs["Position"], vlen.inputs[0])
            ng.links.new(vlen.outputs["Value"], mf.inputs[0]); ng.links.new(mf.outputs["Value"], sn.inputs[0])
            ng.links.new(sn.outputs["Value"], amp.inputs[0]); ng.links.new(amp.outputs["Value"], comb.inputs["Z"])
            ng.links.new(grid.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(comb.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"])
            ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            pobj = bpy.data.objects.new("PondSurface", bpy.data.meshes.new("PondM"))
            bpy.context.scene.collection.objects.link(pobj)
            pobj.modifiers.new("WaterGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.09))
            bpy.context.active_object.scale = (2.6, 2.6, 0.14)
            bpy.context.active_object.data.materials.append(m_rim)
            """,
            "Plan: MeshGrid (64x64) -> InputPosition -> VectorMath LENGTH -> MULTIPLY 18 -> SINE -> MULTIPLY 0.045 -> CombineXYZ Z -> SetPosition.",
        ),
        (
            "f2_02_terraced_hillside",
            "Build a sculpted topographic hillside landscape tile using Map Range and radial elevation fields on a dense Mesh Grid over a basalt base.",
            """\
            m_earth = pbr_mat("MossEarth", (0.24, 0.44, 0.18), roughness=0.78)
            m_basalt = pbr_mat("BasaltTray", (0.14, 0.14, 0.16), roughness=0.65)
            ng = bpy.data.node_groups.new("TerrainHill", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 2.6; grid.inputs["Size Y"].default_value = 2.6
            grid.inputs["Vertices X"].default_value = 56; grid.inputs["Vertices Y"].default_value = 56
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            mx = ng.nodes.new("ShaderNodeMath"); mx.operation = "MULTIPLY"; mx.inputs[1].default_value = 2.4
            my = ng.nodes.new("ShaderNodeMath"); my.operation = "MULTIPLY"; my.inputs[1].default_value = 2.1
            sx = ng.nodes.new("ShaderNodeMath"); sx.operation = "SINE"
            cy = ng.nodes.new("ShaderNodeMath"); cy.operation = "COSINE"
            sm_h = ng.nodes.new("ShaderNodeMath"); sm_h.operation = "ADD"
            mr = ng.nodes.new("ShaderNodeMapRange"); mr.interpolation_type = "STEPPED"; mr.inputs["From Min"].default_value = -2.0; mr.inputs["From Max"].default_value = 2.0; mr.inputs["To Min"].default_value = 0.0; mr.inputs["To Max"].default_value = 0.85; mr.inputs["Steps"].default_value = 6.0
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_earth
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["X"], mx.inputs[0]); ng.links.new(sep.outputs["Y"], my.inputs[0])
            ng.links.new(mx.outputs["Value"], sx.inputs[0]); ng.links.new(my.outputs["Value"], cy.inputs[0])
            ng.links.new(sx.outputs["Value"], sm_h.inputs[0]); ng.links.new(cy.outputs["Value"], sm_h.inputs[1])
            ng.links.new(sm_h.outputs["Value"], mr.inputs["Value"]); ng.links.new(mr.outputs["Result"], comb.inputs["Z"])
            ng.links.new(grid.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(comb.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            tobj = bpy.data.objects.new("HillsideTile", bpy.data.meshes.new("HillM"))
            bpy.context.scene.collection.objects.link(tobj)
            tobj.modifiers.new("TerrainGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.10))
            bpy.context.active_object.scale = (2.7, 2.7, 0.16)
            bpy.context.active_object.data.materials.append(m_basalt)
            """,
            None,
        ),
        (
            "f2_03_acoustic_baffle",
            "Build an architectural chevron acoustic wall baffle tile using ABSOLUTE(Y) + X sine displacement inside an aluminum surround.",
            """\
            m_felt = make_mat("AcousticFelt", (0.22, 0.28, 0.38), rough=0.88)
            m_alu = make_mat("AnodizedAlu", (0.76, 0.78, 0.80), rough=0.28, metallic=0.92)
            ng = bpy.data.node_groups.new("ChevronBaffle", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 2.2; grid.inputs["Size Y"].default_value = 2.2
            grid.inputs["Vertices X"].default_value = 72; grid.inputs["Vertices Y"].default_value = 72
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            ay = ng.nodes.new("ShaderNodeMath"); ay.operation = "ABSOLUTE"
            add_xy = ng.nodes.new("ShaderNodeMath"); add_xy.operation = "ADD"
            mf = ng.nodes.new("ShaderNodeMath"); mf.operation = "MULTIPLY"; mf.inputs[1].default_value = 10.0
            sn = ng.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.08
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_felt
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["Y"], ay.inputs[0])
            ng.links.new(sep.outputs["X"], add_xy.inputs[0]); ng.links.new(ay.outputs["Value"], add_xy.inputs[1])
            ng.links.new(add_xy.outputs["Value"], mf.inputs[0]); ng.links.new(mf.outputs["Value"], sn.inputs[0])
            ng.links.new(sn.outputs["Value"], amp.inputs[0]); ng.links.new(amp.outputs["Value"], comb.inputs["Z"])
            ng.links.new(grid.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(comb.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            bobj = bpy.data.objects.new("Baffle", bpy.data.meshes.new("BafM"))
            bpy.context.scene.collection.objects.link(bobj)
            bobj.modifiers.new("ChevronGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.07))
            bpy.context.active_object.scale = (2.35, 2.35, 0.10)
            bpy.context.active_object.data.materials.append(m_alu)
            """,
            None,
        ),
        (
            "f2_04_water_medallion",
            "Build a concentric carved bronze medallion plaque sculpted with circular cosine interference ridges mounted on an oak backing board.",
            """\
            m_bronze = pbr_mat("AgedBronze", (0.58, 0.42, 0.18), roughness=0.32, metal=0.90)
            m_oak = pbr_mat("DarkOak", (0.24, 0.14, 0.07), roughness=0.58)
            ng = bpy.data.node_groups.new("BronzeMedallion", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 2.0; grid.inputs["Size Y"].default_value = 2.0
            grid.inputs["Vertices X"].default_value = 64; grid.inputs["Vertices Y"].default_value = 64
            pos = ng.nodes.new("GeometryNodeInputPosition")
            vlen = ng.nodes.new("ShaderNodeVectorMath"); vlen.operation = "LENGTH"
            mf = ng.nodes.new("ShaderNodeMath"); mf.operation = "MULTIPLY"; mf.inputs[1].default_value = 22.0
            cs = ng.nodes.new("ShaderNodeMath"); cs.operation = "COSINE"
            ab = ng.nodes.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.09
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_bronze
            ng.links.new(pos.outputs["Position"], vlen.inputs[0])
            ng.links.new(vlen.outputs["Value"], mf.inputs[0]); ng.links.new(mf.outputs["Value"], cs.inputs[0])
            ng.links.new(cs.outputs["Value"], ab.inputs[0]); ng.links.new(ab.outputs["Value"], amp.inputs[0])
            ng.links.new(amp.outputs["Value"], comb.inputs["Z"])
            ng.links.new(grid.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(comb.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"])
            ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            mobj = bpy.data.objects.new("Medallion", bpy.data.meshes.new("MedM"))
            bpy.context.scene.collection.objects.link(mobj)
            mobj.modifiers.new("MedallionGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.06))
            bpy.context.active_object.scale = (2.18, 2.18, 0.12)
            bpy.context.active_object.data.materials.append(m_oak)
            """,
            None,
        ),
        (
            "f2_05_sand_dune_slab",
            "Build a wind-carved desert sand dune terrain slab combining asymmetric ridge math and fine procedural Noise Texture wind ripples.",
            """\
            m_sand = make_mat("DesertSand", (0.84, 0.68, 0.44), rough=0.82)
            m_rock = make_mat("SandstoneSlab", (0.48, 0.34, 0.22), rough=0.76)
            ng = bpy.data.node_groups.new("SandDune", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 3.0; grid.inputs["Size Y"].default_value = 3.0
            grid.inputs["Vertices X"].default_value = 64; grid.inputs["Vertices Y"].default_value = 64
            pos = ng.nodes.new("GeometryNodeInputPosition")
            wave = ng.nodes.new("ShaderNodeTexWave")
            wave.inputs["Scale"].default_value = 1.6; wave.inputs["Distortion"].default_value = 4.2; wave.inputs["Detail"].default_value = 3.0
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.42
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_sand
            ng.links.new(pos.outputs["Position"], wave.inputs["Vector"])
            ng.links.new(wave.outputs["Factor"], amp.inputs[0]); ng.links.new(amp.outputs["Value"], comb.inputs["Z"])
            ng.links.new(grid.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(comb.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"])
            ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            sobj = bpy.data.objects.new("Dunes", bpy.data.meshes.new("DuneM"))
            bpy.context.scene.collection.objects.link(sobj)
            sobj.modifiers.new("DuneGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.10))
            bpy.context.active_object.scale = (3.1, 3.1, 0.16)
            bpy.context.active_object.data.materials.append(m_rock)
            """,
            "Plan: Feed InputPosition into ShaderNodeTexWave (Scale=1.6, Distortion=4.2), take outputs['Factor']*0.42 -> Z offset on MeshGrid (64x64).",
        ),
        (
            "f2_06_quilted_cushion",
            "Build a diamond-quilted tufted burgundy leather upholstery tile using product of ABS(SINE) waves on a dense grid mounted on a brass trim frame.",
            """\
            m_leather = pbr_mat("BurgundyLeather", (0.36, 0.08, 0.10), roughness=0.42)
            m_trim = pbr_mat("BrassFrame", (0.78, 0.58, 0.22), roughness=0.25, metal=0.92)
            ng = bpy.data.node_groups.new("QuiltedTile", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 2.0; grid.inputs["Size Y"].default_value = 2.0
            grid.inputs["Vertices X"].default_value = 64; grid.inputs["Vertices Y"].default_value = 64
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            mx = ng.nodes.new("ShaderNodeMath"); mx.operation = "MULTIPLY"; mx.inputs[1].default_value = 7.5
            my = ng.nodes.new("ShaderNodeMath"); my.operation = "MULTIPLY"; my.inputs[1].default_value = 7.5
            sx = ng.nodes.new("ShaderNodeMath"); sx.operation = "SINE"
            sy = ng.nodes.new("ShaderNodeMath"); sy.operation = "SINE"
            prod = ng.nodes.new("ShaderNodeMath"); prod.operation = "MULTIPLY"
            ab = ng.nodes.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.14
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_leather
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["X"], mx.inputs[0]); ng.links.new(sep.outputs["Y"], my.inputs[0])
            ng.links.new(mx.outputs["Value"], sx.inputs[0]); ng.links.new(my.outputs["Value"], sy.inputs[0])
            ng.links.new(sx.outputs["Value"], prod.inputs[0]); ng.links.new(sy.outputs["Value"], prod.inputs[1])
            ng.links.new(prod.outputs["Value"], ab.inputs[0]); ng.links.new(ab.outputs["Value"], amp.inputs[0])
            ng.links.new(amp.outputs["Value"], comb.inputs["Z"])
            ng.links.new(grid.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(comb.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"])
            ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            qobj = bpy.data.objects.new("QuiltTile", bpy.data.meshes.new("QuiltM"))
            bpy.context.scene.collection.objects.link(qobj)
            qobj.modifiers.new("QuiltGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.06))
            bpy.context.active_object.scale = (2.15, 2.15, 0.10)
            bpy.context.active_object.data.materials.append(m_trim)
            """,
            None,
        ),
        (
            "f2_07_parabolic_shell",
            "Build an architectural concrete hyperbolic-paraboloid saddle roof canopy sculpted via Z = 0.35*(X^2 - Y^2) supported on dark steel corner piers.",
            """\
            m_conc = make_mat("WhiteConcrete", (0.82, 0.81, 0.78), rough=0.64)
            m_steel = make_mat("DarkSteel", (0.14, 0.15, 0.18), rough=0.38, metallic=0.88)
            ng = bpy.data.node_groups.new("HyparShell", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 2.6; grid.inputs["Size Y"].default_value = 2.6
            grid.inputs["Vertices X"].default_value = 48; grid.inputs["Vertices Y"].default_value = 48
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            px = ng.nodes.new("ShaderNodeMath"); px.operation = "POWER"; px.inputs[1].default_value = 2.0
            py = ng.nodes.new("ShaderNodeMath"); py.operation = "POWER"; py.inputs[1].default_value = 2.0
            sub = ng.nodes.new("ShaderNodeMath"); sub.operation = "SUBTRACT"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.35
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_conc
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["X"], px.inputs[0]); ng.links.new(sep.outputs["Y"], py.inputs[0])
            ng.links.new(px.outputs["Value"], sub.inputs[0]); ng.links.new(py.outputs["Value"], sub.inputs[1])
            ng.links.new(sub.outputs["Value"], amp.inputs[0]); ng.links.new(amp.outputs["Value"], comb.inputs["Z"])
            ng.links.new(grid.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(comb.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            hobj = bpy.data.objects.new("HyparRoof", bpy.data.meshes.new("HypM"))
            hobj.location = (0, 0, 1.0)
            bpy.context.scene.collection.objects.link(hobj)
            hobj.modifiers.new("HyparGN", "NODES").node_group = ng
            for sx in (-1.2, 1.2):
                bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.08, depth=1.55, location=(sx, 0, 0.77))
                bpy.context.active_object.data.materials.append(m_steel)
            """,
            None,
        ),
        (
            "f2_08_amphitheater_steps",
            "Build a concentric stepped travertine plaza amphitheater terrace with circular terraced elevation levels around a bronze stage disc.",
            """\
            m_trav = pbr_mat("TravertineStone", (0.76, 0.70, 0.60), roughness=0.66)
            m_stage = pbr_mat("StageBronze", (0.54, 0.40, 0.20), roughness=0.35, metal=0.82)
            ng = bpy.data.node_groups.new("SteppedPlaza", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 3.0; grid.inputs["Size Y"].default_value = 3.0
            grid.inputs["Vertices X"].default_value = 64; grid.inputs["Vertices Y"].default_value = 64
            pos = ng.nodes.new("GeometryNodeInputPosition")
            vlen = ng.nodes.new("ShaderNodeVectorMath"); vlen.operation = "LENGTH"
            mr = ng.nodes.new("ShaderNodeMapRange"); mr.interpolation_type = "STEPPED"
            mr.inputs["From Min"].default_value = 0.4; mr.inputs["From Max"].default_value = 1.9
            mr.inputs["To Min"].default_value = 0.0; mr.inputs["To Max"].default_value = 0.75; mr.inputs["Steps"].default_value = 5.0
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_trav
            ng.links.new(pos.outputs["Position"], vlen.inputs[0])
            ng.links.new(vlen.outputs["Value"], mr.inputs["Value"]); ng.links.new(mr.outputs["Result"], comb.inputs["Z"])
            ng.links.new(grid.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(comb.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            pobj = bpy.data.objects.new("PlazaTerrace", bpy.data.meshes.new("PlzM"))
            bpy.context.scene.collection.objects.link(pobj)
            pobj.modifiers.new("PlazaGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.45, depth=0.06, location=(0, 0, 0.03))
            bpy.context.active_object.data.materials.append(m_stage)
            """,
            None,
        ),
        (
            "f2_09_ribbon_facade",
            "Build a standing curved architectural metallic facade screen rotated upright and sculpted with diagonal sinusoidal waves above a dark steel plinth.",
            """\
            m_copper = make_mat("CopperFacade", (0.76, 0.42, 0.24), rough=0.28, metallic=0.90)
            m_base = make_mat("BaseSteel", (0.15, 0.16, 0.18), rough=0.46, metallic=0.85)
            ng = bpy.data.node_groups.new("FacadeScreen", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 2.8; grid.inputs["Size Y"].default_value = 1.8
            grid.inputs["Vertices X"].default_value = 72; grid.inputs["Vertices Y"].default_value = 32
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            mx = ng.nodes.new("ShaderNodeMath"); mx.operation = "MULTIPLY"; mx.inputs[1].default_value = 5.5
            my = ng.nodes.new("ShaderNodeMath"); my.operation = "MULTIPLY"; my.inputs[1].default_value = 3.5
            add = ng.nodes.new("ShaderNodeMath"); add.operation = "ADD"
            sn = ng.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.18
            comb = ng.nodes.new("ShaderNodeCombineXYZ")
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            xf = ng.nodes.new("GeometryNodeTransform"); xf.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); xf.inputs["Translation"].default_value = (0.0, 0.0, 1.0)
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_copper
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])
            ng.links.new(sep.outputs["X"], mx.inputs[0]); ng.links.new(sep.outputs["Y"], my.inputs[0])
            ng.links.new(mx.outputs["Value"], add.inputs[0]); ng.links.new(my.outputs["Value"], add.inputs[1])
            ng.links.new(add.outputs["Value"], sn.inputs[0]); ng.links.new(sn.outputs["Value"], amp.inputs[0])
            ng.links.new(amp.outputs["Value"], comb.inputs["Z"])
            ng.links.new(grid.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(comb.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], xf.inputs["Geometry"]); ng.links.new(xf.outputs["Geometry"], smooth.inputs["Mesh"])
            ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            fobj = bpy.data.objects.new("Facade", bpy.data.meshes.new("FacM"))
            bpy.context.scene.collection.objects.link(fobj)
            fobj.modifiers.new("FacadeGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.08))
            bpy.context.active_object.scale = (3.0, 0.65, 0.16)
            bpy.context.active_object.data.materials.append(m_base)
            """,
            "Plan: Displace flat MeshGrid with SINE(5.5*X + 3.5*Y)*0.18, stand it upright via GeometryNodeTransform Rotation=(pi/2,0,0) + Translation=(0,0,1), and add steel plinth.",
        ),
    ]
    for i, (rid, prompt, body, reason) in enumerate(specs):
        fn = f"build_{rid[6:]}" if i % 3 == 2 else None
        apply_obj = "pobj" if i == 0 and fn is None else None
        code = _wrap_script(i, _head(i), body, _tail(cam_loc=(4.0, -4.6, 3.4)), fn_name=fn, apply_mod_obj=apply_obj)
        items.append({"id": rid, "prompt": prompt, "code": code, "reasoning": reason})
    return items


# ==============================================================================
# Family 3 (10 recipes): Curve Sweeps (Spiral, Bezier, Arc, Star, CurveToMesh)
# ==============================================================================
def _build_family3() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    specs = [
        (
            "f3_01_shepherd_crook",
            "Build a carved ash-wood shepherd's crook staff combining a straight vertical shaft line and a hooked upper curve arc swept via Curve to Mesh with Fill Caps and a brass ferrule.",
            """\
            m_ash = make_mat("AshWood", (0.54, 0.36, 0.18), rough=0.46)
            m_brass = make_mat("FerruleBrass", (0.80, 0.62, 0.22), rough=0.24, metallic=0.92)
            ng = bpy.data.node_groups.new("ShepherdCrook", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            line = ng.nodes.new("GeometryNodeCurvePrimitiveLine")
            line.inputs["Start"].default_value = (0.0, 0.0, 0.0); line.inputs["End"].default_value = (0.0, 0.0, 1.8)
            arc = ng.nodes.new("GeometryNodeCurveArc")
            arc.inputs["Resolution"].default_value = 32; arc.inputs["Radius"].default_value = 0.24
            arc.inputs["Start Angle"].default_value = 0.0; arc.inputs["Sweep Angle"].default_value = 3.6
            arc_xf = ng.nodes.new("GeometryNodeTransform")
            arc_xf.inputs["Translation"].default_value = (-0.24, 0.0, 1.8)
            arc_xf.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0)
            join = ng.nodes.new("GeometryNodeJoinGeometry")
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); prof.inputs["Resolution"].default_value = 18; prof.inputs["Radius"].default_value = 0.038
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_ash
            ng.links.new(arc.outputs["Curve"], arc_xf.inputs["Geometry"])
            ng.links.new(line.outputs["Curve"], join.inputs["Geometry"])
            ng.links.new(arc_xf.outputs["Geometry"], join.inputs["Geometry"])
            ng.links.new(join.outputs["Geometry"], c2m.inputs["Curve"]); ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            obj = bpy.data.objects.new("Crook", bpy.data.meshes.new("CrookM"))
            bpy.context.scene.collection.objects.link(obj)
            obj.modifiers.new("CrookGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.048, depth=0.14, location=(0, 0, 0.07))
            bpy.context.active_object.data.materials.append(m_brass)
            """,
            "Plan: Join GeometryNodeCurvePrimitiveLine vertical shaft + upright GeometryNodeCurveArc hook, sweep with CurvePrimitiveCircle via GeometryNodeCurveToMesh (Fill Caps=True).",
        ),
        (
            "f3_02_iron_bracket",
            "Build an ornamental wrought-iron S-scroll wall bracket using twin Quadratic Bezier curves swept with a circular profile on an oak mounting plaque.",
            """\
            m_iron = pbr_mat("WroughtIron", (0.11, 0.12, 0.13), roughness=0.45, metal=0.90)
            m_oak = pbr_mat("WallOak", (0.34, 0.20, 0.10), roughness=0.58)
            ng = bpy.data.node_groups.new("IronScrollBracket", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            c1 = ng.nodes.new("GeometryNodeCurveQuadraticBezier")
            c1.inputs["Resolution"].default_value = 32
            c1.inputs["Start"].default_value = (0.0, 0.0, 0.0); c1.inputs["Middle"].default_value = (0.35, 0.0, 0.75); c1.inputs["End"].default_value = (1.15, 0.0, 0.85)
            c2 = ng.nodes.new("GeometryNodeCurveQuadraticBezier")
            c2.inputs["Resolution"].default_value = 32
            c2.inputs["Start"].default_value = (0.0, 0.0, 0.92); c2.inputs["Middle"].default_value = (0.55, 0.0, 0.92); c2.inputs["End"].default_value = (1.25, 0.0, 0.92)
            join = ng.nodes.new("GeometryNodeJoinGeometry")
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); prof.inputs["Resolution"].default_value = 16; prof.inputs["Radius"].default_value = 0.042
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_iron
            ng.links.new(c1.outputs["Curve"], join.inputs["Geometry"]); ng.links.new(c2.outputs["Curve"], join.inputs["Geometry"])
            ng.links.new(join.outputs["Geometry"], c2m.inputs["Curve"]); ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            bobj = bpy.data.objects.new("WallBracket", bpy.data.meshes.new("BrkM"))
            bpy.context.scene.collection.objects.link(bobj)
            bobj.modifiers.new("BracketGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-0.06, 0.0, 0.48))
            bpy.context.active_object.scale = (0.10, 0.28, 1.25)
            bpy.context.active_object.data.materials.append(m_oak)
            """,
            None,
        ),
        (
            "f3_03_ram_horn",
            "Build a tapered ridged ivory ram horn swept along a Curve Spiral whose radius shrinks toward the tip using Spline Parameter and Set Curve Radius.",
            """\
            m_ivory = make_mat("CarvedIvory", (0.86, 0.80, 0.66), rough=0.34)
            m_base = make_mat("EbonyPedestal", (0.10, 0.10, 0.11), rough=0.45)
            ng = bpy.data.node_groups.new("TaperedHorn", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 48; sp.inputs["Rotations"].default_value = 1.65
            sp.inputs["Start Radius"].default_value = 0.58; sp.inputs["End Radius"].default_value = 0.16; sp.inputs["Height"].default_value = 1.15
            spar = ng.nodes.new("GeometryNodeSplineParameter")
            mr = ng.nodes.new("ShaderNodeMapRange")
            mr.inputs["From Min"].default_value = 0.0; mr.inputs["From Max"].default_value = 1.0
            mr.inputs["To Min"].default_value = 0.16; mr.inputs["To Max"].default_value = 0.025
            srad = ng.nodes.new("GeometryNodeSetCurveRadius")
            prof = ng.nodes.new("GeometryNodeCurveStar"); prof.inputs["Points"].default_value = 12; prof.inputs["Inner Radius"].default_value = 0.82; prof.inputs["Outer Radius"].default_value = 1.0
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_ivory
            ng.links.new(spar.outputs["Factor"], mr.inputs["Value"])
            ng.links.new(sp.outputs["Curve"], srad.inputs["Curve"]); ng.links.new(mr.outputs["Result"], srad.inputs["Radius"])
            ng.links.new(srad.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            hobj = bpy.data.objects.new("RamHorn", bpy.data.meshes.new("HornM"))
            hobj.location = (0, 0, 0.25)
            bpy.context.scene.collection.objects.link(hobj)
            hobj.modifiers.new("HornGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.12))
            bpy.context.active_object.scale = (0.9, 0.9, 0.24)
            bpy.context.active_object.data.materials.append(m_base)
            """,
            "Plan: Use CurveSpiral -> SetCurveRadius driven by SplineParameter.Factor through MapRange (0.16->0.025), sweep CurveStar profile with CurveToMesh(Fill Caps=True).",
        ),
        (
            "f3_04_desk_lamp_arm",
            "Build a matte-black architect gooseneck desk lamp arm swept along a Quadratic Bezier with a conical brushed aluminum lamp shade and weighted base.",
            """\
            m_blk = pbr_mat("MatteBlack", (0.10, 0.10, 0.12), roughness=0.48, metal=0.6)
            m_alu = pbr_mat("BrushedAlu", (0.78, 0.80, 0.83), roughness=0.22, metal=0.96)
            ng = bpy.data.node_groups.new("GooseneckArm", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            bz = ng.nodes.new("GeometryNodeCurveQuadraticBezier")
            bz.inputs["Resolution"].default_value = 36
            bz.inputs["Start"].default_value = (0.0, 0.0, 0.08); bz.inputs["Middle"].default_value = (-0.28, 0.0, 1.38); bz.inputs["End"].default_value = (0.52, 0.0, 1.05)
            cir = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); cir.inputs["Resolution"].default_value = 20; cir.inputs["Radius"].default_value = 0.032
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_blk
            ng.links.new(bz.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(cir.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            lobj = bpy.data.objects.new("LampArm", bpy.data.meshes.new("ArmM"))
            bpy.context.scene.collection.objects.link(lobj)
            lobj.modifiers.new("ArmGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.30, depth=0.09, location=(0, 0, 0.045))
            bpy.context.active_object.data.materials.append(m_blk)
            bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=0.22, radius2=0.06, depth=0.28, location=(0.56, 0, 0.94))
            bpy.context.active_object.rotation_euler = (0.0, 0.42, 0.0)
            bpy.context.active_object.data.materials.append(m_alu)
            """,
            None,
        ),
        (
            "f3_05_star_finial",
            "Build a twisted fluted wrought-iron gate post finial by sweeping a Curve Star profile along a resampled line with Set Curve Tilt and Set Curve Radius taper.",
            """\
            m_iron = make_mat("GateIron", (0.14, 0.15, 0.17), rough=0.38, metallic=0.92)
            m_stone = make_mat("PierCap", (0.58, 0.56, 0.52), rough=0.72)
            ng = bpy.data.node_groups.new("TwistedFinial", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ln = ng.nodes.new("GeometryNodeCurvePrimitiveLine")
            ln.inputs["Start"].default_value = (0, 0, 0.22); ln.inputs["End"].default_value = (0, 0, 1.65)
            res = ng.nodes.new("GeometryNodeResampleCurve"); res.inputs["Count"].default_value = 48
            sp = ng.nodes.new("GeometryNodeSplineParameter")
            tw = ng.nodes.new("ShaderNodeMath"); tw.operation = "MULTIPLY"; tw.inputs[1].default_value = 6.283
            stilt = ng.nodes.new("GeometryNodeSetCurveTilt")
            mr = ng.nodes.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = 0.22; mr.inputs["To Max"].default_value = 0.015
            srad = ng.nodes.new("GeometryNodeSetCurveRadius")
            star = ng.nodes.new("GeometryNodeCurveStar"); star.inputs["Points"].default_value = 6; star.inputs["Inner Radius"].default_value = 0.45; star.inputs["Outer Radius"].default_value = 1.0
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_iron
            ng.links.new(ln.outputs["Curve"], res.inputs["Curve"])
            ng.links.new(sp.outputs["Factor"], tw.inputs[0]); ng.links.new(res.outputs["Curve"], stilt.inputs["Curve"])
            ng.links.new(tw.outputs["Value"], stilt.inputs["Tilt"]); ng.links.new(sp.outputs["Factor"], mr.inputs["Value"])
            ng.links.new(stilt.outputs["Curve"], srad.inputs["Curve"]); ng.links.new(mr.outputs["Result"], srad.inputs["Radius"])
            ng.links.new(srad.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(star.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            obj = bpy.data.objects.new("Finial", bpy.data.meshes.new("FinM"))
            bpy.context.scene.collection.objects.link(obj)
            obj.modifiers.new("FinialGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.11))
            bpy.context.active_object.scale = (0.72, 0.72, 0.22)
            bpy.context.active_object.data.materials.append(m_stone)
            """,
            "Plan: Resample vertical CurvePrimitiveLine (Count=48), twist via SetCurveTilt (Factor*2pi) and taper via SetCurveRadius (0.22->0.015), sweep CurveStar (6 points).",
        ),
        (
            "f3_06_arch_bridge_rib",
            "Build a curved structural steel arch bridge span rib with twin end masonry abutments using Quadratic Bezier curve sweeping.",
            """\
            m_steel = pbr_mat("BridgeSteel", (0.24, 0.30, 0.36), roughness=0.35, metal=0.90)
            m_masonry = pbr_mat("AbutmentStone", (0.50, 0.48, 0.44), roughness=0.78)
            ng = bpy.data.node_groups.new("BridgeArchRib", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            bz = ng.nodes.new("GeometryNodeCurveQuadraticBezier")
            bz.inputs["Resolution"].default_value = 48
            bz.inputs["Start"].default_value = (-1.6, 0.0, 0.2); bz.inputs["Middle"].default_value = (0.0, 0.0, 2.2); bz.inputs["End"].default_value = (1.6, 0.0, 0.2)
            star = ng.nodes.new("GeometryNodeCurveStar"); star.inputs["Points"].default_value = 4; star.inputs["Inner Radius"].default_value = 0.09; star.inputs["Outer Radius"].default_value = 0.12
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_steel
            ng.links.new(bz.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(star.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            robj = bpy.data.objects.new("ArchRib", bpy.data.meshes.new("RibM"))
            bpy.context.scene.collection.objects.link(robj)
            robj.modifiers.new("RibGN", "NODES").node_group = ng
            for xx in (-1.65, 1.65):
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=(xx, 0, 0.18))
                bpy.context.active_object.scale = (0.48, 0.56, 0.36)
                bpy.context.active_object.data.materials.append(m_masonry)
            """,
            None,
        ),
        (
            "f3_07_trumpet_horn",
            "Build a polished brass flared musical horn bell swept with exponential Set Curve Radius growth along a curved centerline on a velvet stand.",
            """\
            m_brass = make_mat("HornBrass", (0.86, 0.68, 0.22), rough=0.16, metallic=0.98)
            m_velvet = make_mat("VelvetStand", (0.22, 0.06, 0.10), rough=0.84)
            ng = bpy.data.node_groups.new("TrumpetBell", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            bz = ng.nodes.new("GeometryNodeCurveQuadraticBezier")
            bz.inputs["Resolution"].default_value = 48
            bz.inputs["Start"].default_value = (-1.1, 0.0, 0.55); bz.inputs["Middle"].default_value = (0.25, 0.0, 0.55); bz.inputs["End"].default_value = (1.1, 0.0, 0.85)
            sp = ng.nodes.new("GeometryNodeSplineParameter")
            pw = ng.nodes.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 4.0
            mr = ng.nodes.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = 0.045; mr.inputs["To Max"].default_value = 0.36
            sr = ng.nodes.new("GeometryNodeSetCurveRadius")
            cir = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); cir.inputs["Resolution"].default_value = 32; cir.inputs["Radius"].default_value = 1.0
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_brass
            ng.links.new(sp.outputs["Factor"], pw.inputs[0]); ng.links.new(pw.outputs["Value"], mr.inputs["Value"])
            ng.links.new(bz.outputs["Curve"], sr.inputs["Curve"]); ng.links.new(mr.outputs["Result"], sr.inputs["Radius"])
            ng.links.new(sr.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(cir.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            obj = bpy.data.objects.new("HornBell", bpy.data.meshes.new("BellM"))
            bpy.context.scene.collection.objects.link(obj)
            obj.modifiers.new("HornGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.14))
            bpy.context.active_object.scale = (1.6, 0.5, 0.28)
            bpy.context.active_object.data.materials.append(m_velvet)
            """,
            None,
        ),
        (
            "f3_08_candlestick_stem",
            "Build a baroque twisted bronze candlestick holder stem swept with a scalloped star curve and topped with a cream beeswax pillar candle.",
            """\
            m_bronze = pbr_mat("AgedBronze", (0.56, 0.40, 0.16), roughness=0.30, metal=0.90)
            m_wax = pbr_mat("BeeswaxCandle", (0.93, 0.88, 0.72), roughness=0.45)
            ng = bpy.data.node_groups.new("CandlestickStem", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ln = ng.nodes.new("GeometryNodeCurvePrimitiveLine")
            ln.inputs["Start"].default_value = (0, 0, 0.08); ln.inputs["End"].default_value = (0, 0, 1.05)
            res = ng.nodes.new("GeometryNodeResampleCurve"); res.inputs["Count"].default_value = 40
            sp = ng.nodes.new("GeometryNodeSplineParameter")
            tw = ng.nodes.new("ShaderNodeMath"); tw.operation = "MULTIPLY"; tw.inputs[1].default_value = 9.42
            st = ng.nodes.new("GeometryNodeSetCurveTilt")
            star = ng.nodes.new("GeometryNodeCurveStar"); star.inputs["Points"].default_value = 8; star.inputs["Inner Radius"].default_value = 0.09; star.inputs["Outer Radius"].default_value = 0.14
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_bronze
            ng.links.new(ln.outputs["Curve"], res.inputs["Curve"]); ng.links.new(sp.outputs["Factor"], tw.inputs[0])
            ng.links.new(res.outputs["Curve"], st.inputs["Curve"]); ng.links.new(tw.outputs["Value"], st.inputs["Tilt"])
            ng.links.new(st.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(star.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            cobj = bpy.data.objects.new("Candlestick", bpy.data.meshes.new("CndM"))
            bpy.context.scene.collection.objects.link(cobj)
            cobj.modifiers.new("CandleGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=28, radius=0.26, depth=0.08, location=(0, 0, 0.04))
            bpy.context.active_object.data.materials.append(m_bronze)
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.09, depth=0.45, location=(0, 0, 1.28))
            bpy.context.active_object.data.materials.append(m_wax)
            """,
            None,
        ),
        (
            "f3_09_copper_condenser_coil",
            "Build an industrial copper distiller condenser helix with 6 stacked loops swept cleanly around a brushed steel core manifold.",
            """\
            m_copper = make_mat("TubingCopper", (0.82, 0.44, 0.24), rough=0.20, metallic=0.96)
            m_steel = make_mat("ManifoldSteel", (0.58, 0.60, 0.64), rough=0.28, metallic=0.92)
            ng = bpy.data.node_groups.new("CopperCondenser", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeCurveSpiral")
            sp.inputs["Resolution"].default_value = 36; sp.inputs["Rotations"].default_value = 6.0
            sp.inputs["Start Radius"].default_value = 0.55; sp.inputs["End Radius"].default_value = 0.55; sp.inputs["Height"].default_value = 1.4
            cir = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); cir.inputs["Resolution"].default_value = 18; cir.inputs["Radius"].default_value = 0.055
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_copper
            ng.links.new(sp.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(cir.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            obj = bpy.data.objects.new("Condenser", bpy.data.meshes.new("CondM"))
            obj.location = (0, 0, 0.15)
            bpy.context.scene.collection.objects.link(obj)
            obj.modifiers.new("CondGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.32, depth=1.65, location=(0, 0, 0.85))
            bpy.context.active_object.data.materials.append(m_steel)
            """,
            None,
        ),
        (
            "f3_10_ribbon_arch",
            "Build a ceremonial twisted brass ribbon gateway arch spanning between twin limestone pedestals using Curve Arc, Set Curve Tilt, and Curve Star.",
            """\
            m_gold = pbr_mat("GatewayGold", (0.82, 0.64, 0.24), roughness=0.22, metal=0.95)
            m_lime = pbr_mat("LimestonePlinth", (0.78, 0.75, 0.68), roughness=0.68)
            ng = bpy.data.node_groups.new("RibbonArch", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            arc = ng.nodes.new("GeometryNodeCurveArc")
            arc.inputs["Resolution"].default_value = 64; arc.inputs["Radius"].default_value = 1.25
            arc.inputs["Start Angle"].default_value = 0.0; arc.inputs["Sweep Angle"].default_value = 3.14159
            sp = ng.nodes.new("GeometryNodeSplineParameter")
            tw = ng.nodes.new("ShaderNodeMath"); tw.operation = "MULTIPLY"; tw.inputs[1].default_value = 9.42
            tilt = ng.nodes.new("GeometryNodeSetCurveTilt")
            star = ng.nodes.new("GeometryNodeCurveStar"); star.inputs["Points"].default_value = 4; star.inputs["Inner Radius"].default_value = 0.06; star.inputs["Outer Radius"].default_value = 0.16
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            xf = ng.nodes.new("GeometryNodeTransform"); xf.inputs["Rotation"].default_value = (1.5708, 0.0, 0.0); xf.inputs["Translation"].default_value = (0.0, 0.0, 0.35)
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_gold
            ng.links.new(sp.outputs["Factor"], tw.inputs[0])
            ng.links.new(arc.outputs["Curve"], tilt.inputs["Curve"]); ng.links.new(tw.outputs["Value"], tilt.inputs["Tilt"])
            ng.links.new(tilt.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(star.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], xf.inputs["Geometry"]); ng.links.new(xf.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            aobj = bpy.data.objects.new("CeremonialArch", bpy.data.meshes.new("ArchM"))
            bpy.context.scene.collection.objects.link(aobj)
            aobj.modifiers.new("ArchGN", "NODES").node_group = ng
            for xx in (-1.25, 1.25):
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=(xx, 0, 0.18))
                bpy.context.active_object.scale = (0.42, 0.42, 0.36)
                bpy.context.active_object.data.materials.append(m_lime)
            """,
            "Plan: Semicircular CurveArc -> SetCurveTilt (Factor*3pi) -> CurveToMesh with 4-point CurveStar -> Transform Rotation=(pi/2,0,0) onto limestone plinths.",
        ),
    ]
    for i, (rid, prompt, body, reason) in enumerate(specs):
        fn = f"make_{rid[6:]}" if i % 4 == 1 else None
        apply_obj = "obj" if i == 4 and fn is None else None
        code = _wrap_script(i, _head(i), body, _tail(cam_loc=(4.5, -5.0, 3.2)), fn_name=fn, apply_mod_obj=apply_obj)
        items.append({"id": rid, "prompt": prompt, "code": code, "reasoning": reason})
    return items


# ==============================================================================
# Family 4 (9 recipes): Point Scattering & Instancing
# ==============================================================================
def _build_family4() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    specs = [
        (
            "f4_01_lotus_blossom",
            "Build a stylized sacred pink lotus blossom using Curve to Points around a circle, Align Rotation to Vector, and Instance on Points around a golden seed pod.",
            """\
            m_petal = make_mat("LotusPink", (0.90, 0.48, 0.62), rough=0.38)
            m_pod = make_mat("GoldenPod", (0.86, 0.68, 0.18), rough=0.52)
            ng = bpy.data.node_groups.new("LotusBlossom", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ring = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ring.inputs["Resolution"].default_value = 24; ring.inputs["Radius"].default_value = 0.45
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 12
            sp = ng.nodes.new("GeometryNodeMeshUVSphere"); sp.inputs["Segments"].default_value = 20; sp.inputs["Rings"].default_value = 12; sp.inputs["Radius"].default_value = 0.25
            pt_xf = ng.nodes.new("GeometryNodeTransform"); pt_xf.inputs["Scale"].default_value = (1.6, 0.55, 0.22); pt_xf.inputs["Translation"].default_value = (0.35, 0.0, 0.12)
            al = ng.nodes.new("FunctionNodeAlignRotationToVector")
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_petal
            ng.links.new(ring.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(sp.outputs["Mesh"], pt_xf.inputs["Geometry"])
            ng.links.new(c2p.outputs["Normal"], al.inputs["Vector"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"])
            ng.links.new(pt_xf.outputs["Geometry"], inst.inputs["Instance"])
            ng.links.new(al.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
            ng.links.new(real.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            lobj = bpy.data.objects.new("Lotus", bpy.data.meshes.new("LotM"))
            lobj.location = (0, 0, 0.25)
            bpy.context.scene.collection.objects.link(lobj)
            lobj.modifiers.new("LotusGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.28, depth=0.20, location=(0, 0, 0.28))
            bpy.context.active_object.data.materials.append(m_pod)
            """,
            "Plan: CurvePrimitiveCircle -> CurveToPoints(Count=12), align flattened MeshUVSphere petal via AlignRotationToVector(Normal) in InstanceOnPoints + RealizeInstances.",
        ),
        (
            "f4_02_studded_shield",
            "Build a Nordic circular oak wooden shield studded with 16 iron perimeter rivets via Curve to Points and Instance on Points around a raised iron shield boss.",
            """\
            m_oak = pbr_mat("ShieldOak", (0.42, 0.24, 0.11), roughness=0.62)
            m_iron = pbr_mat("ShieldIron", (0.48, 0.50, 0.54), roughness=0.30, metal=0.94)
            ng = bpy.data.node_groups.new("ShieldStuds", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ring = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); ring.inputs["Radius"].default_value = 0.88
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 16
            stud = ng.nodes.new("GeometryNodeMeshIcoSphere"); stud.inputs["Radius"].default_value = 0.045; stud.inputs["Subdivisions"].default_value = 2
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_iron
            ng.links.new(ring.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(stud.outputs["Mesh"], inst.inputs["Instance"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
            ng.links.new(real.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            sobj = bpy.data.objects.new("ShieldStuds", bpy.data.meshes.new("StdM"))
            sobj.location = (0, 0, 0.12)
            bpy.context.scene.collection.objects.link(sobj)
            sobj.modifiers.new("StudsGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1.0, depth=0.09, location=(0, 0, 0.06))
            bpy.context.active_object.data.materials.append(m_oak)
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.24, location=(0, 0, 0.12))
            bpy.context.active_object.scale = (1.0, 1.0, 0.55)
            bpy.context.active_object.data.materials.append(m_iron)
            """,
            None,
        ),
        (
            "f4_03_pearl_necklace",
            "Build a graduated luster pearl strand choker instanced along a Curve Circle with size modulation around a dark velvet display plinth.",
            """\
            m_pearl = make_mat("NacrePearl", (0.94, 0.91, 0.88), rough=0.14)
            m_velvet = make_mat("VelvetStand", (0.10, 0.11, 0.16), rough=0.88)
            ng = bpy.data.node_groups.new("PearlStrand", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            cir = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); cir.inputs["Radius"].default_value = 0.72
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 32
            bead = ng.nodes.new("GeometryNodeMeshUVSphere"); bead.inputs["Segments"].default_value = 20; bead.inputs["Rings"].default_value = 12; bead.inputs["Radius"].default_value = 0.062
            rv = ng.nodes.new("FunctionNodeRandomValue"); rv.inputs["Min"].default_value = 0.82; rv.inputs["Max"].default_value = 1.25
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_pearl
            ng.links.new(cir.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(bead.outputs["Mesh"], inst.inputs["Instance"])
            ng.links.new(rv.outputs["Value"], inst.inputs["Scale"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
            ng.links.new(real.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            pobj = bpy.data.objects.new("PearlNecklace", bpy.data.meshes.new("PrlM"))
            pobj.location = (0, 0, 0.25)
            bpy.context.scene.collection.objects.link(pobj)
            pobj.modifiers.new("PearlGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.92, depth=0.18, location=(0, 0, 0.09))
            bpy.context.active_object.data.materials.append(m_velvet)
            """,
            None,
        ),
        (
            "f4_04_picket_fence",
            "Build a curved cedar wooden garden picket fence segment by instancing vertical fence pickets oriented along a Quadratic Bezier path.",
            """\
            m_cedar = pbr_mat("CedarWood", (0.58, 0.36, 0.19), roughness=0.64)
            m_grass = pbr_mat("LawnTurf", (0.16, 0.34, 0.14), roughness=0.86)
            ng = bpy.data.node_groups.new("CurvedFence", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            bz = ng.nodes.new("GeometryNodeCurveQuadraticBezier")
            bz.inputs["Resolution"].default_value = 32
            bz.inputs["Start"].default_value = (-1.5, -0.4, 0.0); bz.inputs["Middle"].default_value = (0.0, 0.65, 0.0); bz.inputs["End"].default_value = (1.5, -0.4, 0.0)
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 18
            cube = ng.nodes.new("GeometryNodeMeshCube"); cube.inputs["Size"].default_value = (0.04, 0.12, 0.95)
            up = ng.nodes.new("GeometryNodeTransform"); up.inputs["Translation"].default_value = (0.0, 0.0, 0.48)
            al = ng.nodes.new("FunctionNodeAlignRotationToVector")
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            rail_p = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); rail_p.inputs["Resolution"].default_value = 8; rail_p.inputs["Radius"].default_value = 0.032
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            rail_xf = ng.nodes.new("GeometryNodeTransform"); rail_xf.inputs["Translation"].default_value = (0.0, 0.0, 0.48)
            join = ng.nodes.new("GeometryNodeJoinGeometry")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_cedar
            ng.links.new(bz.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(cube.outputs["Mesh"], up.inputs["Geometry"])
            ng.links.new(c2p.outputs["Tangent"], al.inputs["Vector"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(up.outputs["Geometry"], inst.inputs["Instance"]); ng.links.new(al.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
            ng.links.new(bz.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(rail_p.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], rail_xf.inputs["Geometry"])
            ng.links.new(real.outputs["Geometry"], join.inputs["Geometry"]); ng.links.new(rail_xf.outputs["Geometry"], join.inputs["Geometry"])
            ng.links.new(join.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            fobj = bpy.data.objects.new("CedarFence", bpy.data.meshes.new("FncM"))
            bpy.context.scene.collection.objects.link(fobj)
            fobj.modifiers.new("FenceGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0.1, -0.05))
            bpy.context.active_object.scale = (3.6, 1.8, 0.10)
            bpy.context.active_object.data.materials.append(m_grass)
            """,
            "Plan: Sample CurveQuadraticBezier via CurveToPoints(Count=18), align picket MeshCube to Tangent with AlignRotationToVector, join swept rail, apply CedarWood.",
        ),
        (
            "f4_05_pebble_cairn",
            "Build a balanced stacked river-stone cairn tower using a vertical Mesh Line, tapering Index scale math, Random Value variation, and Instance on Points.",
            """\
            m_stone = make_mat("RiverSlate", (0.42, 0.44, 0.46), rough=0.68)
            m_sand = make_mat("PebbleSand", (0.72, 0.66, 0.54), rough=0.84)
            ng = bpy.data.node_groups.new("StoneCairn", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ln = ng.nodes.new("GeometryNodeMeshLine"); ln.inputs["Count"].default_value = 7; ln.inputs["Offset"].default_value = (0.0, 0.0, 0.22)
            idx = ng.nodes.new("GeometryNodeInputIndex")
            step = ng.nodes.new("ShaderNodeMath"); step.operation = "MULTIPLY"; step.inputs[1].default_value = -0.10
            scl_val = ng.nodes.new("ShaderNodeMath"); scl_val.operation = "ADD"; scl_val.inputs[0].default_value = 1.0
            pebble = ng.nodes.new("GeometryNodeMeshIcoSphere"); pebble.inputs["Radius"].default_value = 0.42; pebble.inputs["Subdivisions"].default_value = 3
            flat = ng.nodes.new("GeometryNodeTransform"); flat.inputs["Scale"].default_value = (1.15, 0.92, 0.34)
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_stone
            ng.links.new(idx.outputs["Index"], step.inputs[0]); ng.links.new(step.outputs["Value"], scl_val.inputs[1])
            ng.links.new(pebble.outputs["Mesh"], flat.inputs["Geometry"])
            ng.links.new(ln.outputs["Mesh"], inst.inputs["Points"]); ng.links.new(flat.outputs["Geometry"], inst.inputs["Instance"]); ng.links.new(scl_val.outputs["Value"], inst.inputs["Scale"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
            ng.links.new(real.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            cobj = bpy.data.objects.new("Cairn", bpy.data.meshes.new("CrnM"))
            cobj.location = (0, 0, 0.15)
            bpy.context.scene.collection.objects.link(cobj)
            cobj.modifiers.new("CairnGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=36, radius=0.95, depth=0.10, location=(0, 0, 0.05))
            bpy.context.active_object.data.materials.append(m_sand)
            """,
            None,
        ),
        (
            "f4_06_chandelier_ring",
            "Build a medieval wrought-iron ring chandelier with 8 taper candles instanced cleanly around the swept hoop circumference.",
            """\
            m_iron = pbr_mat("ChandelierIron", (0.12, 0.12, 0.14), roughness=0.40, metal=0.90)
            m_wax = pbr_mat("TaperWax", (0.94, 0.90, 0.78), roughness=0.42)
            ng = bpy.data.node_groups.new("ChandelierRing", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            cir = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); cir.inputs["Resolution"].default_value = 48; cir.inputs["Radius"].default_value = 0.95
            prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); prof.inputs["Resolution"].default_value = 12; prof.inputs["Radius"].default_value = 0.045
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm1 = ng.nodes.new("GeometryNodeSetMaterial"); sm1.inputs["Material"].default_value = m_iron
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 8
            candle = ng.nodes.new("GeometryNodeMeshCylinder"); candle.inputs["Vertices"].default_value = 16; candle.inputs["Radius"].default_value = 0.04; candle.inputs["Depth"].default_value = 0.34
            up = ng.nodes.new("GeometryNodeTransform"); up.inputs["Translation"].default_value = (0.0, 0.0, 0.19)
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm2 = ng.nodes.new("GeometryNodeSetMaterial"); sm2.inputs["Material"].default_value = m_wax
            join = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(cir.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(c2m.outputs["Mesh"], sm1.inputs["Geometry"])
            ng.links.new(cir.outputs["Curve"], c2p.inputs["Curve"]); ng.links.new(candle.outputs["Mesh"], up.inputs["Geometry"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(up.outputs["Geometry"], inst.inputs["Instance"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm2.inputs["Geometry"])
            ng.links.new(sm1.outputs["Geometry"], join.inputs["Geometry"]); ng.links.new(sm2.outputs["Geometry"], join.inputs["Geometry"])
            ng.links.new(join.outputs["Geometry"], go.inputs["Geometry"])
            chobj = bpy.data.objects.new("Chandelier", bpy.data.meshes.new("ChdM"))
            chobj.location = (0, 0, 1.35)
            bpy.context.scene.collection.objects.link(chobj)
            chobj.modifiers.new("ChandelierGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.025, depth=1.0, location=(0, 0, 1.85))
            bpy.context.active_object.data.materials.append(m_iron)
            """,
            None,
        ),
        (
            "f4_07_spoked_gear",
            "Build a precision machined brass spur gear wheel with 20 radial teeth instanced cleanly along Curve to Points and aligned to radial Normals.",
            """\
            m_brass = make_mat("MachinedBrass", (0.82, 0.62, 0.22), rough=0.22, metallic=0.96)
            m_steel = make_mat("AxleSteel", (0.45, 0.48, 0.52), rough=0.28, metallic=0.92)
            ng = bpy.data.node_groups.new("SpurGear", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            wheel = ng.nodes.new("GeometryNodeMeshCylinder"); wheel.inputs["Vertices"].default_value = 48; wheel.inputs["Radius"].default_value = 0.82; wheel.inputs["Depth"].default_value = 0.22
            cir = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); cir.inputs["Radius"].default_value = 0.85
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 20
            tooth = ng.nodes.new("GeometryNodeMeshCube"); tooth.inputs["Size"].default_value = (0.14, 0.11, 0.22)
            al = ng.nodes.new("FunctionNodeAlignRotationToVector")
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            join = ng.nodes.new("GeometryNodeJoinGeometry")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_brass
            ng.links.new(cir.outputs["Curve"], c2p.inputs["Curve"]); ng.links.new(c2p.outputs["Normal"], al.inputs["Vector"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(tooth.outputs["Mesh"], inst.inputs["Instance"]); ng.links.new(al.outputs["Rotation"], inst.inputs["Rotation"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
            ng.links.new(wheel.outputs["Mesh"], join.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], join.inputs["Geometry"])
            ng.links.new(join.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            gobj = bpy.data.objects.new("BrassGear", bpy.data.meshes.new("GearM"))
            gobj.location = (0, 0, 0.22)
            bpy.context.scene.collection.objects.link(gobj)
            gobj.modifiers.new("GearGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.18, depth=0.55, location=(0, 0, 0.27))
            bpy.context.active_object.data.materials.append(m_steel)
            """,
            None,
        ),
        (
            "f4_08_stepping_stones",
            "Build a curved slate stepping path across a warm gravel garden bed using Curve to Points, random scale variation, and Instance on Points.",
            """\
            m_slate = pbr_mat("GardenSlate", (0.30, 0.32, 0.36), roughness=0.64)
            m_gravel = pbr_mat("GravelBed", (0.64, 0.58, 0.48), roughness=0.88)
            ng = bpy.data.node_groups.new("SteppingPath", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            bz = ng.nodes.new("GeometryNodeCurveQuadraticBezier")
            bz.inputs["Start"].default_value = (-1.6, -0.7, 0.08); bz.inputs["Middle"].default_value = (0.1, 0.85, 0.08); bz.inputs["End"].default_value = (1.6, -0.5, 0.08)
            c2p = ng.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 10
            pad = ng.nodes.new("GeometryNodeMeshCylinder"); pad.inputs["Vertices"].default_value = 7; pad.inputs["Radius"].default_value = 0.22; pad.inputs["Depth"].default_value = 0.07
            rv = ng.nodes.new("FunctionNodeRandomValue"); rv.inputs["Min"].default_value = 0.78; rv.inputs["Max"].default_value = 1.18; rv.inputs["Seed"].default_value = 3
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_slate
            ng.links.new(bz.outputs["Curve"], c2p.inputs["Curve"])
            ng.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(pad.outputs["Mesh"], inst.inputs["Instance"]); ng.links.new(rv.outputs["Value"], inst.inputs["Scale"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
            ng.links.new(real.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            sobj = bpy.data.objects.new("SlatePath", bpy.data.meshes.new("PathM"))
            bpy.context.scene.collection.objects.link(sobj)
            sobj.modifiers.new("PathGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.0))
            bpy.context.active_object.scale = (3.8, 2.4, 0.08)
            bpy.context.active_object.data.materials.append(m_gravel)
            """,
            None,
        ),
        (
            "f4_09_cypress_grove",
            "Build a low-poly mountain cypress grove scattered across a hilly meadow tile via Distribute Points on Faces, Random Value height scaling, and Instance on Points.",
            """\
            m_foliage = make_mat("CypressGreen", (0.12, 0.34, 0.16), rough=0.68)
            m_meadow = make_mat("MeadowSoil", (0.28, 0.22, 0.14), rough=0.84)
            ng = bpy.data.node_groups.new("AlpineGrove", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid"); grid.inputs["Size X"].default_value = 3.2; grid.inputs["Size Y"].default_value = 3.2
            grid.inputs["Vertices X"].default_value = 24; grid.inputs["Vertices Y"].default_value = 24
            dist = ng.nodes.new("GeometryNodeDistributePointsOnFaces"); dist.inputs["Density"].default_value = 6.5; dist.inputs["Seed"].default_value = 9
            cone = ng.nodes.new("GeometryNodeMeshCone"); cone.inputs["Vertices"].default_value = 8; cone.inputs["Radius Bottom"].default_value = 0.22; cone.inputs["Radius Top"].default_value = 0.0; cone.inputs["Depth"].default_value = 0.85
            up = ng.nodes.new("GeometryNodeTransform"); up.inputs["Translation"].default_value = (0.0, 0.0, 0.42)
            rv = ng.nodes.new("FunctionNodeRandomValue"); rv.inputs["Min"].default_value = 0.65; rv.inputs["Max"].default_value = 1.45; rv.inputs["Seed"].default_value = 11
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm1 = ng.nodes.new("GeometryNodeSetMaterial"); sm1.inputs["Material"].default_value = m_foliage
            sm2 = ng.nodes.new("GeometryNodeSetMaterial"); sm2.inputs["Material"].default_value = m_meadow
            join = ng.nodes.new("GeometryNodeJoinGeometry")
            ng.links.new(grid.outputs["Mesh"], dist.inputs["Mesh"]); ng.links.new(cone.outputs["Mesh"], up.inputs["Geometry"])
            ng.links.new(dist.outputs["Points"], inst.inputs["Points"]); ng.links.new(up.outputs["Geometry"], inst.inputs["Instance"]); ng.links.new(rv.outputs["Value"], inst.inputs["Scale"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm1.inputs["Geometry"])
            ng.links.new(grid.outputs["Mesh"], sm2.inputs["Geometry"])
            ng.links.new(sm2.outputs["Geometry"], join.inputs["Geometry"]); ng.links.new(sm1.outputs["Geometry"], join.inputs["Geometry"])
            ng.links.new(join.outputs["Geometry"], go.inputs["Geometry"])
            gobj = bpy.data.objects.new("AlpineGrove", bpy.data.meshes.new("GrvM"))
            bpy.context.scene.collection.objects.link(gobj)
            gobj.modifiers.new("GroveGN", "NODES").node_group = ng
            """,
            "Plan: Create ground MeshGrid -> DistributePointsOnFaces (Density=6.5) -> InstanceOnPoints of cone trees scaled by RandomValue, RealizeInstances, SetMaterial per part, JoinGeometry.",
        ),
    ]
    for i, (rid, prompt, body, reason) in enumerate(specs):
        fn = f"build_{rid[6:]}" if i % 3 == 1 else None
        apply_obj = "sobj" if i == 1 and fn is None else None
        code = _wrap_script(i, _head(i), body, _tail(cam_loc=(4.5, -5.0, 3.6)), fn_name=fn, apply_mod_obj=apply_obj)
        items.append({"id": rid, "prompt": prompt, "code": code, "reasoning": reason})
    return items


# ==============================================================================
# Family 5 (8 recipes): Mesh Topology Ops (DualMesh, ExtrudeMesh, ScaleElements, ConvexHull)
# ==============================================================================
def _build_family5() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    specs = [
        (
            "f5_01_geodesic_dome",
            "Build a geodesic honeycomb observatory dome using GeometryNodeMeshIcoSphere, GeometryNodeDualMesh, GeometryNodeExtrudeMesh, and GeometryNodeScaleElements.",
            """\
            m_shell = make_mat("HoneycombShell", (0.86, 0.88, 0.90), rough=0.28, metallic=0.35)
            m_ring = make_mat("FoundationRing", (0.18, 0.20, 0.24), rough=0.52, metallic=0.82)
            ng = bpy.data.node_groups.new("GeodesicDome", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ico = ng.nodes.new("GeometryNodeMeshIcoSphere"); ico.inputs["Radius"].default_value = 1.05; ico.inputs["Subdivisions"].default_value = 3
            dual = ng.nodes.new("GeometryNodeDualMesh")
            ext = ng.nodes.new("GeometryNodeExtrudeMesh"); ext.inputs["Offset Scale"].default_value = 0.08; ext.inputs["Individual"].default_value = True
            sc = ng.nodes.new("GeometryNodeScaleElements"); sc.inputs["Scale"].default_value = 0.65
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_shell
            ng.links.new(ico.outputs["Mesh"], dual.inputs["Mesh"])
            ng.links.new(dual.outputs["Dual Mesh"], ext.inputs["Mesh"])
            ng.links.new(ext.outputs["Mesh"], sc.inputs["Geometry"]); ng.links.new(ext.outputs["Top"], sc.inputs["Selection"])
            ng.links.new(sc.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            dobj = bpy.data.objects.new("HoneycombDome", bpy.data.meshes.new("DomeM"))
            dobj.location = (0, 0, 0.95)
            bpy.context.scene.collection.objects.link(dobj)
            dobj.modifiers.new("DomeGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1.18, depth=0.16, location=(0, 0, 0.08))
            bpy.context.active_object.data.materials.append(m_ring)
            """,
            "Plan: IcoSphere(Subdivisions=3) -> DualMesh (outputs['Dual Mesh']) -> ExtrudeMesh(Individual=True) -> ScaleElements(Top selection, Scale=0.65).",
        ),
        (
            "f5_02_stepped_ziggurat",
            "Build a tiered Mesopotamian sandstone ziggurat monument using iterated face Extrude Mesh and Scale Elements inside a Geometry Node group.",
            """\
            m_sandstone = pbr_mat("SandstoneBlock", (0.76, 0.62, 0.42), roughness=0.78)
            m_gold = pbr_mat("ShrineGold", (0.86, 0.68, 0.22), roughness=0.22, metal=0.95)
            ng = bpy.data.node_groups.new("SteppedZiggurat", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid"); grid.inputs["Size X"].default_value = 2.4; grid.inputs["Size Y"].default_value = 2.4
            grid.inputs["Vertices X"].default_value = 8; grid.inputs["Vertices Y"].default_value = 8
            ex1 = ng.nodes.new("GeometryNodeExtrudeMesh"); ex1.inputs["Offset Scale"].default_value = 0.28; ex1.inputs["Individual"].default_value = False
            sc1 = ng.nodes.new("GeometryNodeScaleElements"); sc1.inputs["Scale"].default_value = 0.76
            ex2 = ng.nodes.new("GeometryNodeExtrudeMesh"); ex2.inputs["Offset Scale"].default_value = 0.28; ex2.inputs["Individual"].default_value = False
            sc2 = ng.nodes.new("GeometryNodeScaleElements"); sc2.inputs["Scale"].default_value = 0.72
            ex3 = ng.nodes.new("GeometryNodeExtrudeMesh"); ex3.inputs["Offset Scale"].default_value = 0.28; ex3.inputs["Individual"].default_value = False
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_sandstone
            ng.links.new(grid.outputs["Mesh"], ex1.inputs["Mesh"])
            ng.links.new(ex1.outputs["Mesh"], sc1.inputs["Geometry"]); ng.links.new(ex1.outputs["Top"], sc1.inputs["Selection"])
            ng.links.new(sc1.outputs["Geometry"], ex2.inputs["Mesh"]); ng.links.new(ex1.outputs["Top"], ex2.inputs["Selection"])
            ng.links.new(ex2.outputs["Mesh"], sc2.inputs["Geometry"]); ng.links.new(ex2.outputs["Top"], sc2.inputs["Selection"])
            ng.links.new(sc2.outputs["Geometry"], ex3.inputs["Mesh"]); ng.links.new(ex2.outputs["Top"], ex3.inputs["Selection"])
            ng.links.new(ex3.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            zobj = bpy.data.objects.new("Ziggurat", bpy.data.meshes.new("ZigM"))
            bpy.context.scene.collection.objects.link(zobj)
            zobj.modifiers.new("ZigguratGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=0.45, location=(0, 0, 1.05))
            bpy.context.active_object.data.materials.append(m_gold)
            """,
            None,
        ),
        (
            "f5_03_quartz_crystal",
            "Build a faceted translucent amethyst crystal prism cluster by extruding and tapering hexagonal crystal columns via Scale Elements on a dark basalt matrix.",
            """\
            m_crystal = make_mat("AmethystCrystal", (0.58, 0.26, 0.78), rough=0.10)
            m_basalt = make_mat("MatrixBasalt", (0.14, 0.14, 0.16), rough=0.82)
            ng = bpy.data.node_groups.new("QuartzCrystal", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            cyl = ng.nodes.new("GeometryNodeMeshCylinder"); cyl.inputs["Vertices"].default_value = 6; cyl.inputs["Side Segments"].default_value = 4; cyl.inputs["Radius"].default_value = 0.26; cyl.inputs["Depth"].default_value = 1.25
            ex = ng.nodes.new("GeometryNodeExtrudeMesh"); ex.inputs["Offset Scale"].default_value = 0.36; ex.inputs["Individual"].default_value = False
            sc = ng.nodes.new("GeometryNodeScaleElements"); sc.inputs["Scale"].default_value = 0.08
            pos = ng.nodes.new("GeometryNodeInputPosition")
            sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            gt = ng.nodes.new("ShaderNodeMath"); gt.operation = "GREATER_THAN"; gt.inputs[1].default_value = 0.60
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_crystal
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"]); ng.links.new(sep.outputs["Z"], gt.inputs[0])
            ng.links.new(cyl.outputs["Mesh"], ex.inputs["Mesh"]); ng.links.new(gt.outputs["Value"], ex.inputs["Selection"])
            ng.links.new(ex.outputs["Mesh"], sc.inputs["Geometry"]); ng.links.new(ex.outputs["Top"], sc.inputs["Selection"])
            ng.links.new(sc.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            qobj = bpy.data.objects.new("Crystal", bpy.data.meshes.new("CryM"))
            qobj.location = (0, 0, 0.68)
            bpy.context.scene.collection.objects.link(qobj)
            qobj.modifiers.new("CrystalGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.55, location=(0, 0, 0.22))
            bpy.context.active_object.scale = (1.3, 1.1, 0.45)
            bpy.context.active_object.data.materials.append(m_basalt)
            """,
            None,
        ),
        (
            "f5_04_lantern_cage",
            "Build a faceted dark bronze geometric cage dome via IcoSphere + Dual Mesh + Extrude Mesh around an illuminated warm amber globe core.",
            """\
            m_bronze = pbr_mat("CageBronze", (0.28, 0.20, 0.12), roughness=0.35, metal=0.90)
            m_amber = pbr_mat("AmberCore", (0.96, 0.56, 0.14), roughness=0.20)
            ng = bpy.data.node_groups.new("CageDome", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ico = ng.nodes.new("GeometryNodeMeshIcoSphere"); ico.inputs["Radius"].default_value = 0.72; ico.inputs["Subdivisions"].default_value = 2
            dual = ng.nodes.new("GeometryNodeDualMesh")
            ex = ng.nodes.new("GeometryNodeExtrudeMesh"); ex.inputs["Offset Scale"].default_value = 0.06; ex.inputs["Individual"].default_value = True
            sc = ng.nodes.new("GeometryNodeScaleElements"); sc.inputs["Scale"].default_value = 0.48
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_bronze
            ng.links.new(ico.outputs["Mesh"], dual.inputs["Mesh"]); ng.links.new(dual.outputs["Dual Mesh"], ex.inputs["Mesh"])
            ng.links.new(ex.outputs["Mesh"], sc.inputs["Geometry"]); ng.links.new(ex.outputs["Top"], sc.inputs["Selection"])
            ng.links.new(sc.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            cobj = bpy.data.objects.new("LanternCage", bpy.data.meshes.new("CageM"))
            cobj.location = (0, 0, 0.80)
            bpy.context.scene.collection.objects.link(cobj)
            cobj.modifiers.new("CageGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.45, location=(0, 0, 0.80))
            bpy.context.active_object.data.materials.append(m_amber)
            """,
            None,
        ),
        (
            "f5_05_exhibition_plinth",
            "Build a tiered museum exhibition pedestal plinth with chamfered crown moldings via Extrude Mesh and Scale Elements topped by a bronze emblem.",
            """\
            m_plinth = make_mat("GalleryStone", (0.82, 0.80, 0.76), rough=0.46)
            m_emblem = make_mat("EmblemBronze", (0.64, 0.44, 0.18), rough=0.28, metallic=0.92)
            ng = bpy.data.node_groups.new("GalleryPlinth", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            box = ng.nodes.new("GeometryNodeMeshCube"); box.inputs["Size"].default_value = (0.9, 0.9, 1.1); box.inputs["Vertices X"].default_value = 4; box.inputs["Vertices Y"].default_value = 4; box.inputs["Vertices Z"].default_value = 4
            pos = ng.nodes.new("GeometryNodeInputPosition"); sep = ng.nodes.new("ShaderNodeSeparateXYZ")
            gt = ng.nodes.new("ShaderNodeMath"); gt.operation = "GREATER_THAN"; gt.inputs[1].default_value = 0.50
            ex1 = ng.nodes.new("GeometryNodeExtrudeMesh"); ex1.inputs["Offset Scale"].default_value = 0.12; ex1.inputs["Individual"].default_value = False
            sc1 = ng.nodes.new("GeometryNodeScaleElements"); sc1.inputs["Scale"].default_value = 1.18
            ex2 = ng.nodes.new("GeometryNodeExtrudeMesh"); ex2.inputs["Offset Scale"].default_value = 0.08; ex2.inputs["Individual"].default_value = False
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_plinth
            ng.links.new(pos.outputs["Position"], sep.inputs["Vector"]); ng.links.new(sep.outputs["Z"], gt.inputs[0])
            ng.links.new(box.outputs["Mesh"], ex1.inputs["Mesh"]); ng.links.new(gt.outputs["Value"], ex1.inputs["Selection"])
            ng.links.new(ex1.outputs["Mesh"], sc1.inputs["Geometry"]); ng.links.new(ex1.outputs["Top"], sc1.inputs["Selection"])
            ng.links.new(sc1.outputs["Geometry"], ex2.inputs["Mesh"]); ng.links.new(ex1.outputs["Top"], ex2.inputs["Selection"])
            ng.links.new(ex2.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            pobj = bpy.data.objects.new("Plinth", bpy.data.meshes.new("PlnM"))
            pobj.location = (0, 0, 0.55)
            bpy.context.scene.collection.objects.link(pobj)
            pobj.modifiers.new("PlinthGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.28, location=(0, 0, 1.56))
            bpy.context.active_object.data.materials.append(m_emblem)
            """,
            None,
        ),
        (
            "f5_06_waffle_skylight",
            "Build a coffered architectural skylight grid ceiling panel using subdivided Mesh Grid + Extrude Mesh (Individual) + Scale Elements + negative coffer extrusion.",
            """\
            m_oak = pbr_mat("TimberCoffer", (0.48, 0.30, 0.15), roughness=0.52)
            m_frame = pbr_mat("SkylightSteel", (0.16, 0.18, 0.20), roughness=0.35, metal=0.90)
            ng = bpy.data.node_groups.new("WaffleGrid", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid")
            grid.inputs["Size X"].default_value = 2.6; grid.inputs["Size Y"].default_value = 2.6
            grid.inputs["Vertices X"].default_value = 9; grid.inputs["Vertices Y"].default_value = 9
            ex1 = ng.nodes.new("GeometryNodeExtrudeMesh"); ex1.inputs["Offset Scale"].default_value = 0.04; ex1.inputs["Individual"].default_value = True
            sc = ng.nodes.new("GeometryNodeScaleElements"); sc.inputs["Scale"].default_value = 0.78
            ex2 = ng.nodes.new("GeometryNodeExtrudeMesh"); ex2.inputs["Offset Scale"].default_value = -0.14; ex2.inputs["Individual"].default_value = True
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_oak
            ng.links.new(grid.outputs["Mesh"], ex1.inputs["Mesh"])
            ng.links.new(ex1.outputs["Mesh"], sc.inputs["Geometry"]); ng.links.new(ex1.outputs["Top"], sc.inputs["Selection"])
            ng.links.new(sc.outputs["Geometry"], ex2.inputs["Mesh"]); ng.links.new(ex1.outputs["Top"], ex2.inputs["Selection"])
            ng.links.new(ex2.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            wobj = bpy.data.objects.new("WafflePanel", bpy.data.meshes.new("WafM"))
            wobj.location = (0, 0, 0.25)
            bpy.context.scene.collection.objects.link(wobj)
            wobj.modifiers.new("WaffleGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.06))
            bpy.context.active_object.scale = (2.8, 2.8, 0.10)
            bpy.context.active_object.data.materials.append(m_frame)
            """,
            "Plan: MeshGrid (9x9) -> ExtrudeMesh(Individual=True) -> ScaleElements(Top, 0.78) -> ExtrudeMesh(Top, -0.14) to recess coffers cleanly.",
        ),
        (
            "f5_07_honeycomb_wall",
            "Build a hexagonal extruded studio acoustic wall relief sphere cluster using Dual Mesh and per-cell randomized Extrude Mesh depth.",
            """\
            m_felt = make_mat("CharcoalFelt", (0.20, 0.22, 0.26), rough=0.86)
            m_walnut = make_mat("StudioWalnut", (0.34, 0.18, 0.09), rough=0.45)
            ng = bpy.data.node_groups.new("HexAcoustic", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ico = ng.nodes.new("GeometryNodeMeshIcoSphere"); ico.inputs["Radius"].default_value = 0.85; ico.inputs["Subdivisions"].default_value = 3
            dual = ng.nodes.new("GeometryNodeDualMesh")
            rv = ng.nodes.new("FunctionNodeRandomValue"); rv.inputs["Min"].default_value = 0.02; rv.inputs["Max"].default_value = 0.22; rv.inputs["Seed"].default_value = 5
            ex = ng.nodes.new("GeometryNodeExtrudeMesh"); ex.inputs["Individual"].default_value = True
            sc = ng.nodes.new("GeometryNodeScaleElements"); sc.inputs["Scale"].default_value = 0.82
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_felt
            ng.links.new(ico.outputs["Mesh"], dual.inputs["Mesh"])
            ng.links.new(dual.outputs["Dual Mesh"], ex.inputs["Mesh"]); ng.links.new(rv.outputs["Value"], ex.inputs["Offset Scale"])
            ng.links.new(ex.outputs["Mesh"], sc.inputs["Geometry"]); ng.links.new(ex.outputs["Top"], sc.inputs["Selection"])
            ng.links.new(sc.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            hobj = bpy.data.objects.new("AcousticSphere", bpy.data.meshes.new("HexM"))
            hobj.location = (0, 0, 1.0)
            bpy.context.scene.collection.objects.link(hobj)
            hobj.modifiers.new("HexGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.95, depth=0.14, location=(0, 0, 0.07))
            bpy.context.active_object.data.materials.append(m_walnut)
            """,
            None,
        ),
        (
            "f5_08_faceted_monolith",
            "Build a faceted black obsidian monument obelisk enveloped cleanly via GeometryNodeConvexHull from displaced icosphere vertices over a bronze pedestal.",
            """\
            m_obsidian = pbr_mat("ObsidianStone", (0.05, 0.06, 0.08), roughness=0.12)
            m_bronze = pbr_mat("MonumentBronze", (0.64, 0.46, 0.20), roughness=0.32, metal=0.92)
            ng = bpy.data.node_groups.new("FacetedMonolith", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ico = ng.nodes.new("GeometryNodeMeshIcoSphere"); ico.inputs["Radius"].default_value = 0.65; ico.inputs["Subdivisions"].default_value = 3
            xf = ng.nodes.new("GeometryNodeTransform"); xf.inputs["Scale"].default_value = (0.75, 0.65, 2.2)
            pos = ng.nodes.new("GeometryNodeInputPosition")
            nz = ng.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 2.5
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            sp = ng.nodes.new("GeometryNodeSetPosition")
            hull = ng.nodes.new("GeometryNodeConvexHull")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_obsidian
            ng.links.new(ico.outputs["Mesh"], xf.inputs["Geometry"])
            ng.links.new(pos.outputs["Position"], nz.inputs["Vector"])
            ng.links.new(pos.outputs["Position"], vsc.inputs["Vector"]); ng.links.new(nz.outputs["Factor"], vsc.inputs["Scale"])
            ng.links.new(xf.outputs["Geometry"], sp.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], sp.inputs["Offset"])
            ng.links.new(sp.outputs["Geometry"], hull.inputs["Geometry"]); ng.links.new(hull.outputs["Convex Hull"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            mobj = bpy.data.objects.new("Monolith", bpy.data.meshes.new("MonM"))
            mobj.location = (0, 0, 1.15)
            bpy.context.scene.collection.objects.link(mobj)
            mobj.modifiers.new("MonolithGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.12))
            bpy.context.active_object.scale = (1.1, 1.1, 0.24)
            bpy.context.active_object.data.materials.append(m_bronze)
            """,
            "Plan: Stretch IcoSphere -> displace slightly with TexNoise.outputs['Factor'] -> ConvexHull (outputs['Convex Hull']) -> SetMaterial ObsidianStone.",
        ),
    ]
    for i, (rid, prompt, body, reason) in enumerate(specs):
        fn = f"build_{rid[6:]}" if i % 4 == 2 else None
        apply_obj = "dobj" if i == 0 and fn is None else None
        code = _wrap_script(i, _head(i), body, _tail(cam_loc=(4.5, -5.0, 3.2)), fn_name=fn, apply_mod_obj=apply_obj)
        items.append({"id": rid, "prompt": prompt, "code": code, "reasoning": reason})
    return items


# ==============================================================================
# Family 6 (7 recipes): Multi-NodeGroup & Multi-Part Procedural Assemblies
# ==============================================================================
def _build_family6() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    specs = [
        (
            "f6_01_potted_fern",
            "Build a decorative potted ornamental fern assembly with a terracotta planter node group and an arched leafy frond crown node group.",
            """\
            m_pot = make_mat("TerracottaPot", (0.64, 0.26, 0.14), rough=0.68)
            m_leaf = make_mat("FernGreen", (0.14, 0.48, 0.18), rough=0.45)

            ng1 = bpy.data.node_groups.new("PlanterPot", "GeometryNodeTree")
            ng1.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g1 = ng1.nodes.new("NodeGroupOutput")
            cn = ng1.nodes.new("GeometryNodeMeshCone"); cn.inputs["Vertices"].default_value = 36; cn.inputs["Radius Top"].default_value = 0.56; cn.inputs["Radius Bottom"].default_value = 0.36; cn.inputs["Depth"].default_value = 0.68
            sm1 = ng1.nodes.new("GeometryNodeSetMaterial"); sm1.inputs["Material"].default_value = m_pot
            ng1.links.new(cn.outputs["Mesh"], sm1.inputs["Geometry"]); ng1.links.new(sm1.outputs["Geometry"], g1.inputs["Geometry"])

            ng2 = bpy.data.node_groups.new("FernCrown", "GeometryNodeTree")
            ng2.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g2 = ng2.nodes.new("NodeGroupOutput")
            rg = ng2.nodes.new("GeometryNodeCurvePrimitiveCircle"); rg.inputs["Radius"].default_value = 0.18
            c2p = ng2.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 14
            bz = ng2.nodes.new("GeometryNodeCurveQuadraticBezier"); bz.inputs["Start"].default_value = (0, 0, 0); bz.inputs["Middle"].default_value = (0.45, 0, 0.68); bz.inputs["End"].default_value = (0.92, 0, 0.15)
            pc = ng2.nodes.new("GeometryNodeCurvePrimitiveCircle"); pc.inputs["Resolution"].default_value = 8; pc.inputs["Radius"].default_value = 0.045
            c2m = ng2.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            al = ng2.nodes.new("FunctionNodeAlignRotationToVector")
            inst = ng2.nodes.new("GeometryNodeInstanceOnPoints"); real = ng2.nodes.new("GeometryNodeRealizeInstances")
            sm2 = ng2.nodes.new("GeometryNodeSetMaterial"); sm2.inputs["Material"].default_value = m_leaf
            ng2.links.new(rg.outputs["Curve"], c2p.inputs["Curve"]); ng2.links.new(bz.outputs["Curve"], c2m.inputs["Curve"]); ng2.links.new(pc.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng2.links.new(c2p.outputs["Normal"], al.inputs["Vector"]); ng2.links.new(c2p.outputs["Points"], inst.inputs["Points"])
            ng2.links.new(c2m.outputs["Mesh"], inst.inputs["Instance"]); ng2.links.new(al.outputs["Rotation"], inst.inputs["Rotation"])
            ng2.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng2.links.new(real.outputs["Geometry"], sm2.inputs["Geometry"]); ng2.links.new(sm2.outputs["Geometry"], g2.inputs["Geometry"])

            o1 = bpy.data.objects.new("Planter", bpy.data.meshes.new("PotM")); o1.location = (0, 0, 0.34); bpy.context.scene.collection.objects.link(o1); o1.modifiers.new("PotGN", "NODES").node_group = ng1
            o2 = bpy.data.objects.new("Fronds", bpy.data.meshes.new("FrnM")); o2.location = (0, 0, 0.62); bpy.context.scene.collection.objects.link(o2); o2.modifiers.new("FrnGN", "NODES").node_group = ng2
            """,
            "Plan: Build PlanterPot (MeshCone) + FernCrown (CurveToPoints -> InstanceOnPoints of CurveQuadraticBezier fronds).",
        ),
        (
            "f6_02_trophy_cup",
            "Build a gilded championship trophy cup featuring a flared metallic cup GeometryNodeTree and twin-handled scroll curves on a marble plinth.",
            """\
            m_gold = pbr_mat("TrophyGold", (0.90, 0.72, 0.22), roughness=0.14, metal=0.98)
            m_marble = pbr_mat("BlackMarble", (0.09, 0.10, 0.11), roughness=0.22)

            ng1 = bpy.data.node_groups.new("TrophyVessel", "GeometryNodeTree")
            ng1.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g1 = ng1.nodes.new("NodeGroupOutput")
            cn = ng1.nodes.new("GeometryNodeMeshCone"); cn.inputs["Vertices"].default_value = 48; cn.inputs["Side Segments"].default_value = 16
            cn.inputs["Radius Bottom"].default_value = 0.14; cn.inputs["Radius Top"].default_value = 0.46; cn.inputs["Depth"].default_value = 0.85
            sm1 = ng1.nodes.new("GeometryNodeSetMaterial"); sm1.inputs["Material"].default_value = m_gold
            ng1.links.new(cn.outputs["Mesh"], sm1.inputs["Geometry"]); ng1.links.new(sm1.outputs["Geometry"], g1.inputs["Geometry"])

            ng2 = bpy.data.node_groups.new("TrophyHandles", "GeometryNodeTree")
            ng2.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g2 = ng2.nodes.new("NodeGroupOutput")
            bz = ng2.nodes.new("GeometryNodeCurveQuadraticBezier"); bz.inputs["Start"].default_value = (0.36, 0, 1.02); bz.inputs["Middle"].default_value = (0.72, 0, 0.82); bz.inputs["End"].default_value = (0.24, 0, 0.52)
            pr = ng2.nodes.new("GeometryNodeCurvePrimitiveCircle"); pr.inputs["Radius"].default_value = 0.032
            c2m = ng2.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            xf = ng2.nodes.new("GeometryNodeTransform"); xf.inputs["Rotation"].default_value = (0, 0, 3.14159)
            jn = ng2.nodes.new("GeometryNodeJoinGeometry"); sm2 = ng2.nodes.new("GeometryNodeSetMaterial"); sm2.inputs["Material"].default_value = m_gold
            ng2.links.new(bz.outputs["Curve"], c2m.inputs["Curve"]); ng2.links.new(pr.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng2.links.new(c2m.outputs["Mesh"], xf.inputs["Geometry"]); ng2.links.new(c2m.outputs["Mesh"], jn.inputs["Geometry"]); ng2.links.new(xf.outputs["Geometry"], jn.inputs["Geometry"])
            ng2.links.new(jn.outputs["Geometry"], sm2.inputs["Geometry"]); ng2.links.new(sm2.outputs["Geometry"], g2.inputs["Geometry"])

            o1 = bpy.data.objects.new("Cup", bpy.data.meshes.new("CupM")); o1.location = (0, 0, 0.78); bpy.context.scene.collection.objects.link(o1); o1.modifiers.new("CupGN", "NODES").node_group = ng1
            o2 = bpy.data.objects.new("Handles", bpy.data.meshes.new("HndM")); bpy.context.scene.collection.objects.link(o2); o2.modifiers.new("HndGN", "NODES").node_group = ng2
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.16)); bpy.context.active_object.scale = (0.75, 0.75, 0.32); bpy.context.active_object.data.materials.append(m_marble)
            """,
            None,
        ),
        (
            "f6_03_lighthouse_tower",
            "Build a coastal stone lighthouse tower with a tapered fluted masonry shaft and a geodesic glass lantern cupola mounted above a granite parapet.",
            """\
            m_white = make_mat("LighthouseStucco", (0.90, 0.89, 0.86), rough=0.68)
            m_lantern = make_mat("CupolaBronze", (0.20, 0.22, 0.24), rough=0.35, metallic=0.88)

            ng1 = bpy.data.node_groups.new("LighthouseShaft", "GeometryNodeTree")
            ng1.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g1 = ng1.nodes.new("NodeGroupOutput")
            cone = ng1.nodes.new("GeometryNodeMeshCone"); cone.inputs["Vertices"].default_value = 32; cone.inputs["Side Segments"].default_value = 24
            cone.inputs["Radius Bottom"].default_value = 0.62; cone.inputs["Radius Top"].default_value = 0.38; cone.inputs["Depth"].default_value = 2.1
            sm1 = ng1.nodes.new("GeometryNodeSetMaterial"); sm1.inputs["Material"].default_value = m_white
            ng1.links.new(cone.outputs["Mesh"], sm1.inputs["Geometry"]); ng1.links.new(sm1.outputs["Geometry"], g1.inputs["Geometry"])

            ng2 = bpy.data.node_groups.new("LanternCupola", "GeometryNodeTree")
            ng2.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g2 = ng2.nodes.new("NodeGroupOutput")
            ico = ng2.nodes.new("GeometryNodeMeshIcoSphere"); ico.inputs["Radius"].default_value = 0.42; ico.inputs["Subdivisions"].default_value = 2
            dual = ng2.nodes.new("GeometryNodeDualMesh")
            ex = ng2.nodes.new("GeometryNodeExtrudeMesh"); ex.inputs["Offset Scale"].default_value = 0.04; ex.inputs["Individual"].default_value = True
            sm2 = ng2.nodes.new("GeometryNodeSetMaterial"); sm2.inputs["Material"].default_value = m_lantern
            ng2.links.new(ico.outputs["Mesh"], dual.inputs["Mesh"]); ng2.links.new(dual.outputs["Dual Mesh"], ex.inputs["Mesh"])
            ng2.links.new(ex.outputs["Mesh"], sm2.inputs["Geometry"]); ng2.links.new(sm2.outputs["Geometry"], g2.inputs["Geometry"])

            t = bpy.data.objects.new("Tower", bpy.data.meshes.new("TwrM")); t.location = (0, 0, 1.05); bpy.context.scene.collection.objects.link(t); t.modifiers.new("TwrGN", "NODES").node_group = ng1
            c = bpy.data.objects.new("Cupola", bpy.data.meshes.new("CupM")); c.location = (0, 0, 2.38); bpy.context.scene.collection.objects.link(c); c.modifiers.new("CupGN", "NODES").node_group = ng2
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.46, depth=0.14, location=(0, 0, 2.08)); bpy.context.active_object.data.materials.append(m_lantern)
            """,
            None,
        ),
        (
            "f6_04_garden_fountain",
            "Build a sculptural carved stone fountain basin with 8 arched water streams instanced symmetrically from the central nozzle into the stone pool.",
            """\
            m_stone = pbr_mat("FountainStone", (0.70, 0.68, 0.64), roughness=0.65)
            m_water = pbr_mat("StreamWater", (0.52, 0.80, 0.92), roughness=0.06)

            ng1 = bpy.data.node_groups.new("FountainPool", "GeometryNodeTree")
            ng1.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g1 = ng1.nodes.new("NodeGroupOutput")
            cyl = ng1.nodes.new("GeometryNodeMeshCylinder"); cyl.inputs["Vertices"].default_value = 48; cyl.inputs["Radius"].default_value = 1.25; cyl.inputs["Depth"].default_value = 0.28
            ex = ng1.nodes.new("GeometryNodeExtrudeMesh"); ex.inputs["Offset Scale"].default_value = -0.14
            sm1 = ng1.nodes.new("GeometryNodeSetMaterial"); sm1.inputs["Material"].default_value = m_stone
            ng1.links.new(cyl.outputs["Mesh"], ex.inputs["Mesh"]); ng1.links.new(ex.outputs["Mesh"], sm1.inputs["Geometry"]); ng1.links.new(sm1.outputs["Geometry"], g1.inputs["Geometry"])

            ng2 = bpy.data.node_groups.new("WaterJets", "GeometryNodeTree")
            ng2.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g2 = ng2.nodes.new("NodeGroupOutput")
            ring = ng2.nodes.new("GeometryNodeCurvePrimitiveCircle"); ring.inputs["Radius"].default_value = 0.14
            c2p = ng2.nodes.new("GeometryNodeCurveToPoints"); c2p.inputs["Count"].default_value = 8
            bz = ng2.nodes.new("GeometryNodeCurveQuadraticBezier"); bz.inputs["Start"].default_value = (0, 0, 0); bz.inputs["Middle"].default_value = (0.45, 0, 0.55); bz.inputs["End"].default_value = (0.85, 0, -0.75)
            pr = ng2.nodes.new("GeometryNodeCurvePrimitiveCircle"); pr.inputs["Radius"].default_value = 0.024
            c2m = ng2.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            al = ng2.nodes.new("FunctionNodeAlignRotationToVector")
            inst = ng2.nodes.new("GeometryNodeInstanceOnPoints"); real = ng2.nodes.new("GeometryNodeRealizeInstances")
            sm2 = ng2.nodes.new("GeometryNodeSetMaterial"); sm2.inputs["Material"].default_value = m_water
            ng2.links.new(ring.outputs["Curve"], c2p.inputs["Curve"]); ng2.links.new(bz.outputs["Curve"], c2m.inputs["Curve"]); ng2.links.new(pr.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng2.links.new(c2p.outputs["Normal"], al.inputs["Vector"]); ng2.links.new(c2p.outputs["Points"], inst.inputs["Points"]); ng2.links.new(c2m.outputs["Mesh"], inst.inputs["Instance"])
            ng2.links.new(al.outputs["Rotation"], inst.inputs["Rotation"]); ng2.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
            ng2.links.new(real.outputs["Geometry"], sm2.inputs["Geometry"]); ng2.links.new(sm2.outputs["Geometry"], g2.inputs["Geometry"])

            o1 = bpy.data.objects.new("Pool", bpy.data.meshes.new("PoolM")); o1.location = (0, 0, 0.14); bpy.context.scene.collection.objects.link(o1); o1.modifiers.new("PoolGN", "NODES").node_group = ng1
            o2 = bpy.data.objects.new("Jets", bpy.data.meshes.new("JetM")); o2.location = (0, 0, 0.92); bpy.context.scene.collection.objects.link(o2); o2.modifiers.new("JetsGN", "NODES").node_group = ng2
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.18, depth=0.95, location=(0, 0, 0.48)); bpy.context.active_object.data.materials.append(m_stone)
            """,
            None,
        ),
        (
            "f6_05_hourglass_stand",
            "Build an ornate brass hourglass frame with a pinched glass chamber node group and spiral brass frame.",
            """\
            m_glass = make_mat("HourglassGlass", (0.80, 0.90, 0.96), rough=0.06)
            m_brass = make_mat("FrameBrass", (0.80, 0.60, 0.22), rough=0.22, metallic=0.95)

            ng1 = bpy.data.node_groups.new("PinchedBulb", "GeometryNodeTree")
            ng1.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g1 = ng1.nodes.new("NodeGroupOutput")
            c1 = ng1.nodes.new("GeometryNodeMeshCone"); c1.inputs["Vertices"].default_value = 32; c1.inputs["Radius Bottom"].default_value = 0.36; c1.inputs["Radius Top"].default_value = 0.05; c1.inputs["Depth"].default_value = 0.65
            c2 = ng1.nodes.new("GeometryNodeMeshCone"); c2.inputs["Vertices"].default_value = 32; c2.inputs["Radius Bottom"].default_value = 0.05; c2.inputs["Radius Top"].default_value = 0.36; c2.inputs["Depth"].default_value = 0.65
            xf = ng1.nodes.new("GeometryNodeTransform"); xf.inputs["Translation"].default_value = (0, 0, 0.65)
            jn = ng1.nodes.new("GeometryNodeJoinGeometry")
            sm1 = ng1.nodes.new("GeometryNodeSetMaterial"); sm1.inputs["Material"].default_value = m_glass
            ng1.links.new(c2.outputs["Mesh"], xf.inputs["Geometry"])
            ng1.links.new(c1.outputs["Mesh"], jn.inputs["Geometry"]); ng1.links.new(xf.outputs["Geometry"], jn.inputs["Geometry"])
            ng1.links.new(jn.outputs["Geometry"], sm1.inputs["Geometry"]); ng1.links.new(sm1.outputs["Geometry"], g1.inputs["Geometry"])

            ng2 = bpy.data.node_groups.new("SpiralFrame", "GeometryNodeTree")
            ng2.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g2 = ng2.nodes.new("NodeGroupOutput")
            sr = ng2.nodes.new("GeometryNodeCurveSpiral"); sr.inputs["Rotations"].default_value = 2.0; sr.inputs["Start Radius"].default_value = 0.54; sr.inputs["End Radius"].default_value = 0.54; sr.inputs["Height"].default_value = 1.4
            pr = ng2.nodes.new("GeometryNodeCurvePrimitiveCircle"); pr.inputs["Radius"].default_value = 0.032
            c2m = ng2.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm2 = ng2.nodes.new("GeometryNodeSetMaterial"); sm2.inputs["Material"].default_value = m_brass
            ng2.links.new(sr.outputs["Curve"], c2m.inputs["Curve"]); ng2.links.new(pr.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng2.links.new(c2m.outputs["Mesh"], sm2.inputs["Geometry"]); ng2.links.new(sm2.outputs["Geometry"], g2.inputs["Geometry"])

            o1 = bpy.data.objects.new("Bulb", bpy.data.meshes.new("GlM")); o1.location = (0, 0, 0.45); bpy.context.scene.collection.objects.link(o1); o1.modifiers.new("BulbGN", "NODES").node_group = ng1
            o2 = bpy.data.objects.new("Frame", bpy.data.meshes.new("FrM")); o2.location = (0, 0, 0.08); bpy.context.scene.collection.objects.link(o2); o2.modifiers.new("FrameGN", "NODES").node_group = ng2
            for zz in (0.08, 1.48):
                bpy.ops.mesh.primitive_cylinder_add(vertices=28, radius=0.62, depth=0.10, location=(0, 0, zz)); bpy.context.active_object.data.materials.append(m_brass)
            """,
            None,
        ),
        (
            "f6_06_forest_mushroom",
            "Build an organic woodland toadstool mushroom featuring a ribbed domed umbrella cap node group and a curved Bezier stalk node group.",
            """\
            m_cap = pbr_mat("ToadstoolCap", (0.76, 0.16, 0.12), roughness=0.38)
            m_stalk = pbr_mat("CreamStalk", (0.88, 0.84, 0.74), roughness=0.58)

            ng1 = bpy.data.node_groups.new("MushroomCap", "GeometryNodeTree")
            ng1.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g1 = ng1.nodes.new("NodeGroupOutput")
            sp = ng1.nodes.new("GeometryNodeMeshUVSphere"); sp.inputs["Segments"].default_value = 48; sp.inputs["Rings"].default_value = 24; sp.inputs["Radius"].default_value = 0.65
            pos = ng1.nodes.new("GeometryNodeInputPosition"); sep = ng1.nodes.new("ShaderNodeSeparateXYZ")
            ang = ng1.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
            mf = ng1.nodes.new("ShaderNodeMath"); mf.operation = "MULTIPLY"; mf.inputs[1].default_value = 20.0
            sn = ng1.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
            amp = ng1.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.04
            vs = ng1.nodes.new("ShaderNodeVectorMath"); vs.operation = "SCALE"
            set_p = ng1.nodes.new("GeometryNodeSetPosition"); sm1 = ng1.nodes.new("GeometryNodeSetMaterial"); sm1.inputs["Material"].default_value = m_cap
            ng1.links.new(pos.outputs["Position"], sep.inputs["Vector"]); ng1.links.new(sep.outputs["Y"], ang.inputs[0]); ng1.links.new(sep.outputs["X"], ang.inputs[1])
            ng1.links.new(ang.outputs["Value"], mf.inputs[0]); ng1.links.new(mf.outputs["Value"], sn.inputs[0]); ng1.links.new(sn.outputs["Value"], amp.inputs[0])
            ng1.links.new(pos.outputs["Position"], vs.inputs["Vector"]); ng1.links.new(amp.outputs["Value"], vs.inputs["Scale"])
            ng1.links.new(sp.outputs["Mesh"], set_p.inputs["Geometry"]); ng1.links.new(vs.outputs["Vector"], set_p.inputs["Offset"])
            ng1.links.new(set_p.outputs["Geometry"], sm1.inputs["Geometry"]); ng1.links.new(sm1.outputs["Geometry"], g1.inputs["Geometry"])

            ng2 = bpy.data.node_groups.new("CurvedStalk", "GeometryNodeTree")
            ng2.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g2 = ng2.nodes.new("NodeGroupOutput")
            bz = ng2.nodes.new("GeometryNodeCurveQuadraticBezier"); bz.inputs["Start"].default_value = (0, 0, 0); bz.inputs["Middle"].default_value = (0.14, 0, 0.52); bz.inputs["End"].default_value = (0, 0, 1.05)
            pr = ng2.nodes.new("GeometryNodeCurvePrimitiveCircle"); pr.inputs["Radius"].default_value = 0.14
            c2m = ng2.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            sm2 = ng2.nodes.new("GeometryNodeSetMaterial"); sm2.inputs["Material"].default_value = m_stalk
            ng2.links.new(bz.outputs["Curve"], c2m.inputs["Curve"]); ng2.links.new(pr.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng2.links.new(c2m.outputs["Mesh"], sm2.inputs["Geometry"]); ng2.links.new(sm2.outputs["Geometry"], g2.inputs["Geometry"])

            o1 = bpy.data.objects.new("Cap", bpy.data.meshes.new("CapM")); o1.location = (0, 0, 1.1); o1.scale = (1, 1, 0.52); bpy.context.scene.collection.objects.link(o1); o1.modifiers.new("CapGN", "NODES").node_group = ng1
            o2 = bpy.data.objects.new("Stalk", bpy.data.meshes.new("StkM")); bpy.context.scene.collection.objects.link(o2); o2.modifiers.new("StalkGN", "NODES").node_group = ng2
            """,
            None,
        ),
        (
            "f6_07_street_bollard_lamp",
            "Build a vintage cast-iron garden street lamp combining a fluted tapered shaft GeometryNodeTree and an extruded hexagonal lantern head GeometryNodeTree.",
            """\
            m_iron = make_mat("StreetIron", (0.12, 0.14, 0.15), rough=0.38, metallic=0.88)
            m_glow = make_mat("LampAmber", (0.98, 0.80, 0.42), rough=0.16)

            ng1 = bpy.data.node_groups.new("LampPost", "GeometryNodeTree")
            ng1.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g1 = ng1.nodes.new("NodeGroupOutput")
            cone = ng1.nodes.new("GeometryNodeMeshCone"); cone.inputs["Vertices"].default_value = 32; cone.inputs["Radius Bottom"].default_value = 0.24; cone.inputs["Radius Top"].default_value = 0.11; cone.inputs["Depth"].default_value = 1.9
            sm1 = ng1.nodes.new("GeometryNodeSetMaterial"); sm1.inputs["Material"].default_value = m_iron
            ng1.links.new(cone.outputs["Mesh"], sm1.inputs["Geometry"]); ng1.links.new(sm1.outputs["Geometry"], g1.inputs["Geometry"])

            ng2 = bpy.data.node_groups.new("LampHead", "GeometryNodeTree")
            ng2.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            g2 = ng2.nodes.new("NodeGroupOutput")
            hx = ng2.nodes.new("GeometryNodeMeshCylinder"); hx.inputs["Vertices"].default_value = 6; hx.inputs["Radius"].default_value = 0.28; hx.inputs["Depth"].default_value = 0.46
            ex = ng2.nodes.new("GeometryNodeExtrudeMesh"); ex.inputs["Offset Scale"].default_value = 0.18
            sc = ng2.nodes.new("GeometryNodeScaleElements"); sc.inputs["Scale"].default_value = 0.22
            sm2 = ng2.nodes.new("GeometryNodeSetMaterial"); sm2.inputs["Material"].default_value = m_glow
            ng2.links.new(hx.outputs["Mesh"], ex.inputs["Mesh"]); ng2.links.new(ex.outputs["Mesh"], sc.inputs["Geometry"]); ng2.links.new(ex.outputs["Top"], sc.inputs["Selection"])
            ng2.links.new(sc.outputs["Geometry"], sm2.inputs["Geometry"]); ng2.links.new(sm2.outputs["Geometry"], g2.inputs["Geometry"])

            o1 = bpy.data.objects.new("LampPost", bpy.data.meshes.new("PostM")); o1.location = (0, 0, 0.95); bpy.context.scene.collection.objects.link(o1); o1.modifiers.new("PostGN", "NODES").node_group = ng1
            o2 = bpy.data.objects.new("LampHead", bpy.data.meshes.new("HeadM")); o2.location = (0, 0, 2.15); bpy.context.scene.collection.objects.link(o2); o2.modifiers.new("HeadGN", "NODES").node_group = ng2
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.34, depth=0.14, location=(0, 0, 0.07)); bpy.context.active_object.data.materials.append(m_iron)
            """,
            None,
        ),
    ]
    for i, (rid, prompt, body, reason) in enumerate(specs):
        code = _wrap_script(i, _head(i), body, _tail(cam_loc=(4.5, -5.0, 3.3)))
        items.append({"id": rid, "prompt": prompt, "code": code, "reasoning": reason})
    return items


# ==============================================================================
# Family 7 (7 recipes): Noise / Voronoi Organic Sculpting
# ==============================================================================
def _build_family7() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    specs = [
        (
            "f7_01_meteorite_rock",
            "Build a cratered iron meteorite specimen displaced inward along vertex Normals with Voronoi cellular pits and Set Shade Smooth on a steel display cradle.",
            """\
            m_iron = make_mat("MeteoriteIron", (0.22, 0.21, 0.23), rough=0.42, metallic=0.88)
            m_cradle = make_mat("MuseumSteel", (0.62, 0.64, 0.68), rough=0.28, metallic=0.95)
            ng = bpy.data.node_groups.new("MeteoriteRock", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ico = ng.nodes.new("GeometryNodeMeshIcoSphere"); ico.inputs["Radius"].default_value = 0.72; ico.inputs["Subdivisions"].default_value = 4
            pos = ng.nodes.new("GeometryNodeInputPosition"); nrm = ng.nodes.new("GeometryNodeInputNormal")
            vor = ng.nodes.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 4.5
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.24
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_iron
            ng.links.new(pos.outputs["Position"], vor.inputs["Vector"])
            ng.links.new(vor.outputs["Distance"], amp.inputs[0])
            ng.links.new(nrm.outputs["Normal"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(ico.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"])
            ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            mobj = bpy.data.objects.new("Meteorite", bpy.data.meshes.new("MetM"))
            mobj.location = (0, 0, 0.78)
            bpy.context.scene.collection.objects.link(mobj)
            mobj.modifiers.new("MeteorGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.45, depth=0.14, location=(0, 0, 0.07))
            bpy.context.active_object.data.materials.append(m_cradle)
            """,
            "Plan: Displace high-res MeshIcoSphere along InputNormal using ShaderNodeTexVoronoi.outputs['Distance']*0.24 -> VectorMath SCALE -> SetPosition -> SetShadeSmooth.",
        ),
        (
            "f7_02_brain_coral",
            "Build a convoluted labyrinth brain-coral reef dome sculpted with high-distortion Wave Texture bands along vertex Normals over a sandy reef pad.",
            """\
            m_coral = pbr_mat("ReefCoral", (0.78, 0.64, 0.42), roughness=0.68)
            m_sand = pbr_mat("ReefSand", (0.86, 0.80, 0.66), roughness=0.88)
            ng = bpy.data.node_groups.new("BrainCoral", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeMeshUVSphere"); sp.inputs["Segments"].default_value = 64; sp.inputs["Rings"].default_value = 32; sp.inputs["Radius"].default_value = 0.75
            pos = ng.nodes.new("GeometryNodeInputPosition"); nrm = ng.nodes.new("GeometryNodeInputNormal")
            wv = ng.nodes.new("ShaderNodeTexWave"); wv.inputs["Scale"].default_value = 6.5; wv.inputs["Distortion"].default_value = 9.0; wv.inputs["Detail"].default_value = 3.0
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.085
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_coral
            ng.links.new(pos.outputs["Position"], wv.inputs["Vector"]); ng.links.new(wv.outputs["Factor"], amp.inputs[0])
            ng.links.new(nrm.outputs["Normal"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(sp.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"])
            ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            cobj = bpy.data.objects.new("BrainCoral", bpy.data.meshes.new("CorM"))
            cobj.location = (0, 0, 0.55)
            bpy.context.scene.collection.objects.link(cobj)
            cobj.modifiers.new("CoralGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=1.15, depth=0.10, location=(0, 0, 0.05))
            bpy.context.active_object.data.materials.append(m_sand)
            """,
            None,
        ),
        (
            "f7_03_basalt_cluster",
            "Build a cluster of Giant's Causeway hexagonal basalt rock columns on a grid of points whose heights vary organically via procedural Noise Texture.",
            """\
            m_basalt = make_mat("ColumnarBasalt", (0.18, 0.19, 0.21), rough=0.72)
            m_water = make_mat("TidePool", (0.10, 0.26, 0.30), rough=0.08)
            ng = bpy.data.node_groups.new("BasaltColumns", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            grid = ng.nodes.new("GeometryNodeMeshGrid"); grid.inputs["Size X"].default_value = 2.0; grid.inputs["Size Y"].default_value = 2.0
            grid.inputs["Vertices X"].default_value = 8; grid.inputs["Vertices Y"].default_value = 8
            m2p = ng.nodes.new("GeometryNodeMeshToPoints")
            col = ng.nodes.new("GeometryNodeMeshCylinder"); col.inputs["Vertices"].default_value = 6; col.inputs["Radius"].default_value = 0.14; col.inputs["Depth"].default_value = 1.0
            up = ng.nodes.new("GeometryNodeTransform"); up.inputs["Translation"].default_value = (0, 0, 0.5)
            pos = ng.nodes.new("GeometryNodeInputPosition")
            nz = ng.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1.5
            mr = ng.nodes.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = 0.25; mr.inputs["To Max"].default_value = 1.55
            comb = ng.nodes.new("ShaderNodeCombineXYZ"); comb.inputs["X"].default_value = 1.0; comb.inputs["Y"].default_value = 1.0
            inst = ng.nodes.new("GeometryNodeInstanceOnPoints"); real = ng.nodes.new("GeometryNodeRealizeInstances")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_basalt
            ng.links.new(grid.outputs["Mesh"], m2p.inputs["Mesh"]); ng.links.new(col.outputs["Mesh"], up.inputs["Geometry"])
            ng.links.new(pos.outputs["Position"], nz.inputs["Vector"]); ng.links.new(nz.outputs["Factor"], mr.inputs["Value"])
            ng.links.new(mr.outputs["Result"], comb.inputs["Z"])
            ng.links.new(m2p.outputs["Points"], inst.inputs["Points"]); ng.links.new(up.outputs["Geometry"], inst.inputs["Instance"]); ng.links.new(comb.outputs["Vector"], inst.inputs["Scale"])
            ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"]); ng.links.new(real.outputs["Geometry"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            bobj = bpy.data.objects.new("BasaltCluster", bpy.data.meshes.new("BasM"))
            bpy.context.scene.collection.objects.link(bobj)
            bobj.modifiers.new("BasaltGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1.65, depth=0.08, location=(0, 0, 0.04))
            bpy.context.active_object.data.materials.append(m_water)
            """,
            "Plan: MeshGrid(8x8) -> MeshToPoints -> InstanceOnPoints(6-sided MeshCylinder), modulate Z scale using TexNoise.Factor -> MapRange -> CombineXYZ(1,1,h).",
        ),
        (
            "f7_04_hammered_cauldron",
            "Build a hammered antique copper cauldron pot with fine Voronoi dimpled surface texture along vertex Normals on a forged iron tripod ring.",
            """\
            m_copper = pbr_mat("HammeredCopper", (0.78, 0.38, 0.20), roughness=0.28, metal=0.94)
            m_iron = pbr_mat("TripodIron", (0.14, 0.14, 0.16), roughness=0.48, metal=0.88)
            ng = bpy.data.node_groups.new("HammeredCauldron", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            sp = ng.nodes.new("GeometryNodeMeshUVSphere"); sp.inputs["Segments"].default_value = 64; sp.inputs["Rings"].default_value = 32; sp.inputs["Radius"].default_value = 0.68
            pos = ng.nodes.new("GeometryNodeInputPosition"); nrm = ng.nodes.new("GeometryNodeInputNormal")
            vor = ng.nodes.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 14.0
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.032
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_copper
            ng.links.new(pos.outputs["Position"], vor.inputs["Vector"]); ng.links.new(vor.outputs["Distance"], amp.inputs[0])
            ng.links.new(nrm.outputs["Normal"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(sp.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"]); ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            cobj = bpy.data.objects.new("Cauldron", bpy.data.meshes.new("CldM"))
            cobj.location = (0, 0, 0.72)
            bpy.context.scene.collection.objects.link(cobj)
            cobj.modifiers.new("CauldronGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.48, depth=0.16, location=(0, 0, 0.08))
            bpy.context.active_object.data.materials.append(m_iron)
            """,
            None,
        ),
        (
            "f7_05_river_boulder",
            "Build an eroded grey granite river boulder sculpted smoothly from an elongated IcoSphere displaced via 3D Noise Texture over a cedar display stand.",
            """\
            m_granite = make_mat("RiverGranite", (0.48, 0.49, 0.50), rough=0.62)
            m_cedar = make_mat("DisplayCedar", (0.42, 0.24, 0.12), rough=0.45)
            ng = bpy.data.node_groups.new("RiverBoulder", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            ico = ng.nodes.new("GeometryNodeMeshIcoSphere"); ico.inputs["Radius"].default_value = 0.78; ico.inputs["Subdivisions"].default_value = 4
            xf = ng.nodes.new("GeometryNodeTransform"); xf.inputs["Scale"].default_value = (1.35, 0.95, 0.72)
            pos = ng.nodes.new("GeometryNodeInputPosition"); nrm = ng.nodes.new("GeometryNodeInputNormal")
            nz = ng.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1.6; nz.inputs["Detail"].default_value = 4.0
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.28
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_granite
            ng.links.new(ico.outputs["Mesh"], xf.inputs["Geometry"])
            ng.links.new(pos.outputs["Position"], nz.inputs["Vector"]); ng.links.new(nz.outputs["Factor"], amp.inputs[0])
            ng.links.new(nrm.outputs["Normal"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(xf.outputs["Geometry"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"]); ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            bobj = bpy.data.objects.new("Boulder", bpy.data.meshes.new("BldM"))
            bobj.location = (0, 0, 0.62)
            bpy.context.scene.collection.objects.link(bobj)
            bobj.modifiers.new("BoulderGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.08))
            bpy.context.active_object.scale = (1.8, 1.3, 0.14)
            bpy.context.active_object.data.materials.append(m_cedar)
            """,
            None,
        ),
        (
            "f7_06_obsidian_spire",
            "Build a jagged volcanic obsidian shard spire carved from a high-segment cone displaced with Noise Texture and Voronoi ridges on a pumice plinth.",
            """\
            m_obs = pbr_mat("VolcanicObsidian", (0.04, 0.04, 0.06), roughness=0.08)
            m_pumice = pbr_mat("PumiceAsh", (0.36, 0.34, 0.32), roughness=0.88)
            ng = bpy.data.node_groups.new("ObsidianSpire", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            cone = ng.nodes.new("GeometryNodeMeshCone"); cone.inputs["Vertices"].default_value = 32; cone.inputs["Side Segments"].default_value = 24
            cone.inputs["Radius Bottom"].default_value = 0.62; cone.inputs["Radius Top"].default_value = 0.04; cone.inputs["Depth"].default_value = 2.1
            pos = ng.nodes.new("GeometryNodeInputPosition"); nrm = ng.nodes.new("GeometryNodeInputNormal")
            vor = ng.nodes.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 3.6
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.22
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_obs
            ng.links.new(pos.outputs["Position"], vor.inputs["Vector"]); ng.links.new(vor.outputs["Distance"], amp.inputs[0])
            ng.links.new(nrm.outputs["Normal"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(cone.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], sm.inputs["Geometry"]); ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            sobj = bpy.data.objects.new("Spire", bpy.data.meshes.new("SprM"))
            sobj.location = (0, 0, 1.1)
            bpy.context.scene.collection.objects.link(sobj)
            sobj.modifiers.new("SpireGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=28, radius=0.90, depth=0.14, location=(0, 0, 0.07))
            bpy.context.active_object.data.materials.append(m_pumice)
            """,
            None,
        ),
        (
            "f7_07_driftwood_arch",
            "Build a weathered sculptural coastal driftwood arch swept from a tapered Quadratic Bezier curve and sculpted with streaky Noise Texture ridges.",
            """\
            m_drift = make_mat("BleachedDriftwood", (0.64, 0.58, 0.48), rough=0.74)
            m_sand = make_mat("CoastalSand", (0.86, 0.80, 0.66), rough=0.88)
            ng = bpy.data.node_groups.new("DriftwoodArch", "GeometryNodeTree")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            go = ng.nodes.new("NodeGroupOutput")
            bz = ng.nodes.new("GeometryNodeCurveQuadraticBezier"); bz.inputs["Resolution"].default_value = 48
            bz.inputs["Start"].default_value = (-1.15, 0, 0.05); bz.inputs["Middle"].default_value = (-0.1, 0.25, 1.65); bz.inputs["End"].default_value = (1.15, 0, 0.08)
            sp = ng.nodes.new("GeometryNodeSplineParameter"); mr = ng.nodes.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = 0.20; mr.inputs["To Max"].default_value = 0.09
            sr = ng.nodes.new("GeometryNodeSetCurveRadius")
            cir = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); cir.inputs["Resolution"].default_value = 24; cir.inputs["Radius"].default_value = 1.0
            c2m = ng.nodes.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
            pos = ng.nodes.new("GeometryNodeInputPosition"); nrm = ng.nodes.new("GeometryNodeInputNormal")
            nz = ng.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 4.2; nz.inputs["Detail"].default_value = 4.0
            amp = ng.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; amp.inputs[1].default_value = 0.09
            vsc = ng.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
            set_p = ng.nodes.new("GeometryNodeSetPosition")
            smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
            sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = m_drift
            ng.links.new(sp.outputs["Factor"], mr.inputs["Value"]); ng.links.new(bz.outputs["Curve"], sr.inputs["Curve"]); ng.links.new(mr.outputs["Result"], sr.inputs["Radius"])
            ng.links.new(sr.outputs["Curve"], c2m.inputs["Curve"]); ng.links.new(cir.outputs["Curve"], c2m.inputs["Profile Curve"])
            ng.links.new(pos.outputs["Position"], nz.inputs["Vector"]); ng.links.new(nz.outputs["Factor"], amp.inputs[0])
            ng.links.new(nrm.outputs["Normal"], vsc.inputs["Vector"]); ng.links.new(amp.outputs["Value"], vsc.inputs["Scale"])
            ng.links.new(c2m.outputs["Mesh"], set_p.inputs["Geometry"]); ng.links.new(vsc.outputs["Vector"], set_p.inputs["Offset"])
            ng.links.new(set_p.outputs["Geometry"], smooth.inputs["Mesh"]); ng.links.new(smooth.outputs["Mesh"], sm.inputs["Geometry"])
            ng.links.new(sm.outputs["Geometry"], go.inputs["Geometry"])
            dobj = bpy.data.objects.new("Driftwood", bpy.data.meshes.new("DrfM"))
            bpy.context.scene.collection.objects.link(dobj)
            dobj.modifiers.new("DriftGN", "NODES").node_group = ng
            bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=1.55, depth=0.08, location=(0, 0, 0.04))
            bpy.context.active_object.data.materials.append(m_sand)
            """,
            "Plan: Sweep QuadraticBezier tapered with SplineParameter -> MapRange -> SetCurveRadius, then displace along InputNormal via TexNoise.outputs['Factor']*0.09.",
        ),
    ]
    for i, (rid, prompt, body, reason) in enumerate(specs):
        fn = f"build_{rid[6:]}" if i % 4 == 3 else None
        apply_obj = "bobj" if i == 2 and fn is None else None
        code = _wrap_script(i, _head(i), body, _tail(cam_loc=(4.2, -4.8, 3.2)), fn_name=fn, apply_mod_obj=apply_obj)
        items.append({"id": rid, "prompt": prompt, "code": code, "reasoning": reason})
    return items


def _check_anti_leakage(items: list[dict[str, Any]]) -> None:
    for item in items:
        hay = (item["id"] + " " + item["prompt"]).lower()
        for bad in FORBIDDEN_TERMS:
            if bad in hay:
                raise AssertionError(f"Leakage check failed: term '{bad}' found in {item['id']}: {item['prompt']}")


def _verify_one(item: dict[str, Any]) -> tuple[dict[str, Any], bv.VerifyResult]:
    vr = bv.verify_script(item["code"])
    return item, vr


def _select_replay_rows(tok: Any, max_replay: int = 12) -> list[dict[str, Any]]:
    replay: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    win_path = SCRIPT_DIR / "work" / "rl_harvested_winners.jsonl"
    if win_path.is_file():
        candidates = []
        for line in win_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ids, _ = tlora.format_conversation(r, tok, max_seq_len=99999, truncate=False)
            if len(ids) <= 2200:
                candidates.append((len(ids), r))
        candidates.sort(key=lambda x: x[0])
        for _, r in candidates[:4]:
            rid = str(r.get("id", "win"))
            seen_ids.add(rid)
            replay.append(r)

    strat_path = SCRIPT_DIR / "work" / "strat_a_verified.jsonl"
    if strat_path.is_file() and len(replay) < max_replay:
        by_src: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for line in strat_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ids, _ = tlora.format_conversation(r, tok, max_seq_len=99999, truncate=False)
            if len(ids) <= 2200:
                src = str(r.get("source") or "other")
                by_src.setdefault(src, []).append((len(ids), r))
        for src_list in by_src.values():
            src_list.sort(key=lambda x: x[0])

        # Round-robin across diverse sources, preferring rich multi-part sources before blenderbench_level1
        src_order = sorted(
            by_src.keys(),
            key=lambda s: (1 if "camera" in s else 0, by_src[s][0][0]),
        )
        idx = 0
        while len(replay) < max_replay:
            progressed = False
            for s in src_order:
                if len(replay) >= max_replay:
                    break
                if idx < len(by_src[s]):
                    _, r = by_src[s][idx]
                    rid = str(r.get("id"))
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        replay.append(r)
                        progressed = True
            if not progressed:
                break
            idx += 1

    return replay[:max_replay]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "work" / "curriculum_c4_compact_geonodes.jsonl",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1450)
    args = ap.parse_args()

    raw_items = (
        _build_family1()
        + _build_family2()
        + _build_family3()
        + _build_family4()
        + _build_family5()
        + _build_family6()
        + _build_family7()
    )
    assert len(raw_items) == 60, f"Expected 60 GeoNodes recipes, got {len(raw_items)}"
    _check_anti_leakage(raw_items)
    print(f"[c4_data] anti-leakage passed on {len(raw_items)} distinct GeoNodes recipes", flush=True)

    tok = tlora.load_qwen35_tokenizer()

    # Format prompts (~60% exact EVAL_SUFFIX, ~40% ALT_SUFFIXES) and keep ~35% with concise reasoning traces.
    formatted: list[dict[str, Any]] = []
    for i, item in enumerate(raw_items):
        suffix = EVAL_SUFFIX if (i % 5 != 1 and i % 5 != 3) else ALT_SUFFIXES[(i // 2) % len(ALT_SUFFIXES)]
        prompt_full = item["prompt"] + suffix
        # Clean code formatting
        code_clean = item["code"].strip() + "\n"
        reasoning = str(item.get("reasoning") or "").strip()
        row = {
            "id": item["id"],
            "system": GN_SYSTEM_PROMPT,
            "prompt": prompt_full,
            "reasoning": reasoning,
            "response": f"```python\n{code_clean}```",
            "source": "c4_compact_geonodes",
        }
        formatted.append(row)

    # Verify all 60 in parallel headless Blender 5.4
    failures: list[str] = []
    gn_tok_lens: list[int] = []
    rows_out: list[dict[str, Any]] = []

    print(f"[c4_data] verifying {len(formatted)} scripts in headless Blender 5.4 (workers={args.workers})...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(bv.verify_script, [r["response"] for r in formatted]))

    for row, vr in zip(formatted, results):
        gp = bv.is_geonodes_pass(vr)
        ss = vr.scene_summary or {}
        verts = int(ss.get("total_verts", 0))
        ids, _ = tlora.format_conversation(row, tok, max_seq_len=99999, truncate=False)
        n_tok = len(ids)
        gn_tok_lens.append(n_tok)
        row["tokens"] = n_tok
        row["reward"] = vr.reward
        row["verts"] = verts
        row["gn_trees"] = ss.get("geonode_trees_count", 0)
        row["gn_nodes"] = ss.get("geonode_nodes_total", 0)
        row["gn_links"] = ss.get("geonode_links_total", 0)

        ok = vr.syntax_ok and vr.exec_ok and gp and verts >= 100 and vr.reward >= 0.85 and n_tok <= args.max_tokens
        status = "OK  " if ok else "FAIL"
        print(
            f"  [{status}] {row['id']:26s} toks={n_tok:4d} w512={n_tok/512:.1f} "
            f"verts={verts:5d} gn=({row['gn_trees']}/{row['gn_nodes']}/{row['gn_links']}) R={vr.reward:.2f}",
            flush=True,
        )
        if not ok:
            failures.append(
                f"{row['id']}: syntax={vr.syntax_ok} exec={vr.exec_ok} gp={gp} verts={verts} R={vr.reward} toks={n_tok} err={vr.error[:240]}"
            )
        rows_out.append(row)

    if failures:
        print("\nFAILED RECIPES:", flush=True)
        for f in failures:
            print(" ", f)
        raise SystemExit(f"{len(failures)}/60 GeoNodes recipes failed verification gate!")

    replay_rows = _select_replay_rows(tok, max_replay=12)
    final_rows = rows_out + replay_rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in final_rows) + "\n",
        encoding="utf-8",
    )

    wins = [t / 512.0 for t in gn_tok_lens]
    n_reason = sum(1 for r in rows_out if r.get("reasoning"))
    print(f"\n=== DATASET READY: {args.out} ===", flush=True)
    print(f"  GeoNodes rows : {len(rows_out)}/60 passed Blender 5.4 (with_reasoning={n_reason}/60)", flush=True)
    print(f"  Replay rows   : {len(replay_rows)} (BlenderWorld / StratA <=2200 tok)", flush=True)
    print(f"  Total rows    : {len(final_rows)}", flush=True)
    print(
        f"  GeoNodes toks : min={min(gn_tok_lens)}, med={int(statistics.median(gn_tok_lens))}, "
        f"max={max(gn_tok_lens)} | 512-tok windows: med={statistics.median(wins):.2f}, max={max(wins):.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
