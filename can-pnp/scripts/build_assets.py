"""Convert the source assets in ``assets/source`` into MuJoCo-ready files.

* ``purple_can.usda`` -> ``purple_can_{body,seam,lid}.obj`` (visual meshes, unscaled USD units).
* ``carpet_patch.png`` -> ``carpet.png``: the 10 x 10 grid of 0.5 m patches that
  ``envs/isaaclab/scene.py`` authors, with every other patch's UVs rotated by 90 degrees,
  baked into one image so a single plane geom can carry it.
* mjlab ``g1.xml`` + ``g1_29dof_gripper_adapter20.xml`` -> ``robots/g1_aloha/g1_aloha.xml``:
  the G1 that SuperMimic is trained on (bodies, capsule feet, collision classes) with the
  rubber hands replaced by the ALOHA 2 grippers of the can_pnp robot: printed adapter,
  wrist-yaw inertia, gripper subtree, finger coupling and finger contact excludes.

Run once after updating ``assets/source`` or mjlab; outputs are committed under ``assets``.
"""

from __future__ import annotations

import copy
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "src" / "can_pnp" / "assets"
SOURCE = ASSETS / "source"
OUT = ASSETS / "scene"

CARPET_TILES = 10
CARPET_TILE_PX = 128


def _usd_array(block: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)} = \[(.*?)\]", block, re.S)
    if match is None:
        raise ValueError(f"USD mesh has no {key}")
    return match.group(1)


def convert_can_meshes() -> None:
    text = (SOURCE / "purple_can.usda").read_text()
    for name in ("Body", "Seam", "Lid"):
        block = re.search(rf'def Mesh "{name}" \((.*?)\n    }}', text, re.S)
        if block is None:
            raise ValueError(f"purple_can.usda has no mesh {name}")
        body = block.group(1)
        points = re.findall(r"\(([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\)", _usd_array(body, "points"))
        counts = [int(v) for v in _usd_array(body, "faceVertexCounts").split(",")]
        indices = [int(v) for v in _usd_array(body, "faceVertexIndices").split(",")]
        if sum(counts) != len(indices):
            raise ValueError(f"{name}: faceVertexCounts do not match faceVertexIndices")
        lines = [f"# {name} mesh of purple_can.usda"]
        lines += [f"v {x} {y} {z}" for x, y, z in points]
        start = 0
        for count in counts:
            face = indices[start : start + count]
            start += count
            # Fan-triangulate; OBJ indices are 1-based.
            for k in range(1, count - 1):
                lines.append(f"f {face[0] + 1} {face[k] + 1} {face[k + 1] + 1}")
        (OUT / f"purple_can_{name.lower()}.obj").write_text("\n".join(lines) + "\n")


def build_carpet_texture() -> None:
    patch = np.asarray(
        Image.open(SOURCE / "carpet_patch.png").convert("RGB").resize((CARPET_TILE_PX,) * 2)
    )
    # Image rows run top (v = 1) to bottom (v = 0). Patch (i, j) spans +x with i and +y with j.
    # Default UVs: u = +x, v = +y. Rotated patches ((i + j) odd): u = 1 - y, v = +x.
    rotated = np.rot90(patch, k=-1)
    n, px = CARPET_TILES, CARPET_TILE_PX
    canvas = np.zeros((n * px, n * px, 3), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            tile = rotated if (i + j) % 2 else patch
            # World +x -> image column, world +y -> image row (top row = largest y).
            row = (n - 1 - j) * px
            canvas[row : row + px, i * px : (i + 1) * px] = tile
    Image.fromarray(canvas).save(OUT / "carpet.png")


def _find(root: ET.Element, tag: str, name: str, key: str = "name") -> ET.Element:
    element = root.find(f".//{tag}[@{key}='{name}']")
    if element is None:
        raise ValueError(f"no <{tag} {key}={name!r}>")
    return element


def build_g1_aloha(g1_xml: Path) -> Path:
    g1 = ET.parse(g1_xml).getroot()
    aloha = ET.parse(SOURCE / "g1_29dof_gripper_adapter20.xml").getroot()
    out = ASSETS / "robots" / "g1_aloha"
    meshes = out / "meshes"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(g1_xml.parent / g1.find("compiler").get("meshdir"), meshes)
    shutil.copytree(SOURCE / "aloha2", meshes / "aloha2")
    for side in ("left", "right"):
        (meshes / f"{side}_rubber_hand.STL").unlink()

    g1.set("model", "g1_29dof_aloha2")
    g1.find("compiler").set("meshdir", "meshes")
    g1.find("default").append(copy.deepcopy(_find(aloha, "default", "aloha2_finger", key="class")))
    asset = g1.find("asset")
    for side in ("left", "right"):
        asset.remove(_find(asset, "mesh", f"{side}_rubber_hand"))
    for mesh in aloha.find("asset"):
        if mesh.get("file", "").startswith("aloha2/"):
            asset.append(copy.deepcopy(mesh))

    for side in ("left", "right"):
        wrist = _find(g1, "body", f"{side}_wrist_yaw_link")
        source = _find(aloha, "body", f"{side}_wrist_yaw_link")
        for geom in list(wrist.findall("geom")):
            if geom.get("mesh") == f"{side}_rubber_hand" or geom.get("name") == f"{side}_hand_collision":
                wrist.remove(geom)
        # The adapter20 wrist-yaw inertial carries the hand side of the printed adapter.
        wrist.remove(wrist.find("inertial"))
        wrist.insert(0, copy.deepcopy(source.find("inertial")))
        for child in source:
            if child.tag == "geom" and "printed_adapter" in child.get("name", ""):
                wrist.append(copy.deepcopy(child))
            elif child.tag == "body" and child.get("name") == f"{side}_gripper_base":
                wrist.append(copy.deepcopy(child))

    contact = g1.find("contact")
    for exclude in aloha.find("contact"):
        contact.append(copy.deepcopy(exclude))
    equality = ET.Element("equality")
    for joint in aloha.find("equality"):
        equality.append(copy.deepcopy(joint))
    g1.insert(list(g1).index(contact) + 1, equality)

    path = out / "g1_aloha.xml"
    tree = ET.ElementTree(g1)
    ET.indent(tree, space="  ")
    header = (
        f"<!-- Generated by scripts/build_assets.py from mjlab {g1_xml.name} and "
        "g1_29dof_gripper_adapter20.xml; do not edit. -->\n"
    )
    path.write_text(header + ET.tostring(g1, encoding="unicode") + "\n")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    convert_can_meshes()
    build_carpet_texture()
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")
    if len(sys.argv) > 1:
        g1_xml = Path(sys.argv[1])
    else:
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML as g1_xml
    print(f"wrote {build_g1_aloha(Path(g1_xml))}")


if __name__ == "__main__":
    main()
