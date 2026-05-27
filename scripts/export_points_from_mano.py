#!/usr/bin/env python3
"""Export mesh, 100 semantic points, and visualization from a single-hand MANO vector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mano.helpers import decode_single_hand_mano
from utils.mano.mano_load import createManoLayer, resolveManoPath
from utils.mano.payload import load_single_hand_mano
from utils.vis.trimesh_vis import add_sphere, build_hand_mesh


POINT_COLOR = np.array([70, 205, 125, 255], dtype=np.uint8)
MESH_COLOR = np.array([245, 205, 165, 185], dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 100 semantic points and mesh from MANO")
    parser.add_argument("--mano-params-path", type=str, required=True, help="Input MANO parameter file")
    parser.add_argument("--mano-path", type=str, default=None, help="MANO model directory")
    parser.add_argument("--hand-side", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    parser.add_argument("--output-dir", type=str, default="outputs/export_points_from_mano")
    parser.add_argument("--point-radius", type=float, default=0.0021)
    args = parser.parse_args()

    outputDir = Path(args.output_dir)
    outputDir.mkdir(parents=True, exist_ok=True)
    manoPath = resolveManoPath(manoPath=args.mano_path, projectRoot=PROJECT_ROOT)
    sampleIndices = np.load(str(args.sample_index_path)).astype(np.int64).reshape(-1)
    manoParams, meta = load_single_hand_mano(args.mano_params_path, handSide=args.hand_side)

    manoLayer = createManoLayer(modelPath=str(manoPath), device="cpu")
    for side in ("left", "right"):
        manoLayer[side].eval()
        for param in manoLayer[side].parameters():
            param.requires_grad_(False)
    verts, joints = decode_single_hand_mano(manoLayer=manoLayer, manoParams=manoParams, handSide=args.hand_side)
    sampledPoints = verts[sampleIndices]

    scene = trimesh.Scene()
    scene.add_geometry(build_hand_mesh(verts, manoLayer[args.hand_side].faces, color=MESH_COLOR), node_name="mesh")
    for idx, point in enumerate(sampledPoints):
        add_sphere(scene, point, args.point_radius, POINT_COLOR, f"sample_{idx:03d}")

    sampleName = str(meta.get("sample_name", Path(args.mano_params_path).stem))
    np.save(str(outputDir / "sampled_points.npy"), sampledPoints.astype(np.float32))
    np.save(str(outputDir / "mesh_vertices.npy"), verts.astype(np.float32))
    np.save(str(outputDir / "mano_joints.npy"), joints.astype(np.float32))
    np.save(str(outputDir / "mano_params.npy"), manoParams.astype(np.float32))
    (outputDir / "sample_visualization.glb").write_bytes(scene.export(file_type="glb"))
    trimesh.Trimesh(vertices=verts, faces=manoLayer[args.hand_side].faces.astype(np.int64), process=False).export(outputDir / "mesh.obj")
    (outputDir / "summary.json").write_text(
        json.dumps(
            {
                "sample_name": sampleName,
                "hand_side": args.hand_side,
                "sample_index_order": sampleIndices.tolist(),
                "mano_params_path": str(Path(args.mano_params_path).resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    payload = {
        "sample_name": sampleName,
        "hand_side": args.hand_side,
        "sample_index_order": sampleIndices.astype(np.int64),
        "sample_index_source_hand": "right",
        "points_world": sampledPoints.astype(np.float32),
        "mano_params": manoParams.astype(np.float32),
    }
    np.save(str(outputDir / "sample_payload.npy"), payload, allow_pickle=True)
    print(f"[OK] saved mesh and sampled points to: {outputDir}")


if __name__ == "__main__":
    main()
