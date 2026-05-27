#!/usr/bin/env python3
"""Estimate single-hand MANO from 100 semantic points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mano.approx import ApproxForwardManoEstimator
from utils.mano.helpers import decode_single_hand_mano
from utils.mano.mano_load import createManoLayer, resolveManoPath
from utils.mano.payload import invert_permutation, load_single_hand_points, resolve_input_reorder
from utils.vis.trimesh_vis import add_sphere, build_hand_mesh


OBSERVED_COLOR = np.array([255, 90, 90, 255], dtype=np.uint8)
REPROJECTED_COLOR = np.array([70, 205, 125, 255], dtype=np.uint8)
MESH_COLOR = np.array([245, 205, 165, 185], dtype=np.uint8)


def _load_known_shape(path: str | None) -> torch.Tensor | None:
    if not path:
        return None
    raw = np.load(str(path), allow_pickle=True)
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 10:
        raise ValueError(f"known shape must have 10 dimensions, got {arr.shape[0]}")
    return torch.from_numpy(arr)


def _build_estimators(
    *,
    mano_path: Path,
    axis_prior_path: str,
) -> tuple[ApproxForwardManoEstimator, ApproxForwardManoEstimator]:
    left_estimator = ApproxForwardManoEstimator(
        manoPath=str(mano_path),
        handSide="left",
        device="cpu",
        axisPriorPath=str(axis_prior_path),
    )
    right_estimator = ApproxForwardManoEstimator(
        manoPath=str(mano_path),
        handSide="right",
        device="cpu",
        axisPriorPath=str(axis_prior_path),
    )
    return left_estimator, right_estimator


def _resolve_points_and_reorder(
    *,
    args,
    mano_path: Path,
) -> tuple[np.ndarray, dict, ApproxForwardManoEstimator, np.ndarray, str]:
    sample_indices = np.load(str(args.sample_index_path)).astype(np.int64).reshape(-1)
    input_points, meta = load_single_hand_points(args.points_path, handSide=args.hand_side)
    left_estimator, right_estimator = _build_estimators(mano_path=mano_path, axis_prior_path=args.axis_prior_path)
    estimator = left_estimator if args.hand_side == "left" else right_estimator
    source_hand_side = args.source_hand_side if args.source_hand_side != "auto" else str(meta.get("sample_index_source_hand", "auto"))
    reorder_index, detected_source_hand = resolve_input_reorder(
        sampleIndices=meta.get("sample_index_order", sample_indices),
        targetHandSide=args.hand_side,
        sourceHandSide=source_hand_side,
        leftEstimator=left_estimator,
        rightEstimator=right_estimator,
    )
    return input_points, meta, estimator, reorder_index, detected_source_hand


def _build_mano_layer(mano_path: Path):
    mano_layer = createManoLayer(modelPath=str(mano_path), device="cpu")
    for side in ("left", "right"):
        mano_layer[side].eval()
        for param in mano_layer[side].parameters():
            param.requires_grad_(False)
    return mano_layer


def _run_estimate(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mano_path = resolveManoPath(manoPath=args.mano_path, projectRoot=PROJECT_ROOT)
    input_points, meta, estimator, reorder_index, detected_source_hand = _resolve_points_and_reorder(
        args=args,
        mano_path=mano_path,
    )
    ordered_points = input_points[reorder_index]
    estimate = estimator.estimate(torch.from_numpy(ordered_points).float(), knownShape=_load_known_shape(args.known_shape_path))
    pred_mano = estimate.fullMano.detach().cpu().numpy().reshape(61).astype(np.float32)

    mano_layer = _build_mano_layer(mano_path)
    pred_verts, pred_joints = decode_single_hand_mano(manoLayer=mano_layer, manoParams=pred_mano, handSide=args.hand_side)
    pred_points_ordered = pred_verts[np.asarray(estimator.template.indexOrder, dtype=np.int64)]
    inverse_reorder = invert_permutation(reorder_index)
    pred_points_input_order = pred_points_ordered[inverse_reorder]

    point_error = np.linalg.norm(pred_points_input_order - input_points, axis=1)
    metrics = {
        "input_point_rmse_mm": float(np.sqrt(np.mean(np.square(point_error))) * 1000.0),
        "input_point_mean_mm": float(np.mean(point_error) * 1000.0),
        "input_point_max_mm": float(np.max(point_error) * 1000.0),
        "fallback_count": int(estimate.fallbackCount),
        "hand_side": args.hand_side,
        "source_hand_side": detected_source_hand,
        "points_key": str(meta.get("points_key", "array")),
    }

    sample_name = str(meta.get("sample_name", Path(args.points_path).stem))
    scene = trimesh.Scene()
    scene.add_geometry(build_hand_mesh(pred_verts, mano_layer[args.hand_side].faces, color=MESH_COLOR), node_name="pred_mesh")
    for idx, point in enumerate(input_points):
        add_sphere(scene, point, args.point_radius, OBSERVED_COLOR, f"obs_{idx:03d}")
    for idx, point in enumerate(pred_points_input_order):
        add_sphere(scene, point, args.point_radius * 0.78, REPROJECTED_COLOR, f"reproj_{idx:03d}")

    np.save(str(output_dir / "pred_mano.npy"), pred_mano)
    np.save(str(output_dir / "pred_vertices.npy"), pred_verts)
    np.save(str(output_dir / "pred_joints.npy"), pred_joints)
    np.save(str(output_dir / "reprojected_points.npy"), pred_points_input_order)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "estimate_summary.json").write_text(
        json.dumps(
            {
                "sample_name": sample_name,
                "hand_side": args.hand_side,
                "source_hand_side": detected_source_hand,
                "sample_index_order": meta.get("sample_index_order", np.load(str(args.sample_index_path)).astype(np.int64).reshape(-1)).tolist(),
                "input_points_path": str(Path(args.points_path).resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (output_dir / "estimate_visualization.glb").write_bytes(scene.export(file_type="glb"))
    np.save(
        str(output_dir / "estimate_payload.npy"),
        {
            "sample_name": sample_name,
            "hand_side": args.hand_side,
            "sample_index_order": meta.get("sample_index_order", np.load(str(args.sample_index_path)).astype(np.int64).reshape(-1)).astype(np.int64),
            "sample_index_source_hand": detected_source_hand,
            "points_world": input_points.astype(np.float32),
            "reprojected_points_world": pred_points_input_order.astype(np.float32),
            "mano_params": pred_mano.astype(np.float32),
        },
        allow_pickle=True,
    )
    print(f"[OK] saved estimate to: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=False))


def _add_shared_point_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--points-path", type=str, required=True, help="Input file containing 100 semantic points")
    parser.add_argument("--mano-path", type=str, default=None, help="MANO model directory")
    parser.add_argument("--hand-side", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--source-hand-side", type=str, default="auto", choices=["auto", "left", "right"])
    parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    parser.add_argument("--axis-prior-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate MANO from 100 semantic points")
    _add_shared_point_args(parser)
    parser.add_argument("--known-shape-path", type=str, default=None, help="Optional 10D MANO betas file")
    parser.add_argument("--output-dir", type=str, default="outputs/single_ik")
    parser.add_argument("--point-radius", type=float, default=0.0021)
    args = parser.parse_args()
    _run_estimate(args)


if __name__ == "__main__":
    main()
