#!/usr/bin/env python3
"""Estimate MANO from 100 semantic points."""
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


LEFT_OBSERVED_COLOR = np.array([255, 100, 100, 255], dtype=np.uint8)
RIGHT_OBSERVED_COLOR = np.array([90, 180, 255, 255], dtype=np.uint8)
LEFT_REPROJECTED_COLOR = np.array([255, 200, 80, 255], dtype=np.uint8)
RIGHT_REPROJECTED_COLOR = np.array([80, 235, 150, 255], dtype=np.uint8)
LEFT_MESH_COLOR = np.array([245, 165, 165, 185], dtype=np.uint8)
RIGHT_MESH_COLOR = np.array([165, 205, 245, 185], dtype=np.uint8)


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


def _estimate_hand(
    *,
    points_path: str,
    hand_side: str,
    source_hand_side: str,
    sample_index_path: str,
    axis_prior_path: str,
    mano_path: Path,
    known_shape: torch.Tensor | None,
    mano_layer,
) -> dict[str, object]:
    sample_indices = np.load(str(sample_index_path)).astype(np.int64).reshape(-1)
    input_points, meta = load_single_hand_points(points_path, handSide=hand_side)
    left_estimator, right_estimator = _build_estimators(mano_path=mano_path, axis_prior_path=axis_prior_path)
    estimator = left_estimator if hand_side == "left" else right_estimator
    resolved_source_hand_side = source_hand_side if source_hand_side != "auto" else str(meta.get("sample_index_source_hand", "auto"))
    reorder_index, detected_source_hand = resolve_input_reorder(
        sampleIndices=meta.get("sample_index_order", sample_indices),
        targetHandSide=hand_side,
        sourceHandSide=resolved_source_hand_side,
        leftEstimator=left_estimator,
        rightEstimator=right_estimator,
    )
    ordered_points = input_points[reorder_index]
    estimate = estimator.estimate(torch.from_numpy(ordered_points).float(), knownShape=known_shape)
    pred_mano = estimate.fullMano.detach().cpu().numpy().reshape(61).astype(np.float32)
    pred_verts, _ = decode_single_hand_mano(manoLayer=mano_layer, manoParams=pred_mano, handSide=hand_side)
    pred_points_ordered = pred_verts[np.asarray(estimator.template.indexOrder, dtype=np.int64)]
    pred_points_input_order = pred_points_ordered[invert_permutation(reorder_index)]
    point_error = np.linalg.norm(pred_points_input_order - input_points, axis=1)
    return {
        "hand_side": hand_side,
        "input_points": input_points,
        "meta": meta,
        "pred_mano": pred_mano,
        "pred_verts": pred_verts,
        "pred_points_input_order": pred_points_input_order,
        "detected_source_hand": detected_source_hand,
        "fallback_count": int(estimate.fallbackCount),
        "point_error": point_error,
    }


def _build_single_hand_metrics(*, result: dict[str, object]) -> dict[str, object]:
    point_error = np.asarray(result["point_error"], dtype=np.float32)
    meta = dict(result["meta"])
    return {
        "input_point_rmse_mm": float(np.sqrt(np.mean(np.square(point_error))) * 1000.0),
        "input_point_mean_mm": float(np.mean(point_error) * 1000.0),
        "input_point_max_mm": float(np.max(point_error) * 1000.0),
        "fallback_count": int(result["fallback_count"]),
        "hand_side": str(result["hand_side"]),
        "source_hand_side": str(result["detected_source_hand"]),
        "points_key": str(meta.get("points_key", "array")),
        "sample_name": str(meta.get("sample_name", Path(str(meta.get("source_path", "sample"))).stem)),
    }


def _build_both_metrics(*, left_result: dict[str, object], right_result: dict[str, object]) -> dict[str, object]:
    left_error = np.asarray(left_result["point_error"], dtype=np.float32)
    right_error = np.asarray(right_result["point_error"], dtype=np.float32)
    all_error = np.concatenate([left_error, right_error], axis=0)
    left_meta = dict(left_result["meta"])
    right_meta = dict(right_result["meta"])
    sample_name = str(left_meta.get("sample_name", right_meta.get("sample_name", Path(str(left_meta.get("source_path", "sample"))).stem)))
    return {
        "hand_side": "both",
        "source_hand_side": str(left_result["detected_source_hand"]),
        "sample_name": sample_name,
        "left_points_key": str(left_meta.get("points_key", "array")),
        "right_points_key": str(right_meta.get("points_key", "array")),
        "left_input_point_rmse_mm": float(np.sqrt(np.mean(np.square(left_error))) * 1000.0),
        "right_input_point_rmse_mm": float(np.sqrt(np.mean(np.square(right_error))) * 1000.0),
        "input_point_rmse_mm": float(np.sqrt(np.mean(np.square(all_error))) * 1000.0),
        "left_input_point_mean_mm": float(np.mean(left_error) * 1000.0),
        "right_input_point_mean_mm": float(np.mean(right_error) * 1000.0),
        "input_point_mean_mm": float(np.mean(all_error) * 1000.0),
        "left_input_point_max_mm": float(np.max(left_error) * 1000.0),
        "right_input_point_max_mm": float(np.max(right_error) * 1000.0),
        "input_point_max_mm": float(np.max(all_error) * 1000.0),
        "left_fallback_count": int(left_result["fallback_count"]),
        "right_fallback_count": int(right_result["fallback_count"]),
        "fallback_count": int(left_result["fallback_count"]) + int(right_result["fallback_count"]),
    }


def _add_hand_geometry(
    *,
    scene: trimesh.Scene,
    hand_side: str,
    pred_verts: np.ndarray,
    input_points: np.ndarray,
    pred_points_input_order: np.ndarray,
    faces: np.ndarray,
    point_radius: float,
) -> None:
    observed_color = LEFT_OBSERVED_COLOR if hand_side == "left" else RIGHT_OBSERVED_COLOR
    reproj_color = LEFT_REPROJECTED_COLOR if hand_side == "left" else RIGHT_REPROJECTED_COLOR
    mesh_color = LEFT_MESH_COLOR if hand_side == "left" else RIGHT_MESH_COLOR
    scene.add_geometry(build_hand_mesh(pred_verts, faces, color=mesh_color), node_name=f"{hand_side}_pred_mesh")
    for idx, point in enumerate(input_points):
        add_sphere(scene, point, point_radius, observed_color, f"{hand_side}_obs_{idx:03d}")
    for idx, point in enumerate(pred_points_input_order):
        add_sphere(scene, point, point_radius * 0.78, reproj_color, f"{hand_side}_reproj_{idx:03d}")


def _run_estimate(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mano_path = resolveManoPath(manoPath=args.mano_path, projectRoot=PROJECT_ROOT)
    mano_layer = _build_mano_layer(mano_path)
    known_shape = _load_known_shape(args.known_shape_path)

    if args.hand_side in {"left", "right"}:
        result = _estimate_hand(
            points_path=args.points_path,
            hand_side=args.hand_side,
            source_hand_side=args.source_hand_side,
            sample_index_path=args.sample_index_path,
            axis_prior_path=args.axis_prior_path,
            mano_path=mano_path,
            known_shape=known_shape,
            mano_layer=mano_layer,
        )
        metrics = _build_single_hand_metrics(result=result)
        scene = trimesh.Scene()
        _add_hand_geometry(
            scene=scene,
            hand_side=args.hand_side,
            pred_verts=np.asarray(result["pred_verts"], dtype=np.float32),
            input_points=np.asarray(result["input_points"], dtype=np.float32),
            pred_points_input_order=np.asarray(result["pred_points_input_order"], dtype=np.float32),
            faces=mano_layer[args.hand_side].faces,
            point_radius=args.point_radius,
        )
        np.save(str(output_dir / "single_ik_mano_params.npy"), np.asarray(result["pred_mano"], dtype=np.float32))
        (output_dir / "single_ik_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_dir / "single_ik_visualization.glb").write_bytes(scene.export(file_type="glb"))
        print(f"[OK] saved estimate to: {output_dir}")
        print(json.dumps(metrics, ensure_ascii=False))
        return

    left_result = _estimate_hand(
        points_path=args.points_path,
        hand_side="left",
        source_hand_side=args.source_hand_side,
        sample_index_path=args.sample_index_path,
        axis_prior_path=args.axis_prior_path,
        mano_path=mano_path,
        known_shape=known_shape,
        mano_layer=mano_layer,
    )
    right_result = _estimate_hand(
        points_path=args.points_path,
        hand_side="right",
        source_hand_side=args.source_hand_side,
        sample_index_path=args.sample_index_path,
        axis_prior_path=args.axis_prior_path,
        mano_path=mano_path,
        known_shape=known_shape,
        mano_layer=mano_layer,
    )
    metrics = _build_both_metrics(left_result=left_result, right_result=right_result)
    scene = trimesh.Scene()
    _add_hand_geometry(
        scene=scene,
        hand_side="left",
        pred_verts=np.asarray(left_result["pred_verts"], dtype=np.float32),
        input_points=np.asarray(left_result["input_points"], dtype=np.float32),
        pred_points_input_order=np.asarray(left_result["pred_points_input_order"], dtype=np.float32),
        faces=mano_layer["left"].faces,
        point_radius=args.point_radius,
    )
    _add_hand_geometry(
        scene=scene,
        hand_side="right",
        pred_verts=np.asarray(right_result["pred_verts"], dtype=np.float32),
        input_points=np.asarray(right_result["input_points"], dtype=np.float32),
        pred_points_input_order=np.asarray(right_result["pred_points_input_order"], dtype=np.float32),
        faces=mano_layer["right"].faces,
        point_radius=args.point_radius,
    )
    pred_mano = np.concatenate(
        [
            np.asarray(left_result["pred_mano"], dtype=np.float32),
            np.asarray(right_result["pred_mano"], dtype=np.float32),
        ],
        axis=0,
    )
    np.save(str(output_dir / "single_ik_mano_params.npy"), pred_mano)
    (output_dir / "single_ik_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "single_ik_visualization.glb").write_bytes(scene.export(file_type="glb"))
    print(f"[OK] saved estimate to: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=False))


def _add_shared_point_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--points-path", type=str, required=True, help="Input file containing 100 semantic points")
    parser.add_argument("--mano-path", type=str, default=None, help="MANO model directory")
    parser.add_argument("--hand-side", type=str, default="right", choices=["left", "right", "both"])
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
