#!/usr/bin/env python3
"""Evaluate single-step IK with synthetic `100 points -> MANO -> mesh` reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import trimesh
from manotorch.manolayer import ManoLayer

from utils.mano.approx import ApproxForwardManoEstimator


FINGER_SLICES = {
    "index": slice(0, 3),
    "middle": slice(3, 6),
    "pinky": slice(6, 9),
    "ring": slice(9, 12),
    "thumb": slice(12, 15),
}
ORIG_COLOR = np.array([75, 150, 255, 215], dtype=np.uint8)
RECON_COLOR = np.array([255, 140, 60, 215], dtype=np.uint8)
ORIG_ANCHOR_COLOR = np.array([40, 210, 120, 255], dtype=np.uint8)
RECON_ANCHOR_COLOR = np.array([255, 70, 70, 255], dtype=np.uint8)


def _setFingerAxis(handPose: np.ndarray, fingerName: str, axis: int, values: list[float]) -> None:
    handPose[FINGER_SLICES[fingerName], axis] = np.asarray(values, dtype=np.float32)


def _setFingerCurl(handPose: np.ndarray, fingerName: str, values: list[float]) -> None:
    _setFingerAxis(handPose, fingerName, 0, values)


def _buildPosePresets() -> list[dict]:
    presets: list[dict] = []

    restPose = np.zeros((15, 3), dtype=np.float32)
    presets.append({"name": "rest", "hand_pose": restPose, "root_rot": np.zeros((3,), dtype=np.float32), "transl": np.zeros((3,), dtype=np.float32)})

    deepFist = np.zeros((15, 3), dtype=np.float32)
    for fingerName in ("index", "middle", "ring", "pinky"):
        _setFingerCurl(deepFist, fingerName, [1.00, 1.42, 1.12])
    _setFingerCurl(deepFist, "thumb", [0.55, 0.82, 0.64])
    presets.append({
        "name": "deep_fist",
        "hand_pose": deepFist,
        "root_rot": np.array([0.10, -0.18, 0.08], dtype=np.float32),
        "transl": np.array([0.012, -0.006, 0.010], dtype=np.float32),
    })

    hookGrip = np.zeros((15, 3), dtype=np.float32)
    for fingerName in ("index", "middle", "ring", "pinky"):
        _setFingerCurl(hookGrip, fingerName, [0.28, 1.28, 1.02])
    _setFingerCurl(hookGrip, "thumb", [0.12, 0.28, 0.18])
    presets.append({
        "name": "hook_grip",
        "hand_pose": hookGrip,
        "root_rot": np.array([-0.08, 0.22, -0.12], dtype=np.float32),
        "transl": np.array([-0.010, 0.004, 0.008], dtype=np.float32),
    })

    pointing = np.zeros((15, 3), dtype=np.float32)
    _setFingerCurl(pointing, "index", [0.04, 0.06, 0.05])
    for fingerName in ("middle", "ring", "pinky"):
        _setFingerCurl(pointing, fingerName, [1.05, 1.36, 1.12])
    _setFingerCurl(pointing, "thumb", [0.34, 0.52, 0.42])
    _setFingerAxis(pointing, "thumb", 1, [-0.24, -0.18, -0.12])
    presets.append({
        "name": "pointing_folded",
        "hand_pose": pointing,
        "root_rot": np.array([0.06, -0.12, 0.18], dtype=np.float32),
        "transl": np.array([0.006, 0.008, -0.012], dtype=np.float32),
    })

    pinchTight = np.zeros((15, 3), dtype=np.float32)
    _setFingerCurl(pinchTight, "index", [0.88, 1.15, 0.96])
    _setFingerCurl(pinchTight, "middle", [0.44, 0.68, 0.52])
    _setFingerCurl(pinchTight, "ring", [0.22, 0.34, 0.26])
    _setFingerCurl(pinchTight, "pinky", [0.16, 0.30, 0.24])
    _setFingerCurl(pinchTight, "thumb", [0.78, 0.94, 0.72])
    _setFingerAxis(pinchTight, "thumb", 1, [-0.42, -0.28, -0.18])
    presets.append({
        "name": "pinch_tight",
        "hand_pose": pinchTight,
        "root_rot": np.array([-0.10, 0.08, -0.20], dtype=np.float32),
        "transl": np.array([-0.008, -0.010, 0.012], dtype=np.float32),
    })

    spreadClaw = np.zeros((15, 3), dtype=np.float32)
    for fingerName in ("index", "middle", "ring", "pinky"):
        _setFingerCurl(spreadClaw, fingerName, [0.56, 0.88, 0.72])
    _setFingerCurl(spreadClaw, "thumb", [0.30, 0.50, 0.38])
    _setFingerAxis(spreadClaw, "index", 1, [0.45, 0.18, 0.08])
    _setFingerAxis(spreadClaw, "middle", 1, [0.06, 0.02, 0.00])
    _setFingerAxis(spreadClaw, "ring", 1, [-0.28, -0.10, -0.04])
    _setFingerAxis(spreadClaw, "pinky", 1, [-0.54, -0.18, -0.08])
    _setFingerAxis(spreadClaw, "thumb", 1, [0.28, 0.16, 0.10])
    presets.append({
        "name": "spread_claw",
        "hand_pose": spreadClaw,
        "root_rot": np.array([0.14, -0.04, 0.10], dtype=np.float32),
        "transl": np.array([0.010, -0.014, 0.004], dtype=np.float32),
    })

    rng = np.random.default_rng(7)
    randomStrong = np.clip(rng.normal(loc=0.0, scale=0.48, size=(15, 3)), -1.00, 1.00).astype(np.float32)
    presets.append({
        "name": "random_strong",
        "hand_pose": randomStrong,
        "root_rot": np.array([0.18, -0.10, 0.14], dtype=np.float32),
        "transl": np.array([0.014, -0.010, 0.006], dtype=np.float32),
    })
    return presets


def _forwardMano(
    *,
    layer: ManoLayer,
    rootRot: np.ndarray,
    handPose: np.ndarray,
    betas: np.ndarray,
    transl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    poseCoeffs = np.concatenate([rootRot, handPose.reshape(-1)], axis=0).astype(np.float32)
    poseTensor = torch.from_numpy(poseCoeffs[None, :])
    betaTensor = torch.from_numpy(betas[None, :].astype(np.float32))
    output = layer(poseTensor, betaTensor)
    verts = output.verts[0].detach().cpu().numpy().astype(np.float32) + transl[None, :]
    joints = output.joints[0].detach().cpu().numpy().astype(np.float32) + transl[None, :]
    return verts, joints


def _buildMetrics(
    *,
    originalVerts: np.ndarray,
    reconstructedVerts: np.ndarray,
    sampleIndices: np.ndarray,
    gtHandPose: np.ndarray,
    predHandPose: np.ndarray,
) -> dict[str, float]:
    alignedVerts = _rigidAlign(sourceVerts=reconstructedVerts, targetVerts=originalVerts)
    vertError = np.linalg.norm(reconstructedVerts - originalVerts, axis=1)
    pointError = np.linalg.norm(reconstructedVerts[sampleIndices] - originalVerts[sampleIndices], axis=1)
    alignedVertError = np.linalg.norm(alignedVerts - originalVerts, axis=1)
    handPoseDiff = predHandPose - gtHandPose
    return {
        "vertex_rmse_mm": float(np.sqrt(np.mean(np.square(vertError))) * 1000.0),
        "vertex_mean_mm": float(np.mean(vertError) * 1000.0),
        "vertex_max_mm": float(np.max(vertError) * 1000.0),
        "rigid_aligned_vertex_rmse_mm": float(np.sqrt(np.mean(np.square(alignedVertError))) * 1000.0),
        "rigid_aligned_vertex_max_mm": float(np.max(alignedVertError) * 1000.0),
        "anchor_rmse_mm": float(np.sqrt(np.mean(np.square(pointError))) * 1000.0),
        "anchor_mean_mm": float(np.mean(pointError) * 1000.0),
        "anchor_max_mm": float(np.max(pointError) * 1000.0),
        "hand_pose_l2": float(np.linalg.norm(handPoseDiff)),
        "hand_pose_mae": float(np.mean(np.abs(handPoseDiff))),
        "gt_hand_pose_norm": float(np.linalg.norm(gtHandPose)),
        "pred_hand_pose_norm": float(np.linalg.norm(predHandPose)),
    }


def _rigidAlign(*, sourceVerts: np.ndarray, targetVerts: np.ndarray) -> np.ndarray:
    sourceCentered = sourceVerts - np.mean(sourceVerts, axis=0, keepdims=True)
    targetCentered = targetVerts - np.mean(targetVerts, axis=0, keepdims=True)
    cov = sourceCentered.T @ targetCentered
    uVal, _, vVal = np.linalg.svd(cov)
    rotation = uVal @ vVal
    if np.linalg.det(rotation) < 0:
        uVal[:, -1] *= -1.0
        rotation = uVal @ vVal
    translation = np.mean(targetVerts, axis=0) - np.mean(sourceVerts, axis=0) @ rotation
    return sourceVerts @ rotation + translation[None, :]


def _buildOutputStem(gestureFilter: set[str]) -> str:
    if not gestureFilter:
        return "gesture_reconstruction_compare"
    tag = "_".join(sorted(gestureFilter))
    safeTag = re.sub(r"[^a-zA-Z0-9_]+", "_", tag).strip("_")
    return f"gesture_reconstruction_compare_{safeTag}"


def _buildScene(
    *,
    faces: np.ndarray,
    results: list[dict],
    sampleIndices: np.ndarray,
    anchorRadius: float,
) -> trimesh.Scene:
    scene = trimesh.Scene()
    cellWidth = 0.24
    for poseIdx, result in enumerate(results):
        rowIdx = poseIdx // 3
        colIdx = poseIdx % 3
        baseOffset = np.array([colIdx * cellWidth, 0.0, rowIdx * 0.22], dtype=np.float32)
        center = result["original_joints"][0].astype(np.float32)
        originalVerts = result["original_verts"] - center[None, :] + baseOffset[None, :]
        reconstructedVerts = result["reconstructed_verts"] - center[None, :] + baseOffset[None, :]
        originalAnchors = result["original_verts"][sampleIndices] - center[None, :] + baseOffset[None, :]
        reconstructedAnchors = result["reconstructed_verts"][sampleIndices] - center[None, :] + baseOffset[None, :]

        originalMesh = trimesh.Trimesh(vertices=originalVerts, faces=faces, process=False)
        originalMesh.visual.vertex_colors = np.tile(ORIG_COLOR[None, :], (originalMesh.vertices.shape[0], 1))
        scene.add_geometry(originalMesh, geom_name=f"{result['name']}_original")

        reconstructedMesh = trimesh.Trimesh(vertices=reconstructedVerts, faces=faces, process=False)
        reconstructedMesh.visual.vertex_colors = np.tile(RECON_COLOR[None, :], (reconstructedMesh.vertices.shape[0], 1))
        scene.add_geometry(reconstructedMesh, geom_name=f"{result['name']}_reconstructed")
        for anchorIdx, point in enumerate(originalAnchors):
            marker = trimesh.creation.uv_sphere(radius=anchorRadius)
            marker.apply_translation(point)
            marker.visual.vertex_colors = np.tile(ORIG_ANCHOR_COLOR[None, :], (marker.vertices.shape[0], 1))
            scene.add_geometry(marker, geom_name=f"{result['name']}_orig_anchor_{anchorIdx:03d}")
        for anchorIdx, point in enumerate(reconstructedAnchors):
            marker = trimesh.creation.uv_sphere(radius=anchorRadius * 0.82)
            marker.apply_translation(point)
            marker.visual.vertex_colors = np.tile(RECON_ANCHOR_COLOR[None, :], (marker.vertices.shape[0], 1))
            scene.add_geometry(marker, geom_name=f"{result['name']}_recon_anchor_{anchorIdx:03d}")
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-pass MANO IK from 100 sampled points")
    parser.add_argument("--mano-path", type=str, default="assets/mano")
    parser.add_argument("--hand-side", type=str, default="right")
    parser.add_argument("--index-output", type=str, default="assets/part_ik_hand_index_100.npy")
    parser.add_argument("--output-dir", type=str, default="outputs/mano_ik")
    parser.add_argument("--gestures", type=str, default="", help="Comma-separated gesture names to export; empty means export all")
    parser.add_argument("--anchor-radius", type=float, default=0.0024, help="Anchor sphere radius in the exported GLB")
    args = parser.parse_args()

    estimator = ApproxForwardManoEstimator(manoPath=args.mano_path, handSide=args.hand_side, device="cpu")
    indexArray = np.asarray(estimator.template.indexOrder, dtype=np.int64)
    indexOutputPath = Path(args.index_output)
    indexOutputPath.parent.mkdir(parents=True, exist_ok=True)
    np.save(indexOutputPath, indexArray)

    layer = ManoLayer(
        side=args.hand_side,
        use_pca=False,
        flat_hand_mean=True,
        ncomps=45,
        mano_assets_root=args.mano_path,
    )
    faces = layer.th_faces.detach().cpu().numpy().astype(np.int64)
    betas = np.zeros((10,), dtype=np.float32)
    results: list[dict] = []
    gestureFilter = {item.strip() for item in args.gestures.split(",") if item.strip()}

    for preset in _buildPosePresets():
        if gestureFilter and preset["name"] not in gestureFilter:
            continue
        originalVerts, originalJoints = _forwardMano(
            layer=layer,
            rootRot=preset["root_rot"],
            handPose=preset["hand_pose"],
            betas=betas,
            transl=preset["transl"],
        )
        sampledPoints = torch.from_numpy(originalVerts[indexArray][None, :, :])
        estimate = estimator.estimate(sampledPoints)
        recoveredFullMano = estimate.fullMano[0].detach().cpu().numpy().astype(np.float32)
        reconstructedVerts, reconstructedJoints = _forwardMano(
            layer=layer,
            rootRot=recoveredFullMano[0:3],
            handPose=recoveredFullMano[3:48].reshape(15, 3),
            betas=recoveredFullMano[51:61],
            transl=recoveredFullMano[48:51],
        )
        metrics = _buildMetrics(
            originalVerts=originalVerts,
            reconstructedVerts=reconstructedVerts,
            sampleIndices=indexArray,
            gtHandPose=preset["hand_pose"].reshape(-1),
            predHandPose=recoveredFullMano[3:48],
        )
        metrics["fallback_count"] = int(estimate.fallbackCount)
        metrics["root_rot_l2"] = float(np.linalg.norm(recoveredFullMano[0:3] - preset["root_rot"]))
        metrics["transl_l2_mm"] = float(np.linalg.norm(recoveredFullMano[48:51] - preset["transl"]) * 1000.0)
        results.append(
            {
                "name": preset["name"],
                "original_verts": originalVerts,
                "original_joints": originalJoints,
                "reconstructed_verts": reconstructedVerts,
                "reconstructed_joints": reconstructedJoints,
                "metrics": metrics,
            }
        )

    outputDir = Path(args.output_dir)
    outputDir.mkdir(parents=True, exist_ok=True)
    if not results:
        raise ValueError(f"No valid gestures remain after filtering with --gestures: {args.gestures}")
    scene = _buildScene(
        faces=faces,
        results=results,
        sampleIndices=indexArray,
        anchorRadius=float(args.anchor_radius),
    )
    outputStem = _buildOutputStem(gestureFilter)
    glbPath = outputDir / f"{outputStem}.glb"
    glbPath.write_bytes(scene.export(file_type="glb"))

    metricList = [result["metrics"] for result in results]
    summary = {
        "index_output": str(indexOutputPath),
        "glb_output": str(glbPath),
        "gesture_count": len(results),
        "aggregate": {
            "vertex_rmse_mm_mean": float(np.mean([item["vertex_rmse_mm"] for item in metricList])),
            "vertex_rmse_mm_max": float(np.max([item["vertex_rmse_mm"] for item in metricList])),
            "rigid_aligned_vertex_rmse_mm_mean": float(np.mean([item["rigid_aligned_vertex_rmse_mm"] for item in metricList])),
            "rigid_aligned_vertex_rmse_mm_max": float(np.max([item["rigid_aligned_vertex_rmse_mm"] for item in metricList])),
            "anchor_rmse_mm_mean": float(np.mean([item["anchor_rmse_mm"] for item in metricList])),
            "anchor_rmse_mm_max": float(np.max([item["anchor_rmse_mm"] for item in metricList])),
            "hand_pose_mae_mean": float(np.mean([item["hand_pose_mae"] for item in metricList])),
            "hand_pose_l2_mean": float(np.mean([item["hand_pose_l2"] for item in metricList])),
            "transl_l2_mm_mean": float(np.mean([item["transl_l2_mm"] for item in metricList])),
            "transl_l2_mm_max": float(np.max([item["transl_l2_mm"] for item in metricList])),
            "fallback_count_mean": float(np.mean([item["fallback_count"] for item in metricList])),
            "fallback_count_max": int(np.max([item["fallback_count"] for item in metricList])),
        },
        "per_gesture": [
            {
                "name": result["name"],
                **result["metrics"],
            }
            for result in results
        ],
    }
    metricsPath = outputDir / f"{outputStem}_metrics.json"
    metricsPath.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved index npy to: {indexOutputPath}")
    print(f"Saved comparison glb to: {glbPath}")
    print(f"Saved metrics json to: {metricsPath}")
    print(
        "aggregate "
        f"vertex_rmse_mm_mean={summary['aggregate']['vertex_rmse_mm_mean']:.3f} "
        f"vertex_rmse_mm_max={summary['aggregate']['vertex_rmse_mm_max']:.3f} "
        f"rigid_aligned_vertex_rmse_mm_mean={summary['aggregate']['rigid_aligned_vertex_rmse_mm_mean']:.3f} "
        f"anchor_rmse_mm_mean={summary['aggregate']['anchor_rmse_mm_mean']:.3f} "
        f"anchor_rmse_mm_max={summary['aggregate']['anchor_rmse_mm_max']:.3f}"
    )


if __name__ == "__main__":
    main()
