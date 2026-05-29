#!/usr/bin/env python3
"""Visualize a MANO sample sequence, extract 100 semantic points, and run per-frame IK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
projectRootStr = str(PROJECT_ROOT)
if projectRootStr in sys.path:
    sys.path.remove(projectRootStr)
sys.path.insert(0, projectRootStr)

from utils.mano.approx import ApproxForwardManoEstimator
from utils.mano.payload import build_mano_sequence, invert_permutation, load_sequence_entry, resolve_input_reorder
from utils.mano.helpers import (
    FINGER_NAMES,
    build_joint_centers,
    build_ring_points,
    build_root_points,
    build_segment_axes,
    build_wrist_frame,
    decode_single_hand_mano,
    joint_name_to_finger,
    segment_name_to_finger,
)
from utils.mano.mano_load import createManoLayer, resolveManoPath
from utils.vis.trimesh_vis import plot_mesh, plot_round_arrow, set_axes_equal, style_3d_axes


GT_MESH_COLOR = "#d9b89c"
GT_POINT_COLOR = "#1f8f6b"
OBSERVED_POINT_COLOR = "#d55252"
PRED_MESH_COLOR = "#b8c5dd"
REPROJ_POINT_COLOR = "#3a7bd5"
LEFT_MESH_COLOR = "#d84b4b"
RIGHT_MESH_COLOR = "#2f9d57"
LEFT_POINT_COLOR = "#d84b4b"
RIGHT_POINT_COLOR = "#2f9d57"
PRED_MESH_COLOR_UNIFIED = "#4f80e1"
LEFT_REPROJ_COLOR = "#4f80e1"
RIGHT_REPROJ_COLOR = "#4f80e1"
OBS_BONE_LEFT = "#a73b3b"
OBS_BONE_RIGHT = "#247a43"
PRED_BONE_COLOR = "#3f6dcc"
RING_LINE_COLOR = "#6b7280"
AXIS_COLORS = {
    "x": "#d84b4b",
    "y": "#2f9d57",
    "z": "#4f80e1",
}
ROOT_AXIS_LINEWIDTH = 3.4
ROOT_AXIS_TIP_SIZE = 34
ROOT_AXIS_ORIGIN_SIZE = 26
ROOT_AXIS_ARROW_RATIO = 0.24
ROOT_AXIS_RADIUS = 0.0023
METRIC_TEXT_FONTSIZE = 17
BIMANUAL_TITLE_FONTSIZE = 18
BIMANUAL_FIGSIZE = (22.6, 5.55)
BIMANUAL_LEFT = 0.004
BIMANUAL_RIGHT = 0.996
BIMANUAL_TOP = 0.94
BIMANUAL_BOTTOM = 0.10
BIMANUAL_WSPACE = 0.002
BIMANUAL_METRIC_Y = 0.026


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MANO sample sequence with 100 semantic points and IK")
    parser.add_argument("--mano-dataset-path", type=str, default="assets/sequence_mano.npz")
    parser.add_argument("--sample-key", type=str, default=None, help="Sample key; if empty, use --sample-index instead")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index used when --sample-key is empty")
    parser.add_argument("--hand-side", type=str, default="both", choices=["left", "right", "both"])
    parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    parser.add_argument("--axis-prior-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")
    parser.add_argument("--sample-index-source-hand", type=str, default="right", choices=["auto", "left", "right"])
    parser.add_argument("--mano-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/sample_sequence_visualization")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--elev", type=float, default=-12.0)
    parser.add_argument("--azim", type=float, default=-96.0)
    parser.add_argument("--zoom", type=float, default=0.62, help="Values below 1 move the camera closer")
    return parser.parse_args()

def _build_metric_caption(frameMetrics: list[dict[str, float]], frameIdx: int) -> str:
    metric = frameMetrics[frameIdx]
    return f"point RMSE {metric['point_rmse_mm']:.2f} mm | vertex RMSE {metric['vertex_rmse_mm']:.2f} mm"


def _build_palette(count: int, hue_offset: float) -> list[np.ndarray]:
    import colorsys

    colors: list[np.ndarray] = []
    for idx in range(count):
        hue = (hue_offset + idx / max(count, 1)) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.96)
        colors.append(np.array([red, green, blue], dtype=np.float32))
    return colors


def _build_finger_palette(hue_offset: float) -> dict[str, np.ndarray]:
    colors = _build_palette(len(FINGER_NAMES), hue_offset)
    return {finger: colors[idx] for idx, finger in enumerate(FINGER_NAMES)}


def _build_ring_joint_frame_data(
    *,
    estimator: ApproxForwardManoEstimator,
    orderedPoints: np.ndarray,
    predVerts: np.ndarray,
    predJoints: np.ndarray,
) -> dict[str, Any]:
    ringPoints = build_ring_points(template=estimator.template, points_ordered=orderedPoints)
    jointCenters = build_joint_centers(template=estimator.template, points_ordered=orderedPoints)
    segmentAxes = build_segment_axes(joint_centers=jointCenters)
    rootPoints = build_root_points(estimator.template, orderedPoints)
    rootOrigin, rootAxes, rootAxisLength = build_wrist_frame(rootPoints)
    return {
        "ring_points": ringPoints,
        "joint_centers": jointCenters,
        "segment_axes": segmentAxes,
        "root_origin": rootOrigin,
        "root_axes": rootAxes,
        "root_axis_length": rootAxisLength,
    }


def _plot_ring_joint_panel(
    ax,
    *,
    data: dict[str, Any],
    fingerColors: dict[str, np.ndarray],
) -> None:
    for segmentName, ring in data["ring_points"].items():
        finger = segment_name_to_finger(segmentName)
        color = fingerColors[finger]
        verts = np.asarray(ring, dtype=np.float32)
        plot_mesh(ax, verts, np.array([[0, 1, 2]], dtype=np.int64), color=color, alpha=0.28)
        ax.scatter(ring[:, 0], ring[:, 1], ring[:, 2], s=11, c=color[None, :], depthshade=False)
    for jointName, center in data["joint_centers"].items():
        finger = joint_name_to_finger(jointName)
        color = fingerColors[finger]
        ax.scatter([center[0]], [center[1]], [center[2]], s=11, c=color[None, :], depthshade=False)
    for segmentName, (start, end) in data["segment_axes"].items():
        finger = segment_name_to_finger(segmentName)
        color = fingerColors[finger]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=1.6,
            alpha=0.9,
        )
    origin = np.asarray(data["root_origin"], dtype=np.float32)
    axisLength = float(data["root_axis_length"]) * 0.85
    ax.scatter([origin[0]], [origin[1]], [origin[2]], s=ROOT_AXIS_ORIGIN_SIZE * 1.2, c="#f2c14e", depthshade=False)
    for axisName, direction in data["root_axes"].items():
        plot_round_arrow(
            ax,
            origin=origin,
            direction=np.asarray(direction, dtype=np.float32),
            length=axisLength,
            radius=ROOT_AXIS_RADIUS,
            color=AXIS_COLORS[axisName],
            alpha=0.95,
            arrow_ratio=ROOT_AXIS_ARROW_RATIO,
        )


def _save_video(frameDir: Path, outputPath: Path, fps: int) -> bool:
    ffmpegPath = shutil.which("ffmpeg")
    if ffmpegPath is None:
        return False
    cmd = [
        ffmpegPath,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frameDir / "frame_%04d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(outputPath),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def _save_gif(frameDir: Path, outputPath: Path, fps: int) -> bool:
    ffmpegPath = shutil.which("ffmpeg")
    if ffmpegPath is None:
        return False
    with tempfile.TemporaryDirectory(prefix="semantic_mano_ik_palette_") as tmpDir:
        palettePath = Path(tmpDir) / "palette.png"
        paletteCmd = [
            ffmpegPath,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frameDir / "frame_%04d.png"),
            "-vf",
            "palettegen=stats_mode=diff",
            str(palettePath),
        ]
        gifCmd = [
            ffmpegPath,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frameDir / "frame_%04d.png"),
            "-i",
            str(palettePath),
            "-lavfi",
            "paletteuse=dither=sierra2_4a",
            str(outputPath),
        ]
        subprocess.run(paletteCmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(gifCmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def _render_sequence_frames(
    *,
    frameDir: Path,
    sampleName: str,
    handSide: str,
    gtVertsSeq: np.ndarray,
    gtPointsSeq: np.ndarray,
    predVertsSeq: np.ndarray,
    reprojPointsSeq: np.ndarray,
    frameMetrics: list[dict[str, float]],
    faces: np.ndarray,
    elev: float,
    azim: float,
    zoom: float,
) -> None:
    frameDir.mkdir(parents=True, exist_ok=True)
    for frameIdx in range(gtVertsSeq.shape[0]):
        framePoints = np.concatenate([gtVertsSeq[frameIdx], predVertsSeq[frameIdx]], axis=0)
        fig = plt.figure(figsize=(12.6, 6.2))
        leftAx = fig.add_subplot(121, projection="3d")
        rightAx = fig.add_subplot(122, projection="3d")

        plot_mesh(leftAx, gtVertsSeq[frameIdx], faces, color=GT_MESH_COLOR, alpha=0.34)
        leftAx.scatter(
            gtPointsSeq[frameIdx, :, 0],
            gtPointsSeq[frameIdx, :, 1],
            gtPointsSeq[frameIdx, :, 2],
            s=10,
            c=GT_POINT_COLOR,
            depthshade=False,
        )
        set_axes_equal(leftAx, framePoints, zoom=zoom)
        style_3d_axes(leftAx)
        leftAx.view_init(elev=elev, azim=azim)

        plot_mesh(rightAx, predVertsSeq[frameIdx], faces, color=PRED_MESH_COLOR, alpha=0.34)
        rightAx.scatter(
            gtPointsSeq[frameIdx, :, 0],
            gtPointsSeq[frameIdx, :, 1],
            gtPointsSeq[frameIdx, :, 2],
            s=10,
            c=OBSERVED_POINT_COLOR,
            depthshade=False,
        )
        rightAx.scatter(
            reprojPointsSeq[frameIdx, :, 0],
            reprojPointsSeq[frameIdx, :, 1],
            reprojPointsSeq[frameIdx, :, 2],
            s=8,
            c=REPROJ_POINT_COLOR,
            depthshade=False,
        )
        set_axes_equal(rightAx, framePoints, zoom=zoom)
        style_3d_axes(
            rightAx,
            None,
        )
        fig.text(
            0.5,
            0.015,
            _build_metric_caption(frameMetrics, frameIdx),
            ha="center",
            va="bottom",
            fontsize=METRIC_TEXT_FONTSIZE,
        )
        rightAx.view_init(elev=elev, azim=azim)

        fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.055, wspace=0.02)
        fig.savefig(frameDir / f"frame_{frameIdx:04d}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _render_bimanual_frames(
    *,
    frameDir: Path,
    sampleName: str,
    leftData: dict[str, Any],
    rightData: dict[str, Any],
    elev: float,
    azim: float,
    zoom: float,
) -> None:
    frameDir.mkdir(parents=True, exist_ok=True)
    numFrames = min(leftData["gt_verts"].shape[0], rightData["gt_verts"].shape[0])
    sharedFingerColors = _build_finger_palette(0.00)
    usableWidth = BIMANUAL_RIGHT - BIMANUAL_LEFT
    colWidth = (usableWidth - BIMANUAL_WSPACE * 3.0) / 4.0
    axesHeight = BIMANUAL_TOP - BIMANUAL_BOTTOM

    def _col_left(colIdx: int) -> float:
        return BIMANUAL_LEFT + colIdx * (colWidth + BIMANUAL_WSPACE)

    for frameIdx in range(numFrames):
        framePoints = np.concatenate(
            [
                leftData["gt_verts"][frameIdx],
                rightData["gt_verts"][frameIdx],
                leftData["pred_verts"][frameIdx],
                rightData["pred_verts"][frameIdx],
            ],
            axis=0,
        )
        fig = plt.figure(figsize=BIMANUAL_FIGSIZE)
        gtMeshAx = fig.add_axes([_col_left(0), BIMANUAL_BOTTOM, colWidth, axesHeight], projection="3d")
        semanticAx = fig.add_axes([_col_left(1), BIMANUAL_BOTTOM, colWidth, axesHeight], projection="3d")
        ringJointAx = fig.add_axes([_col_left(2), BIMANUAL_BOTTOM, colWidth, axesHeight], projection="3d")
        overlayAx = fig.add_axes([_col_left(3), BIMANUAL_BOTTOM, colWidth, axesHeight], projection="3d")

        plot_mesh(gtMeshAx, leftData["gt_verts"][frameIdx], leftData["faces"], color=LEFT_MESH_COLOR, alpha=0.56)
        plot_mesh(gtMeshAx, rightData["gt_verts"][frameIdx], rightData["faces"], color=RIGHT_MESH_COLOR, alpha=0.56)
        set_axes_equal(gtMeshAx, framePoints, zoom=zoom)
        style_3d_axes(gtMeshAx, "GT Mesh")
        gtMeshAx.view_init(elev=elev, azim=azim)

        semanticAx.scatter(
            leftData["gt_points"][frameIdx, :, 0],
            leftData["gt_points"][frameIdx, :, 1],
            leftData["gt_points"][frameIdx, :, 2],
            s=12,
            c=LEFT_POINT_COLOR,
            depthshade=False,
        )
        semanticAx.scatter(
            rightData["gt_points"][frameIdx, :, 0],
            rightData["gt_points"][frameIdx, :, 1],
            rightData["gt_points"][frameIdx, :, 2],
            s=12,
            c=RIGHT_POINT_COLOR,
            depthshade=False,
        )
        set_axes_equal(semanticAx, framePoints, zoom=zoom)
        style_3d_axes(semanticAx, "100 Semantic Points")
        semanticAx.view_init(elev=elev, azim=azim)

        _plot_ring_joint_panel(
            ringJointAx,
            data=leftData["ring_joint"][frameIdx],
            fingerColors=sharedFingerColors,
        )
        _plot_ring_joint_panel(
            ringJointAx,
            data=rightData["ring_joint"][frameIdx],
            fingerColors=sharedFingerColors,
        )
        set_axes_equal(ringJointAx, framePoints, zoom=zoom)
        style_3d_axes(ringJointAx, "Single-Step Ring Joint")
        ringJointAx.view_init(elev=elev, azim=azim)

        plot_mesh(overlayAx, leftData["gt_verts"][frameIdx], leftData["faces"], color=LEFT_MESH_COLOR, alpha=0.22)
        plot_mesh(overlayAx, rightData["gt_verts"][frameIdx], rightData["faces"], color=RIGHT_MESH_COLOR, alpha=0.22)
        plot_mesh(overlayAx, leftData["pred_verts"][frameIdx], leftData["faces"], color=PRED_MESH_COLOR_UNIFIED, alpha=0.62)
        plot_mesh(overlayAx, rightData["pred_verts"][frameIdx], rightData["faces"], color=PRED_MESH_COLOR_UNIFIED, alpha=0.62)
        set_axes_equal(overlayAx, framePoints, zoom=zoom)
        style_3d_axes(overlayAx, "Single-Step Mesh+GT Mesh")
        overlayAx.view_init(elev=elev, azim=azim)

        for axis in (gtMeshAx, semanticAx, ringJointAx, overlayAx):
            axis.title.set_fontsize(BIMANUAL_TITLE_FONTSIZE)

        fig.text(
            0.5,
            BIMANUAL_METRIC_Y,
            f"L {_build_metric_caption(leftData['frame_metrics'], frameIdx)}    |    "
            f"R {_build_metric_caption(rightData['frame_metrics'], frameIdx)}",
            ha="center",
            va="bottom",
            fontsize=METRIC_TEXT_FONTSIZE,
        )
        fig.savefig(frameDir / f"frame_{frameIdx:04d}.png", dpi=180)
        plt.close(fig)


def _run_single_hand(
    *,
    manoLayer,
    leftEstimator: ApproxForwardManoEstimator,
    rightEstimator: ApproxForwardManoEstimator,
    manoSeq: np.ndarray,
    handSide: str,
    sampleIndices: np.ndarray,
    sampleName: str,
    sourceHandSide: str,
    outputDir: Path,
    elev: float,
    azim: float,
    fps: int,
    skipVideo: bool,
    persistOutputs: bool,
) -> dict[str, Any]:
    estimator = leftEstimator if handSide == "left" else rightEstimator
    reorderIndex, detectedSourceHand = resolve_input_reorder(
        sampleIndices=sampleIndices,
        targetHandSide=handSide,
        sourceHandSide=sourceHandSide,
        leftEstimator=leftEstimator,
        rightEstimator=rightEstimator,
    )
    inverseReorder = invert_permutation(reorderIndex)
    faces = manoLayer[handSide].faces.astype(np.int64)
    numFrames = manoSeq.shape[0]

    gtVertsSeq = []
    gtJointsSeq = []
    gtPointsSeq = []
    predManoSeq = []
    predVertsSeq = []
    predJointsSeq = []
    reprojPointsSeq = []
    ringJointSeq = []
    frameMetrics: list[dict[str, float]] = []

    with torch.no_grad():
        for frameIdx in range(numFrames):
            gtMano = manoSeq[frameIdx]
            gtVerts, gtJoints = decode_single_hand_mano(manoLayer=manoLayer, manoParams=gtMano, handSide=handSide)
            gtPoints = gtVerts[sampleIndices]
            orderedPoints = gtPoints[reorderIndex]
            estimate = estimator.estimate(torch.from_numpy(orderedPoints).float())
            predMano = estimate.fullMano.detach().cpu().numpy().reshape(61).astype(np.float32)
            predVerts, predJoints = decode_single_hand_mano(manoLayer=manoLayer, manoParams=predMano, handSide=handSide)
            predPointsOrdered = predVerts[np.asarray(estimator.template.indexOrder, dtype=np.int64)]
            reprojPoints = predPointsOrdered[inverseReorder]
            ringJointSeq.append(
                _build_ring_joint_frame_data(
                    estimator=estimator,
                    orderedPoints=orderedPoints,
                    predVerts=predVerts,
                    predJoints=predJoints,
                )
            )

            pointError = np.linalg.norm(reprojPoints - gtPoints, axis=1)
            vertexError = np.linalg.norm(predVerts - gtVerts, axis=1)
            frameMetrics.append(
                {
                    "frame_index": frameIdx,
                    "point_mean_mm": float(np.mean(pointError) * 1000.0),
                    "point_rmse_mm": float(np.sqrt(np.mean(np.square(pointError))) * 1000.0),
                    "point_max_mm": float(np.max(pointError) * 1000.0),
                    "vertex_mean_mm": float(np.mean(vertexError) * 1000.0),
                    "vertex_rmse_mm": float(np.sqrt(np.mean(np.square(vertexError))) * 1000.0),
                    "vertex_max_mm": float(np.max(vertexError) * 1000.0),
                    "fallback_count": int(estimate.fallbackCount),
                }
            )

            gtVertsSeq.append(gtVerts)
            gtJointsSeq.append(gtJoints)
            gtPointsSeq.append(gtPoints)
            predManoSeq.append(predMano)
            predVertsSeq.append(predVerts)
            predJointsSeq.append(predJoints)
            reprojPointsSeq.append(reprojPoints.astype(np.float32))

    gtVertsArr = np.stack(gtVertsSeq, axis=0).astype(np.float32)
    gtJointsArr = np.stack(gtJointsSeq, axis=0).astype(np.float32)
    gtPointsArr = np.stack(gtPointsSeq, axis=0).astype(np.float32)
    predManoArr = np.stack(predManoSeq, axis=0).astype(np.float32)
    predVertsArr = np.stack(predVertsSeq, axis=0).astype(np.float32)
    predJointsArr = np.stack(predJointsSeq, axis=0).astype(np.float32)
    reprojPointsArr = np.stack(reprojPointsSeq, axis=0).astype(np.float32)

    videoSaved = False
    gifSaved = False

    summary = {
        "sample_name": sampleName,
        "hand_side": handSide,
        "num_frames": int(numFrames),
        "source_hand_side": detectedSourceHand,
        "sample_index_order": sampleIndices.astype(np.int64).tolist(),
        "point_mean_mm": float(np.mean([item["point_mean_mm"] for item in frameMetrics])),
        "point_rmse_mm": float(np.mean([item["point_rmse_mm"] for item in frameMetrics])),
        "point_max_mm": float(np.max([item["point_max_mm"] for item in frameMetrics])),
        "vertex_mean_mm": float(np.mean([item["vertex_mean_mm"] for item in frameMetrics])),
        "vertex_rmse_mm": float(np.mean([item["vertex_rmse_mm"] for item in frameMetrics])),
        "vertex_max_mm": float(np.max([item["vertex_max_mm"] for item in frameMetrics])),
        "fallback_count_total": int(sum(item["fallback_count"] for item in frameMetrics)),
        "video_saved": bool(videoSaved),
        "gif_saved": bool(gifSaved),
    }
    if persistOutputs:
        outputDir.mkdir(parents=True, exist_ok=True)
        np.save(outputDir / "gt_mano_sequence.npy", manoSeq.astype(np.float32))
        np.save(outputDir / "gt_vertices_sequence.npy", gtVertsArr)
        np.save(outputDir / "gt_joints_sequence.npy", gtJointsArr)
        np.save(outputDir / "semantic_points_100_sequence.npy", gtPointsArr)
        np.save(outputDir / "ik_mano_sequence.npy", predManoArr)
        np.save(outputDir / "ik_vertices_sequence.npy", predVertsArr)
        np.save(outputDir / "ik_joints_sequence.npy", predJointsArr)
        np.save(outputDir / "ik_reprojected_points_sequence.npy", reprojPointsArr)

        frameDir = outputDir / "frames"
        _render_sequence_frames(
            frameDir=frameDir,
            sampleName=sampleName,
            handSide=handSide,
            gtVertsSeq=gtVertsArr,
            gtPointsSeq=gtPointsArr,
            predVertsSeq=predVertsArr,
            reprojPointsSeq=reprojPointsArr,
            frameMetrics=frameMetrics,
            faces=faces,
            elev=elev,
            azim=azim,
            zoom=1.0,
        )
        videoPath = outputDir / "sequence_visualization.mp4"
        videoSaved = False if skipVideo else _save_video(frameDir, videoPath, fps)
        gifPath = outputDir / "sequence_visualization.gif"
        gifSaved = False if skipVideo else _save_gif(frameDir, gifPath, fps)
        if videoSaved or gifSaved:
            shutil.rmtree(frameDir, ignore_errors=True)

        (outputDir / "frame_metrics.json").write_text(
            json.dumps(frameMetrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (outputDir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        payload = {
            "sample_name": sampleName,
            "hand_side": handSide,
            "source_hand_side": detectedSourceHand,
            "sample_index_order": sampleIndices.astype(np.int64),
            "gt_mano_sequence": manoSeq.astype(np.float32),
            "semantic_points_100_sequence": gtPointsArr,
            "ik_mano_sequence": predManoArr,
            "ik_reprojected_points_sequence": reprojPointsArr,
        }
        np.save(outputDir / "sequence_payload.npy", payload, allow_pickle=True)
    return {
        "summary": summary,
        "gt_verts": gtVertsArr,
        "gt_points": gtPointsArr,
        "pred_verts": predVertsArr,
        "reproj_points": reprojPointsArr,
        "faces": faces,
        "ring_joint": ringJointSeq,
        "frame_metrics": frameMetrics,
        "output_dir": outputDir if persistOutputs else None,
    }


def main() -> None:
    args = _parse_args()
    manoDatasetPath = Path(args.mano_dataset_path)
    outputRoot = Path(args.output_dir)
    sampleIndices = np.load(str(args.sample_index_path)).astype(np.int64).reshape(-1)
    sampleKey, entry, keys = load_sequence_entry(manoDatasetPath, args.sample_key, args.sample_index)
    handSides = ["left", "right"] if args.hand_side == "both" else [args.hand_side]
    manoPath = resolveManoPath(manoPath=args.mano_path, projectRoot=PROJECT_ROOT)

    manoLayer = createManoLayer(modelPath=str(manoPath), device="cpu")
    for side in ("left", "right"):
        manoLayer[side].eval()
        for param in manoLayer[side].parameters():
            param.requires_grad_(False)
    leftEstimator = ApproxForwardManoEstimator(
        manoPath=str(manoPath),
        handSide="left",
        device="cpu",
        axisPriorPath=str(args.axis_prior_path),
    )
    rightEstimator = ApproxForwardManoEstimator(
        manoPath=str(manoPath),
        handSide="right",
        device="cpu",
        axisPriorPath=str(args.axis_prior_path),
    )

    manifest = {
        "mano_dataset_path": str(manoDatasetPath.resolve()),
        "sample_key": sampleKey,
        "available_sample_keys": keys,
        "sample_index": int(args.sample_index),
        "hand_sides": handSides,
        "source_hand_side": args.sample_index_source_hand,
        "fps": int(args.fps),
        "outputs": {},
    }
    handResults: dict[str, dict[str, Any]] = {}
    persistSingleHandOutputs = len(handSides) == 1
    for handSide in handSides:
        manoSeq = build_mano_sequence(entry, hand_side=handSide)
        if args.max_frames is not None:
            manoSeq = manoSeq[: args.max_frames]
        sampleName = f"{manoDatasetPath.stem}_sample_{sampleKey}"
        handOutputDir = outputRoot / sampleName / handSide
        result = _run_single_hand(
            manoLayer=manoLayer,
            leftEstimator=leftEstimator,
            rightEstimator=rightEstimator,
            manoSeq=manoSeq,
            handSide=handSide,
            sampleIndices=sampleIndices,
            sampleName=sampleName,
            sourceHandSide=args.sample_index_source_hand,
            outputDir=handOutputDir,
            elev=args.elev,
            azim=args.azim,
            fps=args.fps,
            skipVideo=args.skip_video,
            persistOutputs=persistSingleHandOutputs,
        )
        summary = result["summary"]
        handResults[handSide] = result
        if persistSingleHandOutputs:
            manifest["outputs"][handSide] = {
                "dir": str(handOutputDir.resolve()),
                "summary": summary,
            }

    if set(handSides) == {"left", "right"}:
        sampleName = f"{manoDatasetPath.stem}_sample_{sampleKey}"
        bimanualDir = outputRoot / sampleName / "bimanual"
        frameDir = bimanualDir / "frames"
        _render_bimanual_frames(
            frameDir=frameDir,
            sampleName=sampleName,
            leftData=handResults["left"],
            rightData=handResults["right"],
            elev=args.elev,
            azim=args.azim,
            zoom=args.zoom,
        )
        videoPath = bimanualDir / "sequence_visualization.mp4"
        bimanualDir.mkdir(parents=True, exist_ok=True)
        videoSaved = False if args.skip_video else _save_video(frameDir, videoPath, args.fps)
        gifPath = bimanualDir / "sequence_visualization.gif"
        gifSaved = False if args.skip_video else _save_gif(frameDir, gifPath, args.fps)
        if videoSaved or gifSaved:
            shutil.rmtree(frameDir, ignore_errors=True)
        bimanualSummary = {
            "sample_name": sampleName,
            "hand_side": "both",
            "num_frames": int(min(handResults["left"]["gt_verts"].shape[0], handResults["right"]["gt_verts"].shape[0])),
            "camera": {"elev": float(args.elev), "azim": float(args.azim)},
            "video_saved": bool(videoSaved),
            "gif_saved": bool(gifSaved),
            "left_point_rmse_mm": float(handResults["left"]["summary"]["point_rmse_mm"]),
            "right_point_rmse_mm": float(handResults["right"]["summary"]["point_rmse_mm"]),
            "left_vertex_rmse_mm": float(handResults["left"]["summary"]["vertex_rmse_mm"]),
            "right_vertex_rmse_mm": float(handResults["right"]["summary"]["vertex_rmse_mm"]),
        }
        (bimanualDir / "summary.json").write_text(json.dumps(bimanualSummary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["outputs"]["bimanual"] = {
            "dir": str(bimanualDir.resolve()),
            "summary": bimanualSummary,
        }

    outputRoot.mkdir(parents=True, exist_ok=True)
    (outputRoot / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] saved sequence outputs to: {outputRoot}")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
