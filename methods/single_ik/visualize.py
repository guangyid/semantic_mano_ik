#!/usr/bin/env python3
"""Single-step IK visualizations: anchor groups, wrist frame, and ring/joint diagnostics."""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mano.anchors import SMPLX_SEGMENT_ORDER, buildAnchorScene, buildFlatHandAnchorTemplate, loadFlatHandMano, selectFingerAndPalmGroups
from utils.mano.approx import ApproxForwardManoEstimator
from utils.mano.helpers import build_root_points, build_wrist_frame, decode_single_hand_mano
from utils.mano.mano_load import createManoLayer, resolveManoPath
from utils.mano.payload import load_payload_file, load_single_hand_points, resolve_input_reorder
from utils.mano.reorder import resolveApproxIkInputOrders
from utils.vis.trimesh_vis import add_axes, add_cylinder, add_sphere, build_hand_mesh


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
JOINT_LEVELS = ["joint_1", "joint_2", "joint_3", "tip"]
FINGER_JOINT_GROUPS = OrderedDict([
    ("thumb", (13, 14, 15)),
    ("index", (1, 2, 3)),
    ("middle", (4, 5, 6)),
    ("ring", (10, 11, 12)),
    ("pinky", (7, 8, 9)),
])
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "samples" / "ring_joint_demo.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
POINT_COLOR = np.array([240, 190, 80, 235], dtype=np.uint8)
ROOT_COLOR = np.array([255, 90, 90, 255], dtype=np.uint8)
FRAME_MESH_COLOR = np.array([245, 205, 165, 120], dtype=np.uint8)
AXIS_PRIOR_POINT_ORDER = (
    "ring_mid",
    "ring_pos",
    "ring_neg",
    "prox_pos",
    "prox_neg",
    "dist_pos",
    "dist_neg",
)


def _build_palette(count: int, hue_offset: float) -> list[np.ndarray]:
    import colorsys

    colors: list[np.ndarray] = []
    for idx in range(count):
        hue = (hue_offset + idx / max(count, 1)) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors.append(np.array([int(255 * red), int(255 * green), int(255 * blue), 255], dtype=np.uint8))
    return colors


def _normalize_np(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1.0e-8:
        return np.zeros_like(vec)
    return vec / norm


def _project_to_plane_np(vec: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = _normalize_np(axis)
    return vec - np.dot(vec, axis) * axis


def _build_axis_basis(axis: np.ndarray, y_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_proj = _project_to_plane_np(y_hint, axis)
    if np.linalg.norm(y_proj) < 1.0e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(axis, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        y_proj = _project_to_plane_np(fallback, axis)
    y_axis = _normalize_np(y_proj)
    z_axis = _normalize_np(np.cross(axis, y_axis))
    return y_axis, z_axis


def _segment_endpoint_names(segment_name: str) -> tuple[str, str]:
    finger_name = segment_name.split("_")[0]
    segment_idx = int(segment_name.rsplit("_", 1)[-1])
    proximal_name = f"{finger_name}_joint_{segment_idx}"
    distal_name = f"{finger_name}_tip" if segment_idx == 3 else f"{finger_name}_joint_{segment_idx + 1}"
    return proximal_name, distal_name


def _compute_segment_axis_prior(template, verts: np.ndarray, segment_name: str) -> Dict[str, float]:
    ring = template.segmentRingMap[segment_name]
    proximal_name, distal_name = _segment_endpoint_names(segment_name)
    prox_pair = template.jointPairMap[proximal_name]
    dist_pair = template.jointPairMap[distal_name]

    prox_center = 0.5 * (verts[int(prox_pair["pos"])] + verts[int(prox_pair["neg"])])
    dist_center = 0.5 * (verts[int(dist_pair["pos"])] + verts[int(dist_pair["neg"])])
    axis = _normalize_np(dist_center - prox_center)

    y_hint = (
        verts[int(ring["pos"])] - verts[int(ring["neg"])]
        + verts[int(prox_pair["pos"])] - verts[int(prox_pair["neg"])]
        + verts[int(dist_pair["pos"])] - verts[int(dist_pair["neg"])]
    )
    y_axis, z_axis = _build_axis_basis(axis=axis, y_hint=y_hint)

    point_map = {
        "ring_mid": verts[int(ring["mid"])],
        "ring_pos": verts[int(ring["pos"])],
        "ring_neg": verts[int(ring["neg"])],
        "prox_pos": verts[int(prox_pair["pos"])],
        "prox_neg": verts[int(prox_pair["neg"])],
        "dist_pos": verts[int(dist_pair["pos"])],
        "dist_neg": verts[int(dist_pair["neg"])],
    }
    prior: Dict[str, float] = {}
    for name in AXIS_PRIOR_POINT_ORDER:
        point = point_map[name]
        rel = point - prox_center
        rel = rel - np.dot(rel, axis) * axis
        y_val = float(np.dot(rel, y_axis))
        z_val = float(np.dot(rel, z_axis))
        prior[name] = float(np.arctan2(z_val, y_val))
    return prior


def _build_axis_prior_for_side(*, mano_path: str, hand_side: str) -> Dict[str, Dict[str, float]]:
    verts, joints, faces = loadFlatHandMano(manoPath=mano_path, handSide=hand_side)
    template = buildFlatHandAnchorTemplate(verts=verts, joints=joints, faces=faces)
    prior: Dict[str, Dict[str, float]] = {}
    for segment_name in template.segmentRingMap.keys():
        prior[segment_name] = _compute_segment_axis_prior(template, verts, segment_name)
    return prior


def _add_triangle(scene: trimesh.Scene, points: np.ndarray, color: np.ndarray, name: str) -> None:
    tri = trimesh.Trimesh(vertices=points.astype(np.float32), faces=np.array([[0, 1, 2]], dtype=np.int64), process=False)
    tri.visual.vertex_colors = np.tile(color[None, :], (tri.vertices.shape[0], 1))
    scene.add_geometry(tri, node_name=name)


def _normalize_points_array(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape == (100, 3):
        return arr[None, None, ...]
    if arr.shape == (1, 100, 3):
        return arr[None, ...]
    if arr.shape == (1, 1, 100, 3):
        return arr
    raise ValueError(f"{name} must have shape [100,3], [1,100,3], or [1,1,100,3], got {arr.shape}")


def _normalize_mano_array(mano_params: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(mano_params, dtype=np.float32)
    if arr.shape == (122,):
        return arr[None, None, ...]
    if arr.shape == (1, 122):
        return arr[None, ...]
    if arr.shape == (1, 1, 122):
        return arr
    raise ValueError(f"{name} must have shape [122], [1,122], or [1,1,122], got {arr.shape}")


def _normalize_sample_payload(payload: dict[str, Any]) -> dict[str, Any]:
    left_key = "pred_left_points_world" if "pred_left_points_world" in payload else "left_points_world"
    right_key = "pred_right_points_world" if "pred_right_points_world" in payload else "right_points_world"
    mano_key = "single_ik_mano_params"
    if left_key not in payload or right_key not in payload or mano_key not in payload:
        raise ValueError("Sample file is missing required fields: left/right points or single_ik_mano_params")
    normalized = {
        "pred_left_points_world": _normalize_points_array(payload[left_key], left_key),
        "pred_right_points_world": _normalize_points_array(payload[right_key], right_key),
        "single_ik_mano_params": _normalize_mano_array(payload[mano_key], mano_key),
    }
    if "sample_index_order" in payload:
        normalized["sample_index_order"] = np.asarray(payload["sample_index_order"], dtype=np.int64).reshape(-1)
    if "sample_name" in payload:
        normalized["sample_name"] = str(payload["sample_name"])
    return normalized


def _load_local_sample(sample_path: Path) -> dict[str, Any]:
    return _normalize_sample_payload(load_payload_file(sample_path))


def _load_diagnostic_sample(diagnostic_path: str, sample_stem: str) -> dict[str, np.ndarray]:
    data = np.load(diagnostic_path, allow_pickle=True).item()
    stems = data.get("sample_stems")
    if stems is None:
        raise ValueError("mano_diagnostics.npy is missing sample_stems")
    if sample_stem not in stems:
        raise ValueError(f"sample_stem was not found: {sample_stem}")
    sample_index = int(np.where(stems == sample_stem)[0][0])
    return {key: value[sample_index:sample_index + 1] if isinstance(value, np.ndarray) else value for key, value in data.items()}


def _load_input_sample(*, sample_path: Path | None, diagnostics_path: Path | None, sample_stem: str | None) -> dict[str, Any]:
    if sample_path is not None:
        return _load_local_sample(sample_path)
    if diagnostics_path is None:
        raise ValueError("Missing input source: provide --sample-path or provide both --mano-diagnostics and --sample-stem")
    if not sample_stem:
        raise ValueError("--sample-stem is required when using --mano-diagnostics")
    return _load_diagnostic_sample(str(diagnostics_path), sample_stem)


def _resolve_sample_index_order(*, sample: dict[str, Any], sample_index_path: Path) -> np.ndarray:
    if "sample_index_order" in sample:
        return np.asarray(sample["sample_index_order"], dtype=np.int64)
    return np.load(str(sample_index_path)).astype(np.int64)


def _decode_hand_verts(*, mano_layer, mano_params: torch.Tensor, hand_side: str) -> torch.Tensor:
    side_params = mano_params[..., :61] if hand_side == "left" else mano_params[..., 61:]
    batch_size, time_count, _ = side_params.shape
    flat = side_params.reshape(batch_size * time_count, 61)
    output = mano_layer[hand_side](
        global_orient=flat[:, 0:3],
        hand_pose=flat[:, 3:48],
        betas=flat[:, 51:61],
        transl=flat[:, 48:51],
    )
    return output.vertices.reshape(batch_size, time_count, 778, 3)


def _decode_hand_joints(*, mano_layer, mano_params: torch.Tensor, hand_side: str) -> torch.Tensor:
    side_params = mano_params[..., :61] if hand_side == "left" else mano_params[..., 61:]
    batch_size, time_count, _ = side_params.shape
    flat = side_params.reshape(batch_size * time_count, 61)
    output = mano_layer[hand_side](
        global_orient=flat[:, 0:3],
        hand_pose=flat[:, 3:48],
        betas=flat[:, 51:61],
        transl=flat[:, 48:51],
    )
    return output.joints.reshape(batch_size, time_count, 16, 3)


def _build_hand_geometry(*, mano_layer, mano_params: torch.Tensor, hand_side: str) -> trimesh.Trimesh:
    verts = _decode_hand_verts(mano_layer=mano_layer, mano_params=mano_params, hand_side=hand_side)
    mesh = trimesh.Trimesh(
        vertices=verts[0, -1].detach().cpu().numpy(),
        faces=mano_layer[hand_side].faces.astype(np.int64),
        process=False,
    )
    mesh.visual.vertex_colors = np.tile(np.array([255, 200, 150, 160], dtype=np.uint8)[None, :], (mesh.vertices.shape[0], 1))
    return mesh


def _normalize(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp(min=eps)


def _build_mesh_fingertips(*, verts: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    distal_indices = [group[-1] for group in FINGER_JOINT_GROUPS.values()]
    proximal_indices = [group[-2] for group in FINGER_JOINT_GROUPS.values()]
    distal = joints[distal_indices]
    proximal = joints[proximal_indices]
    directions = _normalize(distal - proximal)
    dist = torch.cdist(verts.unsqueeze(0), distal.unsqueeze(0)).squeeze(0)
    assignment = torch.argmin(dist, dim=-1)
    tips: list[torch.Tensor] = []
    for finger_idx in range(len(distal_indices)):
        mask = assignment == finger_idx
        if not bool(mask.any().item()):
            tips.append(distal[finger_idx])
            continue
        rel = verts[mask] - distal[finger_idx]
        proj = (rel * directions[finger_idx]).sum(dim=-1)
        dist_norm = rel.norm(dim=-1)
        score = proj - 0.2 * dist_norm
        tips.append(verts[mask][torch.argmax(score)])
    return torch.stack(tips, dim=0)


def _build_joint_centers(*, template, points_ordered: np.ndarray) -> dict[str, np.ndarray]:
    index_to_offset = {int(index): offset for offset, index in enumerate(template.indexOrder)}
    centers: dict[str, np.ndarray] = {}
    for finger in FINGER_NAMES:
        for level in JOINT_LEVELS:
            pair_name = f"{finger}_{level}"
            pair = template.jointPairMap[pair_name]
            pos = points_ordered[index_to_offset[int(pair["pos"])]]
            neg = points_ordered[index_to_offset[int(pair["neg"])]]
            centers[pair_name] = 0.5 * (pos + neg)
    return centers


def _build_mesh_joint_centers(*, mesh_joints: np.ndarray, mesh_tips: np.ndarray) -> dict[str, np.ndarray]:
    centers: dict[str, np.ndarray] = {}
    for finger_name, joint_ids in FINGER_JOINT_GROUPS.items():
        for idx, joint_id in enumerate(joint_ids):
            centers[f"{finger_name}_joint_{idx + 1}"] = mesh_joints[joint_id]
        centers[f"{finger_name}_tip"] = mesh_tips[FINGER_NAMES.index(finger_name)]
    return centers


def _build_ring_points(*, template, points_ordered: np.ndarray) -> dict[str, np.ndarray]:
    index_to_offset = {int(index): offset for offset, index in enumerate(template.indexOrder)}
    ring_points: dict[str, np.ndarray] = {}
    for segment_name, ring_map in template.segmentRingMap.items():
        ring_points[segment_name] = np.stack(
            [
                points_ordered[index_to_offset[int(ring_map["mid"])]],
                points_ordered[index_to_offset[int(ring_map["pos"])]],
                points_ordered[index_to_offset[int(ring_map["neg"])]],
            ],
            axis=0,
        )
    return ring_points


def _build_segment_axes(*, joint_centers: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    axes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for finger in FINGER_NAMES:
        for seg_idx in range(1, 4):
            proximal = joint_centers[f"{finger}_joint_{seg_idx}"]
            distal = joint_centers[f"{finger}_tip"] if seg_idx == 3 else joint_centers[f"{finger}_joint_{seg_idx + 1}"]
            axes[f"{finger}_segment_{seg_idx}"] = (proximal, distal)
    return axes


def _darken_color(color: np.ndarray, factor: float = 0.35) -> np.ndarray:
    rgb = np.clip(color[:3].astype(np.float32) * float(factor), 0.0, 255.0)
    return np.array([int(rgb[0]), int(rgb[1]), int(rgb[2]), int(color[3])], dtype=np.uint8)


def _render_hand(
    *,
    scene: trimesh.Scene,
    hand_side: str,
    points_world: np.ndarray,
    reorder_index: np.ndarray,
    estimator: ApproxForwardManoEstimator,
    mano_layer,
    mano_params: torch.Tensor,
    ring_hue_offset: float,
    joint_hue_offset: float,
    point_radius: float,
    joint_radius: float,
    axis_radius: float,
    mesh_joint_radius: float,
    link_radius: float,
) -> None:
    points_ordered = points_world[reorder_index]
    ring_points = _build_ring_points(template=estimator.template, points_ordered=points_ordered)
    joint_centers = _build_joint_centers(template=estimator.template, points_ordered=points_ordered)
    segment_axes = _build_segment_axes(joint_centers=joint_centers)

    segment_names = list(SMPLX_SEGMENT_ORDER)
    ring_colors = _build_palette(len(segment_names), ring_hue_offset)
    joint_names = [f"{finger}_{level}" for finger in FINGER_NAMES for level in JOINT_LEVELS]
    joint_colors = _build_palette(len(joint_names), joint_hue_offset)
    joint_color_map = {name: color for name, color in zip(joint_names, joint_colors)}

    for seg_idx, segment_name in enumerate(segment_names):
        ring_color = ring_colors[seg_idx]
        ring = ring_points[segment_name]
        _add_triangle(scene, ring, ring_color, f"{hand_side}_{segment_name}_ring")
        for point_idx, point in enumerate(ring):
            add_sphere(scene, point, point_radius, ring_color, f"{hand_side}_{segment_name}_ring_point_{point_idx}")
        start, end = segment_axes[segment_name]
        add_cylinder(scene, start, end, axis_radius, ring_color, f"{hand_side}_{segment_name}_axis")

    mesh_joints = _decode_hand_joints(mano_layer=mano_layer, mano_params=mano_params, hand_side=hand_side)[0, -1]
    mesh_verts = _decode_hand_verts(mano_layer=mano_layer, mano_params=mano_params, hand_side=hand_side)[0, -1]
    mesh_tips = _build_mesh_fingertips(verts=mesh_verts, joints=mesh_joints)
    mesh_joint_centers = _build_mesh_joint_centers(mesh_joints=mesh_joints.detach().cpu().numpy(), mesh_tips=mesh_tips.detach().cpu().numpy())

    for joint_name, center in joint_centers.items():
        color = joint_color_map[joint_name]
        add_sphere(scene, center, joint_radius, color, f"{hand_side}_{joint_name}_center_obs")
        mesh_center = mesh_joint_centers[joint_name]
        add_sphere(scene, mesh_center, mesh_joint_radius, _darken_color(color, factor=0.45), f"{hand_side}_{joint_name}_center_mesh")
        add_cylinder(scene, center, mesh_center, link_radius, np.array([120, 120, 120, 200], dtype=np.uint8), f"{hand_side}_{joint_name}_link")

    scene.add_geometry(_build_hand_geometry(mano_layer=mano_layer, mano_params=mano_params, hand_side=hand_side), node_name=f"{hand_side}_mano_mesh")


def _run_anchor_groups(args) -> None:
    verts, joints, faces = loadFlatHandMano(manoPath=args.mano_path, handSide=args.hand_side)
    groups = selectFingerAndPalmGroups(verts=verts, joints=joints, faces=faces)
    scene = buildAnchorScene(verts=verts, faces=faces, groups=groups, pointRadius=float(args.point_radius))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))
    finger_count = sum(len(group.indices) for group in groups if "_" in group.name and not group.name.startswith(("palm_", "wrist_")))
    wrist_count = sum(len(group.indices) for group in groups if group.name.startswith("wrist_"))
    palm_count = sum(len(group.indices) for group in groups if group.name.startswith("palm_"))
    print(f"Exported anchor glb to: {output_path}")
    print(f"finger_points={finger_count}, wrist_points={wrist_count}, palm_surface_points={palm_count}, total_points={finger_count + wrist_count + palm_count}")


def _run_axis_prior(args) -> None:
    output_path = Path(args.output_path)
    prior = {
        "left": _build_axis_prior_for_side(mano_path=args.mano_path, hand_side="left"),
        "right": _build_axis_prior_for_side(mano_path=args.mano_path, hand_side="right"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), prior, allow_pickle=True)
    print(f"[OK] saved axis prior: {output_path}")


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


def _build_mano_layer(mano_path: Path):
    mano_layer = createManoLayer(modelPath=str(mano_path), device="cpu")
    for side in ("left", "right"):
        mano_layer[side].eval()
        for param in mano_layer[side].parameters():
            param.requires_grad_(False)
    return mano_layer


def _run_wrist_frame(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mano_path = resolveManoPath(manoPath=args.mano_path, projectRoot=PROJECT_ROOT)
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
    ordered_points = input_points[reorder_index]
    root_points = build_root_points(estimator.template, ordered_points)
    origin, axes, axis_length = build_wrist_frame(root_points)

    scene = trimesh.Scene()
    for idx, point in enumerate(input_points):
        add_sphere(scene, point, args.point_radius * 0.65, POINT_COLOR, f"point_{idx:03d}")
    for key, point in root_points.items():
        add_sphere(scene, point, args.point_radius * 1.3, ROOT_COLOR, f"root_{key}")
    add_axes(scene, origin=origin, axes=axes, axisLength=axis_length, radius=args.axis_radius, prefix="wrist_frame")

    estimate = estimator.estimate(torch.from_numpy(ordered_points).float())
    mano_layer = _build_mano_layer(mano_path)
    pred_verts, _ = decode_single_hand_mano(
        manoLayer=mano_layer,
        manoParams=estimate.fullMano.detach().cpu().numpy().reshape(61),
        handSide=args.hand_side,
    )
    scene.add_geometry(build_hand_mesh(pred_verts, mano_layer[args.hand_side].faces, color=FRAME_MESH_COLOR), node_name="pred_mesh")

    (output_dir / "wrist_frame.glb").write_bytes(scene.export(file_type="glb"))
    (output_dir / "wrist_frame.json").write_text(
        json.dumps(
            {
                "hand_side": args.hand_side,
                "source_hand_side": detected_source_hand,
                "sample_index_order": meta.get("sample_index_order", sample_indices).tolist(),
                "origin": origin.tolist(),
                "x_axis": axes["x"].tolist(),
                "y_axis": axes["y"].tolist(),
                "z_axis": axes["z"].tolist(),
                "axis_length": axis_length,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] saved wrist frame to: {output_dir}")


def _run_ring_joint(args) -> None:
    sample_path = Path(args.sample_path) if args.sample_path else None
    if sample_path is None and DEFAULT_SAMPLE_PATH.is_file():
        sample_path = DEFAULT_SAMPLE_PATH
    diagnostics_path = Path(args.mano_diagnostics) if args.mano_diagnostics else None
    sample_index_path = Path(args.sample_index_path)
    mano_path = resolveManoPath(manoPath=args.mano_path, projectRoot=PROJECT_ROOT)

    sample = _load_input_sample(sample_path=sample_path, diagnostics_path=diagnostics_path, sample_stem=args.sample_stem)
    pred_left_points = sample["pred_left_points_world"][0, -1]
    pred_right_points = sample["pred_right_points_world"][0, -1]
    single_ik_mano = torch.from_numpy(sample["single_ik_mano_params"]).float()
    sample_name = str(sample.get("sample_name") or args.sample_stem or "ring_joint_demo")
    output_glb = Path(args.output_glb) if args.output_glb else (DEFAULT_OUTPUT_DIR / f"{sample_name}_ring_joint.glb")
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    sample_indices = _resolve_sample_index_order(sample=sample, sample_index_path=sample_index_path)
    left_estimator = ApproxForwardManoEstimator(manoPath=str(mano_path), handSide="left", device="cpu")
    right_estimator = ApproxForwardManoEstimator(manoPath=str(mano_path), handSide="right", device="cpu")
    left_reorder, right_reorder, _ = resolveApproxIkInputOrders(sampleIndices=sample_indices, leftEstimator=left_estimator, rightEstimator=right_estimator)

    mano_layer = createManoLayer(modelPath=str(mano_path), device="cpu")
    for side in ("left", "right"):
        mano_layer[side].eval()
        for param in mano_layer[side].parameters():
            param.requires_grad_(False)

    scene = trimesh.Scene()
    if args.hand_side in ("left", "both"):
        _render_hand(
            scene=scene,
            hand_side="left",
            points_world=pred_left_points,
            reorder_index=left_reorder.numpy(),
            estimator=left_estimator,
            mano_layer=mano_layer,
            mano_params=single_ik_mano,
            ring_hue_offset=0.05,
            joint_hue_offset=0.15,
            point_radius=args.point_radius,
            joint_radius=args.joint_radius,
            axis_radius=args.axis_radius,
            mesh_joint_radius=args.mesh_joint_radius,
            link_radius=args.link_radius,
        )
    if args.hand_side in ("right", "both"):
        _render_hand(
            scene=scene,
            hand_side="right",
            points_world=pred_right_points,
            reorder_index=right_reorder.numpy(),
            estimator=right_estimator,
            mano_layer=mano_layer,
            mano_params=single_ik_mano,
            ring_hue_offset=0.55,
            joint_hue_offset=0.65,
            point_radius=args.point_radius,
            joint_radius=args.joint_radius,
            axis_radius=args.axis_radius,
            mesh_joint_radius=args.mesh_joint_radius,
            link_radius=args.link_radius,
        )

    scene.export(output_glb)
    print(f"[OK] ring-joint glb saved to: {output_glb}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-step IK visualizations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    anchor_parser = subparsers.add_parser("anchor-groups", help="Export the flat-hand 100-point anchor groups as a GLB")
    anchor_parser.add_argument("--mano-path", type=str, default="assets/mano")
    anchor_parser.add_argument("--output-path", type=str, default="assets/mano_flat_hand_anchor_groups.glb")
    anchor_parser.add_argument("--hand-side", type=str, default="right", choices=["left", "right"])
    anchor_parser.add_argument("--point-radius", type=float, default=0.0038)

    prior_parser = subparsers.add_parser("axis-prior", help="Build the roll-axis prior used by single_ik")
    prior_parser.add_argument("--mano-path", type=str, default="assets/mano")
    prior_parser.add_argument("--output-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")

    frame_parser = subparsers.add_parser("wrist-frame", help="Export a wrist-frame visualization")
    frame_parser.add_argument("--points-path", type=str, required=True, help="Input file containing 100 semantic points")
    frame_parser.add_argument("--mano-path", type=str, default=None, help="MANO model directory")
    frame_parser.add_argument("--hand-side", type=str, default="right", choices=["left", "right"])
    frame_parser.add_argument("--source-hand-side", type=str, default="auto", choices=["auto", "left", "right"])
    frame_parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    frame_parser.add_argument("--axis-prior-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")
    frame_parser.add_argument("--output-dir", type=str, default="outputs/single_ik_wrist_frame")
    frame_parser.add_argument("--point-radius", type=float, default=0.0020)
    frame_parser.add_argument("--axis-radius", type=float, default=0.0016)

    ring_parser = subparsers.add_parser("ring-joint", help="Export a ring/joint diagnostic GLB")
    ring_parser.add_argument("--sample-path", type=str, default=None)
    ring_parser.add_argument("--mano-diagnostics", type=str, default=None)
    ring_parser.add_argument("--sample-stem", type=str, default=None)
    ring_parser.add_argument("--output-glb", type=str, default=None)
    ring_parser.add_argument("--mano-path", type=str, default=None)
    ring_parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    ring_parser.add_argument("--hand-side", type=str, default="both", choices=["left", "right", "both"])
    ring_parser.add_argument("--point-radius", type=float, default=0.0022)
    ring_parser.add_argument("--joint-radius", type=float, default=0.0035)
    ring_parser.add_argument("--axis-radius", type=float, default=0.0016)
    ring_parser.add_argument("--mesh-joint-radius", type=float, default=0.0026)
    ring_parser.add_argument("--link-radius", type=float, default=0.0010)

    args = parser.parse_args()
    if args.command == "anchor-groups":
        _run_anchor_groups(args)
        return
    if args.command == "axis-prior":
        _run_axis_prior(args)
        return
    if args.command == "wrist-frame":
        _run_wrist_frame(args)
        return
    if args.command == "ring-joint":
        _run_ring_joint(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
