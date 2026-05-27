"""Shared helpers for mano fitting / refine IK scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import trimesh

from ..vis.trimesh_vis import add_sphere, build_hand_mesh
from .fitting import LEFT_SHAPE_SLICE, RIGHT_SHAPE_SLICE, decode_mano_params_to_hand_verts_joints
from .payload import load_payload_file


def parse_weight_json(path: str | None) -> Dict[str, float]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("point weight JSON must be an object")
    return {str(key): float(value) for key, value in payload.items()}


def build_point_weights(*, template, weight_map: Dict[str, float], device: torch.device) -> torch.Tensor | None:
    if not weight_map:
        return None
    index_order = [int(index) for index in template.indexOrder]
    index_to_offset = {index: offset for offset, index in enumerate(index_order)}
    default_weight = float(weight_map.get("default", 1.0))
    weights = torch.full((len(index_order),), default_weight, device=device)
    for group in template.groups:
        group_name = str(group.name)
        if group_name.endswith("_ring"):
            weight = float(weight_map.get("ring", default_weight))
        elif group_name.endswith("_tip_updown"):
            weight = float(weight_map.get("tip", default_weight))
        elif group_name.endswith("_joint_3_updown"):
            weight = float(weight_map.get("joint_3", default_weight))
        elif group_name.endswith("_joint_2_updown"):
            weight = float(weight_map.get("joint_2", default_weight))
        elif group_name.endswith("_joint_1_updown"):
            weight = float(weight_map.get("joint_1", default_weight))
        elif group_name.startswith("wrist_cuff"):
            weight = float(weight_map.get("wrist_cuff", default_weight))
        elif group_name.startswith("palm_"):
            weight = float(weight_map.get("palm_surface", default_weight))
        else:
            weight = default_weight
        for index in group.indices:
            weights[index_to_offset[int(index)]] = weight
    return weights / weights.mean().clamp(min=1.0e-6)


def infer_sample_count(payload: Dict[str, Any]) -> int:
    if "sample_stems" in payload:
        return int(np.asarray(payload["sample_stems"]).shape[0])
    for key in ("pred_left_points_world", "left_points_world"):
        if key in payload:
            arr = np.asarray(payload[key])
            return int(arr.shape[0]) if arr.ndim >= 4 else 1
    return 1


def build_sample_stems(payload: Dict[str, Any], sample_count: int) -> list[str]:
    if "sample_stems" in payload:
        return [str(stem) for stem in np.asarray(payload["sample_stems"]).tolist()]
    sample_name = str(payload.get("sample_name", "sample"))
    if sample_count == 1:
        return [sample_name]
    return [f"{sample_name}_{index:05d}" for index in range(sample_count)]


def select_sample_item(value: Any, sample_index: int, sample_count: int) -> Any:
    if isinstance(value, np.ndarray) and value.shape[:1] == (sample_count,):
        return value[sample_index:sample_index + 1]
    return value


def extract_sample_payload(payload: Dict[str, Any], sample_index: int, sample_count: int) -> Dict[str, Any]:
    return {key: select_sample_item(value, sample_index, sample_count) for key, value in payload.items()}


def normalize_points_sequence(points: Any, *, name: str) -> torch.Tensor:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return torch.from_numpy(arr[None, None, ...])
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return torch.from_numpy(arr[None, ...])
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return torch.from_numpy(arr)
    raise ValueError(f"{name} must have shape [100,3], [T,100,3], or [B,T,100,3], got {arr.shape}")


def normalize_mano_sequence(params: Any, *, name: str) -> torch.Tensor:
    arr = np.asarray(params, dtype=np.float32)
    if arr.ndim == 1:
        return torch.from_numpy(arr[None, None, :])
    if arr.ndim == 2:
        return torch.from_numpy(arr[None, ...])
    if arr.ndim == 3:
        return torch.from_numpy(arr)
    raise ValueError(f"{name} must have shape [D], [T,D], or [B,T,D], got {arr.shape}")


def resolve_known_shape(
    *,
    sample: Dict[str, Any],
    source: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    candidates: list[tuple[str, str]] = []
    if source == "auto":
        candidates = [
            ("gt_mano_params", "gt_mano_params"),
            ("fit_mano_mano_params", "fit_mano_mano_params"),
            ("single_ik_mano_params", "single_ik_mano_params"),
            ("pred_mano_params_main", "pred_mano_params_main"),
            ("mano_action", "mano_action"),
        ]
    elif source != "none":
        candidates = [(source, source)]
    for key, tag in candidates:
        if key not in sample:
            continue
        mano = normalize_mano_sequence(sample[key], name=key).to(device=device, dtype=torch.float32)
        frame0 = mano[:, 0, :]
        if frame0.shape[-1] == 122:
            return frame0[:, LEFT_SHAPE_SLICE], frame0[:, RIGHT_SHAPE_SLICE], tag
    return None, None, "none"


def sample_hand_points(
    *,
    mano_params: torch.Tensor,
    mano_layer,
    sample_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_verts, right_verts, _, _ = decode_mano_params_to_hand_verts_joints(
        mano_params=mano_params,
        mano_layer=mano_layer,
    )
    sample_idx = torch.from_numpy(sample_indices).long().to(mano_params.device)
    return (
        left_verts[:, :, sample_idx, :],
        right_verts[:, :, sample_idx, :],
        left_verts,
        right_verts,
    )


def compute_point_metrics(
    *,
    mano_params: torch.Tensor,
    target_left_points: torch.Tensor,
    target_right_points: torch.Tensor,
    mano_layer,
    sample_indices: np.ndarray,
) -> Dict[str, float]:
    left_sampled, right_sampled, _, _ = sample_hand_points(
        mano_params=mano_params,
        mano_layer=mano_layer,
        sample_indices=sample_indices,
    )
    left_error = torch.linalg.norm(left_sampled - target_left_points, dim=-1)
    right_error = torch.linalg.norm(right_sampled - target_right_points, dim=-1)
    return {
        "left_point_rmse_cm": float(torch.sqrt(left_error.square().mean()).item() * 100.0),
        "right_point_rmse_cm": float(torch.sqrt(right_error.square().mean()).item() * 100.0),
        "point_rmse_cm": float(torch.sqrt(torch.cat([left_error, right_error], dim=-1).square().mean()).item() * 100.0),
        "left_point_mean_cm": float(left_error.mean().item() * 100.0),
        "right_point_mean_cm": float(right_error.mean().item() * 100.0),
        "left_point_max_cm": float(left_error.max().item() * 100.0),
        "right_point_max_cm": float(right_error.max().item() * 100.0),
    }


def export_hand_comparison_glb(
    *,
    output_path: Path,
    variant_name: str,
    mano_params: torch.Tensor,
    target_left_points: torch.Tensor,
    target_right_points: torch.Tensor,
    mano_layer,
    sample_indices: np.ndarray,
) -> None:
    _, _, left_verts, right_verts = sample_hand_points(
        mano_params=mano_params,
        mano_layer=mano_layer,
        sample_indices=sample_indices,
    )
    last_left_verts = left_verts[0, -1].detach().cpu().numpy()
    last_right_verts = right_verts[0, -1].detach().cpu().numpy()
    left_faces = mano_layer["left"].faces.astype(np.int64)
    right_faces = mano_layer["right"].faces.astype(np.int64)
    scene = trimesh.Scene()
    scene.add_geometry(build_hand_mesh(last_left_verts, left_faces, color=np.array([90, 150, 245, 150], dtype=np.uint8)), node_name=f"{variant_name}_left_mesh")
    scene.add_geometry(build_hand_mesh(last_right_verts, right_faces, color=np.array([245, 160, 90, 150], dtype=np.uint8)), node_name=f"{variant_name}_right_mesh")
    left_sampled = last_left_verts[sample_indices]
    right_sampled = last_right_verts[sample_indices]
    for idx, point in enumerate(target_left_points[0, -1].detach().cpu().numpy()):
        add_sphere(scene, point, 0.0020, np.array([50, 180, 255, 255], dtype=np.uint8), f"target_left_{idx:03d}")
    for idx, point in enumerate(target_right_points[0, -1].detach().cpu().numpy()):
        add_sphere(scene, point, 0.0020, np.array([255, 120, 70, 255], dtype=np.uint8), f"target_right_{idx:03d}")
    for idx, point in enumerate(left_sampled):
        add_sphere(scene, point, 0.0015, np.array([80, 255, 150, 255], dtype=np.uint8), f"pred_left_{idx:03d}")
    for idx, point in enumerate(right_sampled):
        add_sphere(scene, point, 0.0015, np.array([200, 255, 80, 255], dtype=np.uint8), f"pred_right_{idx:03d}")
    output_path.write_bytes(scene.export(file_type="glb"))


def iter_target_stems(*, stems: List[str], requested: str) -> List[str]:
    if not requested.strip():
        return stems
    wanted = [stem.strip() for stem in requested.split(",") if stem.strip()]
    missing = [stem for stem in wanted if stem not in stems]
    if missing:
        raise ValueError(f"sample_stems were not found: {missing}")
    return wanted


def load_compare_payload(input_path: str | Path) -> tuple[Dict[str, Any], int, list[str]]:
    payload = load_payload_file(input_path)
    if not isinstance(payload, dict):
        raise ValueError("input-path must resolve to a dict payload")
    sample_count = infer_sample_count(payload)
    stems = build_sample_stems(payload, sample_count)
    return payload, sample_count, stems
