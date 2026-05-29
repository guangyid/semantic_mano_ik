"""Small MANO-centric helper functions used by CLI scripts."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
JOINT_LEVELS = ["joint_1", "joint_2", "joint_3", "tip"]
FINGER_JOINT_GROUPS = OrderedDict([
    ("thumb", (13, 14, 15)),
    ("index", (1, 2, 3)),
    ("middle", (4, 5, 6)),
    ("ring", (10, 11, 12)),
    ("pinky", (7, 8, 9)),
])


def decode_single_hand_mano(
    *,
    manoLayer,
    manoParams: np.ndarray | torch.Tensor,
    handSide: str,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(manoParams, np.ndarray):
        params = torch.from_numpy(manoParams).float()
    else:
        params = manoParams.detach().float().cpu()
    params = params.reshape(-1)
    if params.shape[0] != 61:
        raise ValueError(f"Single-hand MANO parameters must have length 61, got {params.shape[0]}")
    output = manoLayer[handSide](
        global_orient=params[None, 0:3],
        hand_pose=params[None, 3:48],
        betas=params[None, 51:61],
        transl=params[None, 48:51],
    )
    verts = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = output.joints[0].detach().cpu().numpy().astype(np.float32)
    return verts, joints


def build_root_points(template, orderedPoints: np.ndarray) -> dict[str, np.ndarray]:
    index_to_offset = {int(index): offset for offset, index in enumerate(template.indexOrder)}
    return {
        name: orderedPoints[index_to_offset[int(index)]].astype(np.float32)
        for name, index in template.rootPointMap.items()
    }


def normalize_vector(vec: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return np.zeros_like(arr)
    return arr / norm


def project_to_plane(vec: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis_unit = normalize_vector(axis)
    return np.asarray(vec, dtype=np.float32) - np.dot(vec, axis_unit) * axis_unit


def build_wrist_frame(rootPoints: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    wrist_radial = np.asarray(rootPoints["wrist_radial"], dtype=np.float32)
    wrist_ulnar = np.asarray(rootPoints["wrist_ulnar"], dtype=np.float32)
    index_base = np.asarray(rootPoints["index_base"], dtype=np.float32)
    middle_base = np.asarray(rootPoints["middle_base"], dtype=np.float32)
    pinky_base = np.asarray(rootPoints["pinky_base"], dtype=np.float32)

    wrist_center = 0.5 * (wrist_radial + wrist_ulnar)
    x_axis = normalize_vector(middle_base - wrist_center)
    y_seed = project_to_plane(index_base - pinky_base, x_axis)
    y_axis = normalize_vector(y_seed)
    z_axis = normalize_vector(np.cross(x_axis, y_axis))
    if float(np.linalg.norm(z_axis)) < 1.0e-8:
        z_axis = normalize_vector(np.cross(index_base - wrist_center, pinky_base - wrist_center))
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    base_scale = max(
        float(np.linalg.norm(middle_base - wrist_center)),
        0.5 * float(np.linalg.norm(index_base - pinky_base)),
        1.0e-4,
    )
    axis_length = float(base_scale * 0.85)
    return wrist_center.astype(np.float32), {
        "x": x_axis.astype(np.float32),
        "y": y_axis.astype(np.float32),
        "z": z_axis.astype(np.float32),
    }, axis_length


def _normalize_torch(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp(min=eps)


def build_mesh_fingertips(*, verts: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    distal_indices = [group[-1] for group in FINGER_JOINT_GROUPS.values()]
    proximal_indices = [group[-2] for group in FINGER_JOINT_GROUPS.values()]
    distal = joints[distal_indices]
    proximal = joints[proximal_indices]
    directions = _normalize_torch(distal - proximal)
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


def build_joint_centers(*, template, points_ordered: np.ndarray) -> dict[str, np.ndarray]:
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


def build_mesh_joint_centers(*, mesh_joints: np.ndarray, mesh_tips: np.ndarray) -> dict[str, np.ndarray]:
    centers: dict[str, np.ndarray] = {}
    for finger_name, joint_ids in FINGER_JOINT_GROUPS.items():
        for idx, joint_id in enumerate(joint_ids):
            centers[f"{finger_name}_joint_{idx + 1}"] = mesh_joints[joint_id]
        centers[f"{finger_name}_tip"] = mesh_tips[FINGER_NAMES.index(finger_name)]
    return centers


def build_ring_points(*, template, points_ordered: np.ndarray) -> dict[str, np.ndarray]:
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


def build_segment_axes(*, joint_centers: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    axes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for finger in FINGER_NAMES:
        for seg_idx in range(1, 4):
            proximal = joint_centers[f"{finger}_joint_{seg_idx}"]
            distal = joint_centers[f"{finger}_tip"] if seg_idx == 3 else joint_centers[f"{finger}_joint_{seg_idx + 1}"]
            axes[f"{finger}_segment_{seg_idx}"] = (proximal, distal)
    return axes


def segment_name_to_finger(segment_name: str) -> str:
    for finger in FINGER_NAMES:
        if segment_name.startswith(finger):
            return finger
    raise KeyError(f"unknown segment name: {segment_name}")


def joint_name_to_finger(joint_name: str) -> str:
    for finger in FINGER_NAMES:
        if joint_name.startswith(finger):
            return finger
    raise KeyError(f"unknown joint name: {joint_name}")
