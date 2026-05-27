"""Small MANO-centric helper functions used by CLI scripts."""
from __future__ import annotations

import numpy as np
import torch


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
