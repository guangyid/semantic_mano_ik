#!/usr/bin/env python3
"""Build the local demo sample used for ring/joint visualization and quick tests."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mano.approx import ApproxForwardManoEstimator
from utils.mano.mano_load import createManoLayer, resolveManoPath
from utils.mano.reorder import resolveApproxIkInputOrders


FINGER_SLICES = {
    "index": slice(0, 3),
    "middle": slice(3, 6),
    "pinky": slice(6, 9),
    "ring": slice(9, 12),
    "thumb": slice(12, 15),
}


def _set_finger_axis(hand_pose: np.ndarray, finger_name: str, axis: int, values: list[float]) -> None:
    hand_pose[FINGER_SLICES[finger_name], axis] = np.asarray(values, dtype=np.float32)


def _set_finger_curl(hand_pose: np.ndarray, finger_name: str, values: list[float]) -> None:
    _set_finger_axis(hand_pose, finger_name, 0, values)


def _build_demo_hand_params(hand_side: str) -> dict[str, np.ndarray]:
    pose = np.zeros((15, 3), dtype=np.float32)
    if hand_side == "left":
        _set_finger_curl(pose, "index", [0.10, 0.14, 0.08])
        for finger_name in ("middle", "ring", "pinky"):
            _set_finger_curl(pose, finger_name, [1.00, 1.28, 1.06])
        _set_finger_curl(pose, "thumb", [0.36, 0.54, 0.44])
        _set_finger_axis(pose, "thumb", 1, [-0.24, -0.18, -0.12])
        root_rot = np.array([0.08, -0.10, 0.18], dtype=np.float32)
        transl = np.array([-0.055, 0.012, 0.000], dtype=np.float32)
    else:
        for finger_name in ("index", "middle", "ring", "pinky"):
            _set_finger_curl(pose, finger_name, [0.58, 0.92, 0.76])
        _set_finger_curl(pose, "thumb", [0.30, 0.48, 0.38])
        _set_finger_axis(pose, "index", 1, [0.42, 0.16, 0.08])
        _set_finger_axis(pose, "middle", 1, [0.05, 0.02, 0.00])
        _set_finger_axis(pose, "ring", 1, [-0.24, -0.10, -0.04])
        _set_finger_axis(pose, "pinky", 1, [-0.46, -0.18, -0.08])
        _set_finger_axis(pose, "thumb", 1, [0.26, 0.14, 0.08])
        root_rot = np.array([0.10, 0.04, -0.08], dtype=np.float32)
        transl = np.array([0.055, -0.010, 0.004], dtype=np.float32)
    return {
        "root_rot": root_rot,
        "hand_pose": pose.reshape(-1),
        "transl": transl,
        "shape": np.zeros((10,), dtype=np.float32),
    }


def _decode_vertices(
    *,
    mano_layer,
    hand_side: str,
    mano_params: dict[str, np.ndarray],
) -> np.ndarray:
    root_rot = torch.from_numpy(mano_params["root_rot"][None, :]).float()
    hand_pose = torch.from_numpy(mano_params["hand_pose"][None, :]).float()
    betas = torch.from_numpy(mano_params["shape"][None, :]).float()
    transl = torch.from_numpy(mano_params["transl"][None, :]).float()
    output = mano_layer[hand_side](
        global_orient=root_rot,
        hand_pose=hand_pose,
        betas=betas,
        transl=transl,
    )
    return output.vertices[0].detach().cpu().numpy().astype(np.float32)


def _pack_full_mano(mano_params: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            mano_params["root_rot"],
            mano_params["hand_pose"],
            mano_params["transl"],
            mano_params["shape"],
        ],
        axis=0,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local ring/joint demo sample")
    parser.add_argument("--mano-path", type=str, default=None)
    parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    parser.add_argument("--axis-prior-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")
    parser.add_argument("--output-path", type=str, default="samples/ring_joint_demo.npy")
    args = parser.parse_args()

    mano_path = resolveManoPath(manoPath=args.mano_path, projectRoot=PROJECT_ROOT)
    sample_index_path = Path(args.sample_index_path)
    axis_prior_path = Path(args.axis_prior_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_index_order = np.load(str(sample_index_path)).astype(np.int64)
    mano_layer = createManoLayer(modelPath=mano_path, device="cpu")
    for side in ("left", "right"):
        mano_layer[side].eval()
        for param in mano_layer[side].parameters():
            param.requires_grad_(False)

    left_gt = _build_demo_hand_params("left")
    right_gt = _build_demo_hand_params("right")
    left_verts = _decode_vertices(mano_layer=mano_layer, hand_side="left", mano_params=left_gt)
    right_verts = _decode_vertices(mano_layer=mano_layer, hand_side="right", mano_params=right_gt)
    left_points = left_verts[sample_index_order]
    right_points = right_verts[sample_index_order]

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
    left_reorder, right_reorder, source_hand_side = resolveApproxIkInputOrders(
        sampleIndices=sample_index_order,
        leftEstimator=left_estimator,
        rightEstimator=right_estimator,
    )
    left_estimate = left_estimator.estimate(torch.from_numpy(left_points[left_reorder.numpy()]).float())
    right_estimate = right_estimator.estimate(torch.from_numpy(right_points[right_reorder.numpy()]).float())
    single_ik_full = np.concatenate(
        [
            left_estimate.fullMano.detach().cpu().numpy().reshape(61),
            right_estimate.fullMano.detach().cpu().numpy().reshape(61),
        ],
        axis=0,
    ).astype(np.float32)
    gt_full = np.concatenate([_pack_full_mano(left_gt), _pack_full_mano(right_gt)], axis=0).astype(np.float32)

    payload = {
        "format_version": 1,
        "sample_name": "ring_joint_demo",
        "sample_index_order": sample_index_order,
        "sample_index_source_hand": source_hand_side,
        "left_points_world": left_points.astype(np.float32),
        "right_points_world": right_points.astype(np.float32),
        "single_ik_mano_params": single_ik_full.reshape(1, 1, 122),
        "gt_mano_params": gt_full.reshape(1, 1, 122),
    }
    np.save(str(output_path), payload, allow_pickle=True)
    print(f"[OK] saved demo sample: {output_path}")


if __name__ == "__main__":
    main()
