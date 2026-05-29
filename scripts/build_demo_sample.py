#!/usr/bin/env python3
"""Build the local demo sample from a frame of the bundled MANO sequence asset."""
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
from utils.mano.payload import load_payload_file
from utils.mano.reorder import resolveApproxIkInputOrders


def _unwrap_object_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def _resolve_frame_index(frame_index: int, frame_count: int) -> int:
    resolved = frame_index if frame_index >= 0 else frame_count + frame_index
    if resolved < 0 or resolved >= frame_count:
        raise IndexError(f"frame_index={frame_index} is out of range for frame_count={frame_count}")
    return resolved


def _load_demo_hand_params(
    *,
    sequence_path: Path,
    sample_key: str,
    frame_index: int,
    hand_side: str,
) -> tuple[dict[str, np.ndarray], int]:
    payload = load_payload_file(sequence_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{sequence_path} must contain a dict-like sequence payload")
    if sample_key not in payload:
        raise KeyError(f"{sequence_path} does not contain sample_key={sample_key}; available keys: {sorted(payload.keys())}")
    entry = _unwrap_object_scalar(payload[sample_key])
    if not isinstance(entry, dict):
        raise ValueError(f"{sequence_path}:{sample_key} is not a dict payload")

    prefix = "left" if hand_side == "left" else "right"
    pose = np.asarray(entry[f"{prefix}_pose"], dtype=np.float32)
    transl = np.asarray(entry[f"{prefix}_trans"], dtype=np.float32)
    shape = np.asarray(entry[f"{prefix}_shape"], dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 48:
        raise ValueError(f"{prefix}_pose must have shape [T,48], got {pose.shape}")
    if transl.shape != (pose.shape[0], 3):
        raise ValueError(f"{prefix}_trans must have shape [T,3], got {transl.shape}")
    if shape.shape != (pose.shape[0], 10):
        raise ValueError(f"{prefix}_shape must have shape [T,10], got {shape.shape}")

    resolved_frame = _resolve_frame_index(frame_index=frame_index, frame_count=int(pose.shape[0]))
    pose_frame = pose[resolved_frame]
    return {
        "root_rot": pose_frame[:3].astype(np.float32),
        "hand_pose": pose_frame[3:].astype(np.float32),
        "transl": transl[resolved_frame].astype(np.float32),
        "shape": shape[resolved_frame].astype(np.float32),
    }, resolved_frame


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
    parser = argparse.ArgumentParser(description="Build the local ring/joint demo sample from sequence_mano")
    parser.add_argument("--mano-path", type=str, default=None)
    parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    parser.add_argument("--axis-prior-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")
    parser.add_argument("--sequence-path", type=str, default="assets/sequence_mano.npz")
    parser.add_argument("--sample-key", type=str, default="0")
    parser.add_argument("--frame-index", type=int, default=-1, help="Frame index inside the chosen sequence sample; -1 means the last frame")
    parser.add_argument("--output-path", type=str, default="outputs/ring_joint_demo.npy")
    args = parser.parse_args()

    mano_path = resolveManoPath(manoPath=args.mano_path, projectRoot=PROJECT_ROOT)
    sample_index_path = Path(args.sample_index_path)
    axis_prior_path = Path(args.axis_prior_path)
    sequence_path = Path(args.sequence_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_index_order = np.load(str(sample_index_path)).astype(np.int64)
    mano_layer = createManoLayer(modelPath=mano_path, device="cpu")
    for side in ("left", "right"):
        mano_layer[side].eval()
        for param in mano_layer[side].parameters():
            param.requires_grad_(False)

    left_gt, resolved_frame = _load_demo_hand_params(
        sequence_path=sequence_path,
        sample_key=str(args.sample_key),
        frame_index=int(args.frame_index),
        hand_side="left",
    )
    right_gt, _ = _load_demo_hand_params(
        sequence_path=sequence_path,
        sample_key=str(args.sample_key),
        frame_index=int(args.frame_index),
        hand_side="right",
    )
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
        "sample_name": f"sequence_mano_sample_{args.sample_key}_frame_{resolved_frame:03d}",
        "sample_index_order": sample_index_order,
        "sample_index_source_hand": source_hand_side,
        "source_sequence_path": str(sequence_path),
        "source_sequence_key": str(args.sample_key),
        "source_frame_index": int(resolved_frame),
        "left_points_world": left_points.astype(np.float32),
        "right_points_world": right_points.astype(np.float32),
        "single_ik_mano_params": single_ik_full.reshape(1, 1, 122),
        "gt_mano_params": gt_full.reshape(1, 1, 122),
    }
    np.save(str(output_path), payload, allow_pickle=True)
    print(f"[OK] saved demo sample: {output_path}")


if __name__ == "__main__":
    main()
