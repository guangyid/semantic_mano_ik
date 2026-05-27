"""Payload parsing helpers for semantic point and MANO parameter files."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

from .reorder import buildApproxIkInputOrder, resolveApproxIkInputOrders


def _install_numpy_pickle_compat_aliases() -> None:
    # Older pickled numpy payloads may reference `numpy._core.*`.
    try:
        import numpy.core as numpy_core

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    except Exception:
        return


def load_payload_file(path: str | Path) -> Any:
    _install_numpy_pickle_compat_aliases()
    loaded = np.load(str(path), allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {key: loaded[key] for key in loaded.files}
    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        return loaded.item()
    return loaded


def normalize_points_array(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape == (100, 3):
        return arr
    if arr.shape == (1, 100, 3):
        return arr[0]
    if arr.shape == (1, 1, 100, 3):
        return arr[0, 0]
    raise ValueError(f"{name} must have shape [100,3], [1,100,3], or [1,1,100,3], got {arr.shape}")


def normalize_mano_array(params: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(params, dtype=np.float32)
    if arr.shape == (61,):
        return arr
    if arr.shape == (1, 61):
        return arr[0]
    if arr.shape == (1, 1, 61):
        return arr[0, 0]
    if arr.shape == (122,):
        return arr
    if arr.shape == (1, 122):
        return arr[0]
    if arr.shape == (1, 1, 122):
        return arr[0, 0]
    raise ValueError(f"{name} must be [61] or [122], or the corresponding single-batch variants, got {arr.shape}")


def _points_candidate_keys(hand_side: str) -> list[str]:
    side_keys = [
        f"{hand_side}_points_world",
        f"pred_{hand_side}_points_world",
        f"{hand_side}_semantic_points",
        f"pred_{hand_side}_semantic_points",
    ]
    return side_keys + ["points_world", "semantic_points", "sampled_points"]


def load_single_hand_points(path: str | Path, *, handSide: str) -> tuple[np.ndarray, dict[str, Any]]:
    payload = load_payload_file(path)
    meta = {"source_path": str(path)}
    if isinstance(payload, dict):
        for key in _points_candidate_keys(handSide):
            if key in payload:
                meta["points_key"] = key
                if "sample_name" in payload:
                    meta["sample_name"] = str(payload["sample_name"])
                if "sample_index_order" in payload:
                    meta["sample_index_order"] = np.asarray(payload["sample_index_order"], dtype=np.int64).reshape(-1)
                if "sample_index_source_hand" in payload:
                    meta["sample_index_source_hand"] = str(payload["sample_index_source_hand"])
                return normalize_points_array(payload[key], name=key), meta
        raise ValueError(f"No 100-point field for the {handSide} hand was found in {path}")
    return normalize_points_array(payload, name=str(path)), meta


def _mano_candidate_keys(hand_side: str) -> list[str]:
    side_keys = [
        f"{hand_side}_mano_params",
        f"pred_{hand_side}_mano_params",
        f"{hand_side}_mano",
        f"pred_{hand_side}_mano",
    ]
    return side_keys + ["mano_params", "pred_mano_params", "single_ik_mano_params", "gt_mano_params"]


def slice_single_hand_mano(full_or_single: np.ndarray, *, handSide: str) -> np.ndarray:
    arr = normalize_mano_array(full_or_single, name="mano_params")
    if arr.shape[0] == 61:
        return arr.astype(np.float32)
    if arr.shape[0] != 122:
        raise ValueError(f"MANO parameter length must be 61 or 122, got {arr.shape[0]}")
    return arr[:61].astype(np.float32) if handSide == "left" else arr[61:].astype(np.float32)


def load_single_hand_mano(path: str | Path, *, handSide: str) -> tuple[np.ndarray, dict[str, Any]]:
    payload = load_payload_file(path)
    meta = {"source_path": str(path)}
    if isinstance(payload, dict):
        for key in _mano_candidate_keys(handSide):
            if key in payload:
                meta["mano_key"] = key
                if "sample_name" in payload:
                    meta["sample_name"] = str(payload["sample_name"])
                return slice_single_hand_mano(payload[key], handSide=handSide), meta
        raise ValueError(f"No MANO parameter field was found in {path}")
    return slice_single_hand_mano(payload, handSide=handSide), meta


def _build_reorder_index(
    *,
    sampleIndices: np.ndarray,
    sourceHandSide: str,
    targetHandSide: str,
    leftEstimator,
    rightEstimator,
) -> np.ndarray:
    source_template = leftEstimator.template if sourceHandSide == "left" else rightEstimator.template
    target_template = leftEstimator.template if targetHandSide == "left" else rightEstimator.template
    reorder = buildApproxIkInputOrder(
        sourceTemplate=source_template,
        targetTemplate=target_template,
        sourceHandSide=sourceHandSide,
        targetHandSide=targetHandSide,
    )
    return reorder.detach().cpu().numpy().astype(np.int64)


def resolve_input_reorder(
    *,
    sampleIndices: np.ndarray,
    targetHandSide: str,
    sourceHandSide: str,
    leftEstimator,
    rightEstimator,
) -> tuple[np.ndarray, str]:
    if sourceHandSide == "auto":
        left_reorder, right_reorder, detected = resolveApproxIkInputOrders(
            sampleIndices=sampleIndices,
            leftEstimator=leftEstimator,
            rightEstimator=rightEstimator,
        )
        reorder = left_reorder if targetHandSide == "left" else right_reorder
        return reorder.detach().cpu().numpy().astype(np.int64), detected
    if sourceHandSide not in {"left", "right"}:
        raise ValueError(f"sourceHandSide must be auto, left, or right, got {sourceHandSide}")
    reorder = _build_reorder_index(
        sampleIndices=sampleIndices,
        sourceHandSide=sourceHandSide,
        targetHandSide=targetHandSide,
        leftEstimator=leftEstimator,
        rightEstimator=rightEstimator,
    )
    return reorder, sourceHandSide


def invert_permutation(order: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.shape[0], dtype=order.dtype)
    return inverse
