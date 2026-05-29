#!/usr/bin/env python3
"""Refine MANO from single-step IK initialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mano.approx import ApproxForwardManoEstimator
from utils.mano.compare import (
    build_point_weights,
    compute_point_metrics,
    export_hand_comparison_glb,
    extract_sample_payload,
    iter_target_stems,
    load_compare_payload,
    normalize_points_sequence,
    parse_weight_json,
    resolve_known_shape,
)
from utils.mano.fitting import (
    build_anatomy_loss_config,
    build_dual_hand_mano_tensor,
    mano_param_list_to_tensor,
    refineManoFromInit,
)
from utils.mano.mano_load import createManoLayer, resolveManoPath
from utils.mano.reorder import resolveApproxIkInputOrders


def _build_single_hand_init(full_mano: torch.Tensor) -> dict[str, torch.Tensor]:
    if full_mano.ndim == 1:
        full_mano = full_mano.unsqueeze(0)
    return {
        "rot": full_mano[..., 0:3].detach().clone(),
        "pose": full_mano[..., 3:48].detach().clone(),
        "trans": full_mano[..., 48:51].detach().clone(),
        "shape": full_mano[..., 51:61].detach().clone(),
    }


def _save_single_sample_result(
    *,
    output_dir: Path,
    stem: str,
    mano_params: torch.Tensor,
    metrics: dict[str, float],
    known_shape_tag: str,
    single_ik_point_rmse_cm: float,
    glb_path: Path | None,
) -> None:
    np.save(str(output_dir / "refine_ik_mano_params.npy"), mano_params[0].detach().cpu().numpy().astype(np.float32))
    payload = {
        "sample_name": stem,
        "method": "refine_ik",
        "known_shape_source": known_shape_tag,
        "single_ik_point_rmse_cm": float(single_ik_point_rmse_cm),
        **metrics,
    }
    (output_dir / "refine_ik_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if glb_path is not None and glb_path.is_file():
        glb_path.replace(output_dir / "refine_ik_visualization.glb")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine IK from 100 semantic points")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mano-path", type=str, default=None)
    parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    parser.add_argument("--axis-prior-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")
    parser.add_argument("--sample-stems", type=str, default="")
    parser.add_argument("--known-shape-source", type=str, default="auto")
    parser.add_argument("--refine-steps", type=int, default=80)
    parser.add_argument("--refine-optimizer", type=str, default="adam")
    parser.add_argument("--point-weight-json", type=str, default=None)
    parser.add_argument("--anatomy-loss-weight", type=float, default=0.0)
    parser.add_argument("--export-glb", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, sample_count, stems = load_compare_payload(args.input_path)
    target_stems = iter_target_stems(stems=stems, requested=args.sample_stems)

    mano_path = resolveManoPath(manoPath=args.mano_path, projectRoot=PROJECT_ROOT)
    sample_indices = np.load(str(args.sample_index_path)).astype(np.int64).reshape(-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano_layer = createManoLayer(modelPath=str(mano_path), device=str(device))
    for side in ("left", "right"):
        mano_layer[side].eval()
        for param in mano_layer[side].parameters():
            param.requires_grad_(False)

    left_estimator = ApproxForwardManoEstimator(manoPath=str(mano_path), handSide="left", device=str(device), axisPriorPath=str(args.axis_prior_path))
    right_estimator = ApproxForwardManoEstimator(manoPath=str(mano_path), handSide="right", device=str(device), axisPriorPath=str(args.axis_prior_path))
    left_reorder, right_reorder, source_hand = resolveApproxIkInputOrders(sampleIndices=sample_indices, leftEstimator=left_estimator, rightEstimator=right_estimator)
    source_template = right_estimator.template if source_hand == "right" else left_estimator.template
    point_weights = build_point_weights(template=source_template, weight_map=parse_weight_json(args.point_weight_json), device=device)
    anatomy_loss_config = build_anatomy_loss_config(mano_path=str(mano_path), device=device, weight=float(args.anatomy_loss_weight))

    rows: list[dict[str, object]] = []
    glb_dir = output_dir / "glbs"
    if args.export_glb:
        glb_dir.mkdir(parents=True, exist_ok=True)

    compact_result: dict[str, object] | None = None
    for stem in target_stems:
        sample_index = stems.index(stem)
        sample = extract_sample_payload(payload, sample_index, sample_count)
        left_key = "pred_left_points_world" if "pred_left_points_world" in sample else "left_points_world"
        right_key = "pred_right_points_world" if "pred_right_points_world" in sample else "right_points_world"
        left_points = normalize_points_sequence(sample[left_key], name=left_key).to(device=device, dtype=torch.float32)
        right_points = normalize_points_sequence(sample[right_key], name=right_key).to(device=device, dtype=torch.float32)
        left_known_shape, right_known_shape, known_shape_tag = resolve_known_shape(sample=sample, source=args.known_shape_source, device=device)

        left_single = left_estimator.estimate(left_points.index_select(2, left_reorder.to(device)), knownShape=left_known_shape).fullMano
        right_single = right_estimator.estimate(right_points.index_select(2, right_reorder.to(device)), knownShape=right_known_shape).fullMano
        single_ik = build_dual_hand_mano_tensor(left_full_mano=left_single, right_full_mano=right_single)
        refine_init = [{"left": _build_single_hand_init(left_single[0]), "right": _build_single_hand_init(right_single[0])}]
        refined_list, _, _ = refineManoFromInit(
            jointsLRef=left_points,
            jointsRRef=right_points,
            manoLayer=mano_layer,
            predHorizon=int(left_points.shape[1]),
            initManoParams=refine_init,
            idxL=sample_indices,
            idxR=sample_indices,
            handKeypoints=int(sample_indices.shape[0]),
            steps=int(args.refine_steps),
            verbose=bool(args.verbose),
            alignRoot=False,
            pointWeights=point_weights,
            anatomyLossConfig=anatomy_loss_config,
            optimizerName=str(args.refine_optimizer),
        )
        refine_ik = mano_param_list_to_tensor(mano_param_list=refined_list, device=device)

        single_metrics = compute_point_metrics(
            mano_params=single_ik,
            target_left_points=left_points,
            target_right_points=right_points,
            mano_layer=mano_layer,
            sample_indices=sample_indices,
        )
        refine_metrics = compute_point_metrics(
            mano_params=refine_ik,
            target_left_points=left_points,
            target_right_points=right_points,
            mano_layer=mano_layer,
            sample_indices=sample_indices,
        )
        rows.append({
            "sample_stem": stem,
            "method": "refine_ik",
            "known_shape_source": known_shape_tag,
            **refine_metrics,
            "single_ik_point_rmse_cm": single_metrics["point_rmse_cm"],
        })
        glb_path = glb_dir / f"{stem}_refine_ik.glb" if args.export_glb else None
        if args.export_glb:
            export_hand_comparison_glb(
                output_path=glb_path,
                variant_name="refine_ik",
                mano_params=refine_ik,
                target_left_points=left_points,
                target_right_points=right_points,
                mano_layer=mano_layer,
                sample_indices=sample_indices,
            )
        if len(target_stems) == 1:
            compact_result = {
                "stem": stem,
                "mano_params": refine_ik,
                "metrics": refine_metrics,
                "known_shape_tag": known_shape_tag,
                "single_ik_point_rmse_cm": single_metrics["point_rmse_cm"],
                "glb_path": glb_path,
            }

    if len(target_stems) == 1 and compact_result is not None:
        _save_single_sample_result(
            output_dir=output_dir,
            stem=str(compact_result["stem"]),
            mano_params=compact_result["mano_params"],
            metrics=compact_result["metrics"],
            known_shape_tag=str(compact_result["known_shape_tag"]),
            single_ik_point_rmse_cm=float(compact_result["single_ik_point_rmse_cm"]),
            glb_path=compact_result["glb_path"],
        )
        if glb_dir.exists():
            glb_dir.rmdir()
        print(f"[OK] metrics: {output_dir / 'refine_ik_metrics.json'}")
        return

    csv_path = output_dir / "refine_ik_metrics.csv"
    fieldnames = list(rows[0].keys())
    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in fieldnames
        if key not in {"sample_stem", "method", "known_shape_source"}
    }
    summary["known_shape_source"] = str(rows[0]["known_shape_source"])
    (output_dir / "refine_ik_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] metrics: {csv_path}")


if __name__ == "__main__":
    main()
