#!/usr/bin/env python3
"""Fit dual-hand MANO from 100 semantic points."""
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
from utils.mano.fitting import build_anatomy_loss_config, fitMano, mano_param_list_to_tensor
from utils.mano.mano_load import createManoLayer, resolveManoPath
from utils.mano.reorder import resolveApproxIkInputOrders
from utils.mano.approx import ApproxForwardManoEstimator


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit MANO from 100 semantic points")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mano-path", type=str, default=None)
    parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    parser.add_argument("--axis-prior-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")
    parser.add_argument("--sample-stems", type=str, default="")
    parser.add_argument("--known-shape-source", type=str, default="auto")
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--fit-optimizer", type=str, default="adam")
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
    _, _, source_hand = resolveApproxIkInputOrders(sampleIndices=sample_indices, leftEstimator=left_estimator, rightEstimator=right_estimator)
    source_template = right_estimator.template if source_hand == "right" else left_estimator.template
    point_weights = build_point_weights(template=source_template, weight_map=parse_weight_json(args.point_weight_json), device=device)
    anatomy_loss_config = build_anatomy_loss_config(mano_path=str(mano_path), device=device, weight=float(args.anatomy_loss_weight))

    rows: list[dict[str, object]] = []
    glb_dir = output_dir / "glbs"
    if args.export_glb:
        glb_dir.mkdir(parents=True, exist_ok=True)

    for stem in target_stems:
        sample_index = stems.index(stem)
        sample = extract_sample_payload(payload, sample_index, sample_count)
        left_key = "pred_left_points_world" if "pred_left_points_world" in sample else "left_points_world"
        right_key = "pred_right_points_world" if "pred_right_points_world" in sample else "right_points_world"
        left_points = normalize_points_sequence(sample[left_key], name=left_key).to(device=device, dtype=torch.float32)
        right_points = normalize_points_sequence(sample[right_key], name=right_key).to(device=device, dtype=torch.float32)
        _, _, known_shape_tag = resolve_known_shape(sample=sample, source=args.known_shape_source, device=device)
        fit_list, _, _ = fitMano(
            jointsLRef=left_points,
            jointsRRef=right_points,
            manoLayer=mano_layer,
            predHorizon=int(left_points.shape[1]),
            idxL=sample_indices,
            idxR=sample_indices,
            handKeypoints=int(sample_indices.shape[0]),
            steps=int(args.fit_steps),
            verbose=bool(args.verbose),
            initManoParams=None,
            alignRoot=True,
            pointWeights=point_weights,
            anatomyLossConfig=anatomy_loss_config,
            optimizerName=str(args.fit_optimizer),
        )
        mano_fitting = mano_param_list_to_tensor(mano_param_list=fit_list, device=device)
        metrics = compute_point_metrics(
            mano_params=mano_fitting,
            target_left_points=left_points,
            target_right_points=right_points,
            mano_layer=mano_layer,
            sample_indices=sample_indices,
        )
        rows.append({"sample_stem": stem, "method": "mano_fitting", "known_shape_source": known_shape_tag, **metrics})
        np.save(str(output_dir / f"{stem}_mano_fitting.npy"), mano_fitting.detach().cpu().numpy())
        if args.export_glb:
            export_hand_comparison_glb(
                output_path=glb_dir / f"{stem}_mano_fitting.glb",
                variant_name="mano_fitting",
                mano_params=mano_fitting,
                target_left_points=left_points,
                target_right_points=right_points,
                mano_layer=mano_layer,
                sample_indices=sample_indices,
            )

    csv_path = output_dir / "mano_fitting_metrics.csv"
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
    (output_dir / "mano_fitting_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] metrics: {csv_path}")


if __name__ == "__main__":
    main()
