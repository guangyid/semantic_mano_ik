#!/usr/bin/env python3
"""Independent refinement comparison from local diagnostics/sample payloads."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    build_dual_hand_mano_tensor,
    build_anatomy_loss_config,
    fitMano,
    mano_param_list_to_tensor,
    refineManoFromInit,
)
from utils.mano.mano_load import createManoLayer, resolveManoPath
from utils.mano.reorder import resolveApproxIkInputOrders


def _save_single_sample_variant(
    *,
    output_dir: Path,
    variant_name: str,
    mano_params: torch.Tensor,
    glb_path: Path | None,
) -> None:
    np.save(str(output_dir / f"{variant_name}_mano_params.npy"), mano_params[0].detach().cpu().numpy().astype(np.float32))
    if glb_path is not None and glb_path.is_file():
        glb_path.replace(output_dir / f"{variant_name}_visualization.glb")


def _build_single_hand_init(full_mano: torch.Tensor) -> Dict[str, torch.Tensor]:
    if full_mano.ndim == 1:
        full_mano = full_mano.unsqueeze(0)
    if full_mano.ndim != 2 or full_mano.shape[-1] != 61:
        raise ValueError(f"full_mano must have shape [T,61] or [61], got {tuple(full_mano.shape)}")
    return {
        "rot": full_mano[..., 0:3].detach().clone(),
        "pose": full_mano[..., 3:48].detach().clone(),
        "trans": full_mano[..., 48:51].detach().clone(),
        "shape": full_mano[..., 51:61].detach().clone(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single_ik / mano_fitting / refine_ik without upstream touch3d")
    parser.add_argument("--input-path", type=str, required=True, help="mano_diagnostics.npy or a compatible payload file")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mano-path", type=str, default=None)
    parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    parser.add_argument("--axis-prior-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")
    parser.add_argument("--sample-stems", type=str, default="")
    parser.add_argument("--known-shape-source", type=str, default="auto", help="auto | gt_mano_params | fit_mano_mano_params | single_ik_mano_params | pred_mano_params_main | mano_action | none")
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--refine-steps", type=int, default=80)
    parser.add_argument("--fit-optimizer", type=str, default="adam")
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

    left_estimator = ApproxForwardManoEstimator(
        manoPath=str(mano_path),
        handSide="left",
        device=str(device),
        axisPriorPath=str(args.axis_prior_path),
    )
    right_estimator = ApproxForwardManoEstimator(
        manoPath=str(mano_path),
        handSide="right",
        device=str(device),
        axisPriorPath=str(args.axis_prior_path),
    )
    left_reorder, right_reorder, source_hand = resolveApproxIkInputOrders(
        sampleIndices=sample_indices,
        leftEstimator=left_estimator,
        rightEstimator=right_estimator,
    )
    source_template = right_estimator.template if source_hand == "right" else left_estimator.template
    point_weights = build_point_weights(
        template=source_template,
        weight_map=parse_weight_json(args.point_weight_json),
        device=device,
    )
    anatomy_loss_config = build_anatomy_loss_config(
        mano_path=str(mano_path),
        device=device,
        weight=float(args.anatomy_loss_weight),
    )

    rows: list[Dict[str, object]] = []
    glb_dir = output_dir / "glbs"
    if args.export_glb:
        glb_dir.mkdir(parents=True, exist_ok=True)

    compact_variants: dict[str, dict[str, object]] = {}
    for stem in target_stems:
        sample_index = stems.index(stem)
        sample = extract_sample_payload(payload, sample_index, sample_count)
        left_points_key = "pred_left_points_world" if "pred_left_points_world" in sample else "left_points_world"
        right_points_key = "pred_right_points_world" if "pred_right_points_world" in sample else "right_points_world"
        pred_left_points_world = normalize_points_sequence(sample[left_points_key], name=left_points_key).to(device=device, dtype=torch.float32)
        pred_right_points_world = normalize_points_sequence(sample[right_points_key], name=right_points_key).to(device=device, dtype=torch.float32)
        pred_horizon = int(pred_left_points_world.shape[1])
        left_known_shape, right_known_shape, known_shape_tag = resolve_known_shape(
            sample=sample,
            source=args.known_shape_source,
            device=device,
        )
        left_single = left_estimator.estimate(
            pred_left_points_world.index_select(2, left_reorder.to(device)),
            knownShape=left_known_shape,
        ).fullMano
        right_single = right_estimator.estimate(
            pred_right_points_world.index_select(2, right_reorder.to(device)),
            knownShape=right_known_shape,
        ).fullMano
        single_ik_mano = build_dual_hand_mano_tensor(
            left_full_mano=left_single,
            right_full_mano=right_single,
        )
        fit_list, _, _ = fitMano(
            jointsLRef=pred_left_points_world,
            jointsRRef=pred_right_points_world,
            manoLayer=mano_layer,
            predHorizon=pred_horizon,
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
        refine_init = [{
            "left": _build_single_hand_init(left_single[0]),
            "right": _build_single_hand_init(right_single[0]),
        }]
        refined_list, _, _ = refineManoFromInit(
            jointsLRef=pred_left_points_world,
            jointsRRef=pred_right_points_world,
            manoLayer=mano_layer,
            predHorizon=pred_horizon,
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
        refined_mano = mano_param_list_to_tensor(mano_param_list=refined_list, device=device)

        variants = {
            "single_ik": single_ik_mano,
            "mano_fitting": mano_fitting,
            "refine_ik": refined_mano,
        }
        for variant_name, mano_params in variants.items():
            metrics = compute_point_metrics(
                mano_params=mano_params,
                target_left_points=pred_left_points_world,
                target_right_points=pred_right_points_world,
                mano_layer=mano_layer,
                sample_indices=sample_indices,
            )
            rows.append({
                "sample_stem": stem,
                "variant": variant_name,
                "known_shape_source": known_shape_tag,
                **metrics,
            })
            glb_path = glb_dir / f"{stem}_{variant_name}.glb" if args.export_glb else None
            if args.export_glb:
                export_hand_comparison_glb(
                    output_path=glb_path,
                    variant_name=variant_name,
                    mano_params=mano_params,
                    target_left_points=pred_left_points_world,
                    target_right_points=pred_right_points_world,
                    mano_layer=mano_layer,
                    sample_indices=sample_indices,
                )
            if len(target_stems) == 1:
                compact_variants[variant_name] = {
                    "sample_name": stem,
                    "known_shape_source": known_shape_tag,
                    "metrics": metrics,
                    "mano_params": mano_params,
                    "glb_path": glb_path,
                }

    if not rows:
        raise ValueError("No comparison results were generated")
    fieldnames = list(rows[0].keys())
    summary: Dict[str, Dict[str, float | str]] = {}
    for variant_name in ("single_ik", "mano_fitting", "refine_ik"):
        variant_rows = [row for row in rows if row["variant"] == variant_name]
        if not variant_rows:
            continue
        summary[variant_name] = {
            key: float(np.mean([float(row[key]) for row in variant_rows]))
            for key in fieldnames
            if key not in {"sample_stem", "variant", "known_shape_source"}
        }
        summary[variant_name]["known_shape_source"] = str(variant_rows[0]["known_shape_source"])

    if len(target_stems) == 1:
        for variant_name, payload in compact_variants.items():
            _save_single_sample_variant(
                output_dir=output_dir,
                variant_name=variant_name,
                mano_params=payload["mano_params"],
                glb_path=payload["glb_path"],
            )
            summary[variant_name]["sample_name"] = str(payload["sample_name"])
        if glb_dir.exists():
            glb_dir.rmdir()
        summary_path = output_dir / "compare_methods_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] summary: {summary_path}")
        return

    csv_path = output_dir / "compare_methods_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "compare_methods_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] metrics: {csv_path}")
    print(f"[OK] summary: {summary_path}")
    if args.export_glb:
        print(f"[OK] glbs: {glb_dir}")


if __name__ == "__main__":
    main()
