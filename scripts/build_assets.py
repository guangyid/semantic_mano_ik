#!/usr/bin/env python3
"""Build the small assets used by this repository."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
projectRootStr = str(PROJECT_ROOT)
sys.path = [path for path in sys.path if path != projectRootStr]
sys.path.insert(0, projectRootStr)
utilsModule = sys.modules.get("utils")
if utilsModule is not None:
    utilsFile = getattr(utilsModule, "__file__", "") or ""
    if utilsFile and not str(Path(utilsFile).resolve()).startswith(str(PROJECT_ROOT.resolve())):
        del sys.modules["utils"]

from utils.mano.anchors import buildFlatHandAnchorTemplate, loadFlatHandMano


def _run_internal(script_path: Path, forwarded_args: list[str]) -> None:
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.path = [path for path in old_path if path != projectRootStr]
        sys.path.insert(0, projectRootStr)
        sys.argv = [str(script_path), *forwarded_args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path = old_path


def _save_indices(*, mano_path: str, output_path: str, hand_side: str) -> None:
    verts, joints, faces = loadFlatHandMano(manoPath=mano_path, handSide=hand_side)
    template = buildFlatHandAnchorTemplate(verts=verts, joints=joints, faces=faces)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(template.indexOrder, dtype=np.int64))
    print(f"Saved semantic-point index file to: {output}")
    print(f"point_count={len(template.indexOrder)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build repository assets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    save_parser = subparsers.add_parser("save-indices", help="Save the 100-point semantic index order")
    save_parser.add_argument("--mano-path", type=str, default="assets/mano")
    save_parser.add_argument("--output-path", type=str, default="assets/part_ik_hand_index_100.npy")
    save_parser.add_argument("--hand-side", type=str, default="right")

    prior_parser = subparsers.add_parser("axis-prior", help="Build the roll-axis prior used by single-step IK")
    prior_parser.add_argument("--mano-path", type=str, default="assets/mano")
    prior_parser.add_argument("--output-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")

    sample_parser = subparsers.add_parser("demo-sample", help="Build the local ring/joint demo sample")
    sample_parser.add_argument("--mano-path", type=str, default=None)
    sample_parser.add_argument("--sample-index-path", type=str, default="assets/part_ik_hand_index_100.npy")
    sample_parser.add_argument("--axis-prior-path", type=str, default="assets/mano_flat_hand_axis_prior.npy")
    sample_parser.add_argument("--output-path", type=str, default="outputs/ring_joint_demo.npy")

    all_parser = subparsers.add_parser("all", help="Build the semantic index file and axis prior together")
    all_parser.add_argument("--mano-path", type=str, default="assets/mano")
    all_parser.add_argument("--hand-side", type=str, default="right")
    all_parser.add_argument("--index-output", type=str, default="assets/part_ik_hand_index_100.npy")
    all_parser.add_argument("--axis-prior-output", type=str, default="assets/mano_flat_hand_axis_prior.npy")

    args = parser.parse_args()
    if args.command == "save-indices":
        _save_indices(mano_path=args.mano_path, output_path=args.output_path, hand_side=args.hand_side)
        return
    if args.command == "axis-prior":
        _run_internal(
            PROJECT_ROOT / "methods" / "single_ik" / "visualize.py",
            ["axis-prior", "--mano-path", args.mano_path, "--output-path", args.output_path],
        )
        return
    if args.command == "demo-sample":
        _run_internal(
            PROJECT_ROOT / "scripts" / "build_demo_sample.py",
            [
                "--mano-path", str(args.mano_path) if args.mano_path is not None else "",
                "--sample-index-path", args.sample_index_path,
                "--axis-prior-path", args.axis_prior_path,
                "--output-path", args.output_path,
            ] if args.mano_path is not None else [
                "--sample-index-path", args.sample_index_path,
                "--axis-prior-path", args.axis_prior_path,
                "--output-path", args.output_path,
            ],
        )
        return
    if args.command == "all":
        _save_indices(mano_path=args.mano_path, output_path=args.index_output, hand_side=args.hand_side)
        _run_internal(
            PROJECT_ROOT / "methods" / "single_ik" / "visualize.py",
            ["axis-prior", "--mano-path", args.mano_path, "--output-path", args.axis_prior_output],
        )
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
