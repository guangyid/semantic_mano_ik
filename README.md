# semantic_mano_ik

![semantic_ik_overview](assets/semantic_ik.png)

`semantic_mano_ik` is a compact MANO repository built around a fixed set of 100 semantic hand points. It provides single-step IK, direct MANO fitting, refinement from IK initialization, forward point export from MANO, method comparison, sequence visualization, and asset-building utilities.

## Installation

```bash
pip install -r requirements.txt
```

Set the MANO model path before running the scripts:

```bash
export MANO_PATH=/path/to/mano
```

MANO model files are not included in this repository because of the MANO license. Download them from the official website: https://mano.is.tue.mpg.de/ and set `MANO_PATH` to the extracted model directory.


## Quick Start

- build the local demo payload once if `outputs/ring_joint_demo.npy` is not present.
```bash
python -u scripts/build_demo_sample.py --output-path outputs/ring_joint_demo.npy
```
- `single_ik`: estimate MANO from 100 semantic points. 
```bash
python -u methods/single_ik/run_single_ik.py --points-path outputs/ring_joint_demo.npy --hand-side both --output-dir outputs/single_ik
```
- `refine_ik`: start from `single_ik` and continue iterative optimization. 
```bash
python -u methods/refine_ik/refine_from_points.py --input-path outputs/ring_joint_demo.npy --output-dir outputs/refine_ik --export-glb
```
- `mano_fitting`: directly optimize MANO parameters from the same 100-point input. 
```bash
python -u methods/mano_fitting/fit_from_points.py --input-path outputs/ring_joint_demo.npy --output-dir outputs/mano_fitting --export-glb
```
- `single_ik` visualization: inspect the semantic rings, joint centers, and reconstructed hand. 
```bash
python -u methods/single_ik/visualize.py ring-joint --sample-path outputs/ring_joint_demo.npy
```
`assets/sequence_mano.npz` is a MANO motion asset, and its inline preview rendered from that sequence like:

![sequence_visualization](assets/sequence_visualization.gif)

- method comparison: compare `single_ik`, `mano_fitting`, and `refine_ik` on the same payload. 
```bash
python -u scripts/compare_methods.py --input-path outputs/ring_joint_demo.npy --output-dir outputs/compare_methods --export-glb
```


## Assets

- `assets/part_ik_hand_index_100.npy`: fixed 100-point semantic index order
- `assets/mano_flat_hand_axis_prior.npy`: roll-axis prior used by `single_ik`
- `assets/mano_flat_hand_anchor_groups.glb`: flat-hand anchor-group visualization
- `outputs/ring_joint_demo.npy`: generated demo payload created by `scripts/build_demo_sample.py`

## 100 Semantic Points 

- `45` segment-ring points: `15` finger segments, each with `3` ring points.
- `40` joint/tip pair points: `20` joint or tip sites, each with `2` opposite-side surface points.
- `6` wrist-cuff points: wrist boundary anchors for root-frame and translation estimation.
- `9` palm-surface points: extra palm coverage points for layout stability and visualization.

This structured 100-point design follows the semantic anchor setup used in:

```bibtex
@misc{xxx,
  title={THREAD: Joint 2D-3D Generation of Egocentric Hand-Object Interactions},
  author={Guangyi Han, Wei Zhai, Yuhang Yang, Zining Wang, Yang Cao, Zheng-Jun Zha},
  year={2026},
  eprint={26xx.xxxx},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/xxx},
}
```
