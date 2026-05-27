"""Independent MANO fitting utilities used by refinement comparison."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..common.geometry import kabsch, rotmat_to_axis_angle


MANO_FIT_LR = 1.0e-3
MANO_FIT_EARLY_STOP = 1.0e-6
MANO_FIT_OPTIMIZERS = {"adam", "lbfgs"}
MANO_FIT_LBFGS_HISTORY = 10

LEFT_ROT_SLICE = slice(0, 3)
LEFT_POSE_SLICE = slice(3, 48)
LEFT_TRANS_SLICE = slice(48, 51)
LEFT_SHAPE_SLICE = slice(51, 61)
RIGHT_ROT_SLICE = slice(61, 64)
RIGHT_POSE_SLICE = slice(64, 109)
RIGHT_TRANS_SLICE = slice(109, 112)
RIGHT_SHAPE_SLICE = slice(112, 122)


def _to_numpy_float32(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def build_dual_hand_mano_tensor(
    *,
    left_full_mano: torch.Tensor,
    right_full_mano: torch.Tensor,
) -> torch.Tensor:
    if left_full_mano.shape != right_full_mano.shape:
        raise ValueError(
            f"Left and right MANO shapes must match, got {tuple(left_full_mano.shape)} / {tuple(right_full_mano.shape)}"
        )
    if left_full_mano.shape[-1] != 61:
        raise ValueError(f"Single-hand MANO tensors must have last dimension 61, got {left_full_mano.shape[-1]}")
    return torch.cat([left_full_mano, right_full_mano], dim=-1)


def split_dual_hand_mano(
    mano_params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mano_params.shape[-1] != 122:
        raise ValueError(f"Dual-hand MANO tensors must have last dimension 122, got {mano_params.shape[-1]}")
    return mano_params[..., :61], mano_params[..., 61:]


def decode_mano_params_to_hand_verts_joints(
    *,
    mano_params: torch.Tensor,
    mano_layer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mano_params.ndim != 3 or mano_params.shape[-1] != 122:
        raise ValueError(f"mano_params must have shape [B,T,122], got {tuple(mano_params.shape)}")
    batch_size, time_count, _ = mano_params.shape
    left_params, right_params = split_dual_hand_mano(mano_params)

    def _decode_side(side_params: torch.Tensor, hand_side: str) -> tuple[torch.Tensor, torch.Tensor]:
        flat = side_params.reshape(batch_size * time_count, 61)
        output = mano_layer[hand_side](
            global_orient=flat[:, 0:3],
            hand_pose=flat[:, 3:48],
            transl=flat[:, 48:51],
            betas=flat[:, 51:61],
        )
        return (
            output.vertices.reshape(batch_size, time_count, 778, 3),
            output.joints.reshape(batch_size, time_count, 16, 3),
        )

    left_verts, left_joints = _decode_side(left_params, "left")
    right_verts, right_joints = _decode_side(right_params, "right")
    return left_verts, right_verts, left_joints, right_joints


def mano_param_list_to_tensor(
    *,
    mano_param_list: List[Dict],
    device: torch.device,
) -> torch.Tensor:
    batch_list: list[np.ndarray] = []
    for mano_param in mano_param_list:
        left_rot = _to_numpy_float32(mano_param["left"]["rot"])
        left_pose = _to_numpy_float32(mano_param["left"]["pose"])
        left_trans = _to_numpy_float32(mano_param["left"]["trans"])
        left_shape = _to_numpy_float32(mano_param["left"]["shape"])
        right_rot = _to_numpy_float32(mano_param["right"]["rot"])
        right_pose = _to_numpy_float32(mano_param["right"]["pose"])
        right_trans = _to_numpy_float32(mano_param["right"]["trans"])
        right_shape = _to_numpy_float32(mano_param["right"]["shape"])
        time_count = left_rot.shape[0]
        left_shape_full = np.repeat(left_shape, time_count, axis=0) if left_shape.shape[0] == 1 else left_shape
        right_shape_full = np.repeat(right_shape, time_count, axis=0) if right_shape.shape[0] == 1 else right_shape
        left_params = np.concatenate([left_rot, left_pose, left_trans, left_shape_full], axis=-1)
        right_params = np.concatenate([right_rot, right_pose, right_trans, right_shape_full], axis=-1)
        batch_list.append(np.concatenate([left_params, right_params], axis=-1))
    return torch.from_numpy(np.stack(batch_list, axis=0)).float().to(device)


def _clone_init_mano_value(
    value,
    *,
    expected_last_dim: int,
    name: str,
    device: torch.device | str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone().float().to(device)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device).clone()
    if tensor.ndim != 2 or tensor.shape[-1] != expected_last_dim:
        raise ValueError(f"initManoParams[{name}] must have shape [T,{expected_last_dim}] or [1,{expected_last_dim}], got {tuple(tensor.shape)}")
    return tensor.requires_grad_(True)


def make_init_mano_params_optimizable(
    init_mano_param: Dict,
    *,
    device: torch.device | str,
) -> Dict:
    required_dims = {"pose": 45, "rot": 3, "trans": 3, "shape": 10}
    output = {"left": {}, "right": {}}
    for hand_type in ("left", "right"):
        if hand_type not in init_mano_param:
            raise ValueError(f"initManoParams is missing hand side: {hand_type}")
        for key, dim in required_dims.items():
            output[hand_type][key] = _clone_init_mano_value(
                init_mano_param[hand_type][key],
                expected_last_dim=dim,
                name=f"{hand_type}.{key}",
                device=device,
            )
    return output


def initialize_mano_params(batch_size: int, *, device: torch.device | str) -> Dict:
    return {
        "left": {
            "pose": torch.zeros((batch_size, 45), requires_grad=True, device=device),
            "rot": torch.zeros((batch_size, 3), requires_grad=True, device=device),
            "trans": torch.zeros((batch_size, 3), requires_grad=True, device=device),
            "shape": torch.zeros((1, 10), requires_grad=True, device=device),
        },
        "right": {
            "pose": torch.zeros((batch_size, 45), requires_grad=True, device=device),
            "rot": torch.zeros((batch_size, 3), requires_grad=True, device=device),
            "trans": torch.zeros((batch_size, 3), requires_grad=True, device=device),
            "shape": torch.zeros((1, 10), requires_grad=True, device=device),
        },
    }


def _prepare_point_weight_tensor(
    *,
    point_weights: Optional[torch.Tensor],
    hand_keypoints: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if point_weights is None:
        return None
    if point_weights.ndim != 1 or int(point_weights.shape[0]) != int(hand_keypoints):
        raise ValueError(f"pointWeights must have shape [{hand_keypoints}], got {tuple(point_weights.shape)}")
    return point_weights.to(device).view(1, hand_keypoints, 1)


def build_anatomy_loss_config(
    *,
    mano_path: str,
    device: torch.device,
    weight: float,
) -> Optional[Dict[str, object]]:
    if weight <= 0.0:
        return None
    from manotorch.anatomy_loss import AnatomyConstraintLossEE
    from manotorch.axislayer import AxisLayerFK
    from manotorch.manolayer import ManoLayer

    left_layer = ManoLayer(side="left", use_pca=False, flat_hand_mean=True, ncomps=45, mano_assets_root=mano_path).to(device)
    right_layer = ManoLayer(side="right", use_pca=False, flat_hand_mean=True, ncomps=45, mano_assets_root=mano_path).to(device)
    axis_left = AxisLayerFK(side="left", mano_assets_root=mano_path).to(device)
    axis_right = AxisLayerFK(side="right", mano_assets_root=mano_path).to(device)
    for module in (left_layer, right_layer, axis_left, axis_right):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    anatomy_loss = AnatomyConstraintLossEE(reduction="mean").to(device)
    anatomy_loss.setup()
    for param in anatomy_loss.parameters():
        param.requires_grad_(False)
    return {
        "weight": float(weight),
        "loss_fn": anatomy_loss,
        "axis_layer_left": axis_left,
        "axis_layer_right": axis_right,
        "mano_layer": {"left": left_layer, "right": right_layer},
        "hands_mean_left": left_layer.th_hands_mean.detach().clone(),
        "hands_mean_right": right_layer.th_hands_mean.detach().clone(),
    }


def _prepare_anatomy_loss_pack(
    *,
    anatomy_loss_config: Optional[Dict[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    if anatomy_loss_config is None:
        return {"weight": 0.0}
    anatomy_weight = float(anatomy_loss_config.get("weight", 0.0))
    if anatomy_weight <= 0.0:
        return {"weight": 0.0}
    return {
        "weight": anatomy_weight,
        "loss_fn": anatomy_loss_config["loss_fn"],
        "axis_left": anatomy_loss_config["axis_layer_left"],
        "axis_right": anatomy_loss_config["axis_layer_right"],
        "mano_layer": anatomy_loss_config["mano_layer"],
        "hands_mean_left": anatomy_loss_config["hands_mean_left"].to(device),
        "hands_mean_right": anatomy_loss_config["hands_mean_right"].to(device),
    }


def _build_mano_optimizer(
    *,
    optimizer_name: str,
    params: List[torch.Tensor],
    steps: int,
) -> optim.Optimizer:
    if optimizer_name == "adam":
        return optim.Adam(params, lr=MANO_FIT_LR)
    if optimizer_name == "lbfgs":
        return optim.LBFGS(
            params,
            lr=MANO_FIT_LR,
            max_iter=max(int(steps), 1),
            history_size=MANO_FIT_LBFGS_HISTORY,
            line_search_fn="strong_wolfe",
        )
    raise ValueError(f"Unsupported optimizerName: {optimizer_name}")


def get_hand_verts_joints(
    mano_layer: Dict,
    hand_type: str,
    mano_dict: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = mano_dict[hand_type]["rot"].shape[0]
    output = mano_layer[hand_type](
        global_orient=mano_dict[hand_type]["rot"],
        hand_pose=mano_dict[hand_type]["pose"],
        betas=torch.tile(mano_dict[hand_type]["shape"], (batch_size, 1)),
        transl=mano_dict[hand_type]["trans"],
    )
    return output.vertices, output.joints


def _compute_anatomy_loss(
    *,
    mano_params: Dict,
    anatomy_pack: Dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    anatomy_loss_fn = anatomy_pack["loss_fn"]
    anatomy_axis_left = anatomy_pack["axis_left"]
    anatomy_axis_right = anatomy_pack["axis_right"]
    anatomy_mano_layer = anatomy_pack["mano_layer"]
    left_mean = anatomy_pack["hands_mean_left"]
    right_mean = anatomy_pack["hands_mean_right"]

    left_pose_adj = mano_params["left"]["pose"] - left_mean
    right_pose_adj = mano_params["right"]["pose"] - right_mean
    left_full_pose = torch.cat([mano_params["left"]["rot"], left_pose_adj], dim=-1)
    right_full_pose = torch.cat([mano_params["right"]["rot"], right_pose_adj], dim=-1)
    left_shape = mano_params["left"]["shape"]
    right_shape = mano_params["right"]["shape"]
    if left_shape.shape[0] == 1 and left_full_pose.shape[0] != 1:
        left_shape = left_shape.repeat(left_full_pose.shape[0], 1)
    if right_shape.shape[0] == 1 and right_full_pose.shape[0] != 1:
        right_shape = right_shape.repeat(right_full_pose.shape[0], 1)
    left_output = anatomy_mano_layer["left"](left_full_pose, left_shape)
    right_output = anatomy_mano_layer["right"](right_full_pose, right_shape)
    _, _, left_euler = anatomy_axis_left(left_output.transforms_abs)
    _, _, right_euler = anatomy_axis_right(right_output.transforms_abs)
    return (anatomy_loss_fn(left_euler) + anatomy_loss_fn(right_euler)).to(device)


def _compute_mano_fit_loss(
    *,
    mano_layer: Dict,
    mano_params: Dict,
    joints_left_ref: torch.Tensor,
    joints_right_ref: torch.Tensor,
    idx_left: Optional[np.ndarray],
    idx_right: Optional[np.ndarray],
    batch_index: int,
    weight_tensor: Optional[torch.Tensor],
    extra_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]],
    anatomy_pack: Dict[str, object],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    verts_left, joints_left = get_hand_verts_joints(mano_layer, "left", mano_params)
    verts_right, joints_right = get_hand_verts_joints(mano_layer, "right", mano_params)
    joints_left_sampled = verts_left[:, idx_left] if idx_left is not None else joints_left
    joints_right_sampled = verts_right[:, idx_right] if idx_right is not None else joints_right
    if weight_tensor is None:
        mse_loss = nn.MSELoss()(joints_left_sampled, joints_left_ref[batch_index]) + nn.MSELoss()(joints_right_sampled, joints_right_ref[batch_index])
    else:
        diff_left = joints_left_sampled - joints_left_ref[batch_index]
        diff_right = joints_right_sampled - joints_right_ref[batch_index]
        mse_loss = (diff_left.square() * weight_tensor).mean() + (diff_right.square() * weight_tensor).mean()
    if extra_loss_fn is not None:
        extra_loss = extra_loss_fn(verts_left, verts_right, batch_index)
        if extra_loss is not None:
            mse_loss = mse_loss + extra_loss
    if anatomy_pack.get("weight", 0.0) > 0.0:
        anatomy_loss = _compute_anatomy_loss(mano_params=mano_params, anatomy_pack=anatomy_pack, device=device)
        mse_loss = mse_loss + float(anatomy_pack["weight"]) * anatomy_loss
    return mse_loss, verts_left, verts_right


def _root_align_single_hand(
    mano_params: Dict,
    mano_layer: Dict,
    pred_points: torch.Tensor,
    point_indices: Optional[np.ndarray],
    *,
    hand_keypoints: int,
    hand_type: str,
) -> Dict:
    verts, joints = get_hand_verts_joints(mano_layer, hand_type, mano_params)
    source = verts[:, point_indices] if point_indices is not None else joints
    if point_indices is None and hand_keypoints == 100:
        raise ValueError("100-point alignment requires explicit point_indices")
    source_center = source.mean(dim=1)
    target_center = pred_points.mean(dim=1)
    rotation = kabsch(source, pred_points)
    translation = target_center - torch.matmul(source_center.unsqueeze(1), rotation).squeeze(1)
    mano_params[hand_type]["rot"] = rotmat_to_axis_angle(rotation).detach().clone().requires_grad_(True)
    mano_params[hand_type]["trans"] = translation.detach().clone().requires_grad_(True)
    return mano_params


def _root_align(
    mano_params: Dict,
    mano_layer: Dict,
    left_pred_points: torch.Tensor,
    right_pred_points: torch.Tensor,
    left_point_indices: Optional[np.ndarray],
    right_point_indices: Optional[np.ndarray],
    *,
    hand_keypoints: int,
) -> Dict:
    mano_params = _root_align_single_hand(
        mano_params,
        mano_layer,
        left_pred_points,
        left_point_indices,
        hand_keypoints=hand_keypoints,
        hand_type="left",
    )
    return _root_align_single_hand(
        mano_params,
        mano_layer,
        right_pred_points,
        right_point_indices,
        hand_keypoints=hand_keypoints,
        hand_type="right",
    )


def _fit_mano_single(
    *,
    mano_layer: Dict,
    pred_horizon: int,
    joints_left_ref: torch.Tensor,
    joints_right_ref: torch.Tensor,
    idx_left: Optional[np.ndarray],
    idx_right: Optional[np.ndarray],
    hand_keypoints: int,
    steps: int,
    verbose: bool,
    init_mano_params: Optional[List[Dict]],
    align_root: bool,
    extra_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]],
    weight_tensor: Optional[torch.Tensor],
    anatomy_pack: Dict[str, object],
    batch_index: int,
    optimizer_name: str,
    device: torch.device,
) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        mano_params = (
            initialize_mano_params(pred_horizon, device=device)
            if init_mano_params is None
            else make_init_mano_params_optimizable(init_mano_params[batch_index], device=device)
        )
        if align_root:
            mano_params = _root_align(
                mano_params,
                mano_layer,
                joints_left_ref[batch_index].detach(),
                joints_right_ref[batch_index].detach(),
                idx_left,
                idx_right,
                hand_keypoints=hand_keypoints,
            )
        params = [
            mano_params["left"]["pose"],
            mano_params["left"]["rot"],
            mano_params["left"]["trans"],
            mano_params["left"]["shape"],
            mano_params["right"]["pose"],
            mano_params["right"]["rot"],
            mano_params["right"]["trans"],
            mano_params["right"]["shape"],
        ]
        optimizer = _build_mano_optimizer(optimizer_name=optimizer_name, params=params, steps=steps)
        if optimizer_name == "adam":
            for step_index in range(steps + 1):
                loss, verts_left, verts_right = _compute_mano_fit_loss(
                    mano_layer=mano_layer,
                    mano_params=mano_params,
                    joints_left_ref=joints_left_ref,
                    joints_right_ref=joints_right_ref,
                    idx_left=idx_left,
                    idx_right=idx_right,
                    batch_index=batch_index,
                    weight_tensor=weight_tensor,
                    extra_loss_fn=extra_loss_fn,
                    anatomy_pack=anatomy_pack,
                    device=device,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step_index == steps and verbose:
                    print(f"  MANO fit loss: {loss.item():.6f}", flush=True)
                if loss < MANO_FIT_EARLY_STOP:
                    break
            return mano_params, verts_left, verts_right

        def closure():
            optimizer.zero_grad()
            loss, _, _ = _compute_mano_fit_loss(
                mano_layer=mano_layer,
                mano_params=mano_params,
                joints_left_ref=joints_left_ref,
                joints_right_ref=joints_right_ref,
                idx_left=idx_left,
                idx_right=idx_right,
                batch_index=batch_index,
                weight_tensor=weight_tensor,
                extra_loss_fn=extra_loss_fn,
                anatomy_pack=anatomy_pack,
                device=device,
            )
            loss.backward()
            return loss

        loss = optimizer.step(closure) if steps > 0 else closure().detach()
        verts_left, _ = get_hand_verts_joints(mano_layer, "left", mano_params)
        verts_right, _ = get_hand_verts_joints(mano_layer, "right", mano_params)
        if verbose:
            print(f"  MANO fit loss: {loss.item():.6f}", flush=True)
        return mano_params, verts_left, verts_right


def _optimize_mano_batch(
    joints_left_ref: torch.Tensor,
    joints_right_ref: torch.Tensor,
    mano_layer: Dict,
    pred_horizon: int,
    idx_left: Optional[np.ndarray] = None,
    idx_right: Optional[np.ndarray] = None,
    hand_keypoints: int = 100,
    steps: int = 4000,
    verbose: bool = True,
    init_mano_params: Optional[List[Dict]] = None,
    align_root: bool = True,
    extra_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = None,
    point_weights: Optional[torch.Tensor] = None,
    anatomy_loss_config: Optional[Dict[str, object]] = None,
    optimizer_name: str = "adam",
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    batch_size = int(joints_left_ref.shape[0])
    device = joints_left_ref.device
    if init_mano_params is not None and len(init_mano_params) != batch_size:
        raise ValueError(f"initManoParams must have length {batch_size}, got {len(init_mano_params)}")
    if optimizer_name not in MANO_FIT_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizerName: {optimizer_name}")
    weight_tensor = _prepare_point_weight_tensor(point_weights=point_weights, hand_keypoints=hand_keypoints, device=device)
    anatomy_pack = _prepare_anatomy_loss_pack(anatomy_loss_config=anatomy_loss_config, device=device)
    mano_params_list: list[Dict] = []
    verts_left_list: list[np.ndarray] = []
    verts_right_list: list[np.ndarray] = []
    for batch_index in range(batch_size):
        if verbose:
            print(f"Fitting MANO for batch item {batch_index}", flush=True)
        mano_params, verts_left, verts_right = _fit_mano_single(
            mano_layer=mano_layer,
            pred_horizon=pred_horizon,
            joints_left_ref=joints_left_ref,
            joints_right_ref=joints_right_ref,
            idx_left=idx_left,
            idx_right=idx_right,
            hand_keypoints=hand_keypoints,
            steps=steps,
            verbose=verbose,
            init_mano_params=init_mano_params,
            align_root=align_root,
            extra_loss_fn=extra_loss_fn,
            weight_tensor=weight_tensor,
            anatomy_pack=anatomy_pack,
            batch_index=batch_index,
            optimizer_name=optimizer_name,
            device=device,
        )
        mano_params_list.append(mano_params)
        verts_left_list.append(verts_left.detach().cpu().numpy())
        verts_right_list.append(verts_right.detach().cpu().numpy())
    return mano_params_list, np.array(verts_left_list), np.array(verts_right_list)


def fitMano(
    jointsLRef: torch.Tensor,
    jointsRRef: torch.Tensor,
    manoLayer: Dict,
    predHorizon: int,
    idxL: Optional[np.ndarray] = None,
    idxR: Optional[np.ndarray] = None,
    handKeypoints: int = 100,
    steps: int = 4000,
    verbose: bool = True,
    initManoParams: Optional[List[Dict]] = None,
    alignRoot: bool = True,
    extraLossFn: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = None,
    pointWeights: Optional[torch.Tensor] = None,
    anatomyLossConfig: Optional[Dict[str, object]] = None,
    optimizerName: str = "adam",
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    return _optimize_mano_batch(
        joints_left_ref=jointsLRef,
        joints_right_ref=jointsRRef,
        mano_layer=manoLayer,
        pred_horizon=predHorizon,
        idx_left=idxL,
        idx_right=idxR,
        hand_keypoints=handKeypoints,
        steps=steps,
        verbose=verbose,
        init_mano_params=initManoParams,
        align_root=alignRoot,
        extra_loss_fn=extraLossFn,
        point_weights=pointWeights,
        anatomy_loss_config=anatomyLossConfig,
        optimizer_name=optimizerName,
    )


def refineManoFromInit(
    jointsLRef: torch.Tensor,
    jointsRRef: torch.Tensor,
    manoLayer: Dict,
    predHorizon: int,
    initManoParams: List[Dict],
    idxL: Optional[np.ndarray] = None,
    idxR: Optional[np.ndarray] = None,
    handKeypoints: int = 100,
    steps: int = 100,
    verbose: bool = True,
    alignRoot: bool = False,
    extraLossFn: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = None,
    pointWeights: Optional[torch.Tensor] = None,
    anatomyLossConfig: Optional[Dict[str, object]] = None,
    optimizerName: str = "adam",
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    if initManoParams is None:
        raise ValueError("refineManoFromInit requires explicit initManoParams")
    return _optimize_mano_batch(
        joints_left_ref=jointsLRef,
        joints_right_ref=jointsRRef,
        mano_layer=manoLayer,
        pred_horizon=predHorizon,
        idx_left=idxL,
        idx_right=idxR,
        hand_keypoints=handKeypoints,
        steps=steps,
        verbose=verbose,
        init_mano_params=initManoParams,
        align_root=alignRoot,
        extra_loss_fn=extraLossFn,
        point_weights=pointWeights,
        anatomy_loss_config=anatomyLossConfig,
        optimizer_name=optimizerName,
    )
