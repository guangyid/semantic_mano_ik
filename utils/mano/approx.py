"""Single-pass approximate MANO estimator from fixed ring anchors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import torch
from manotorch.manolayer import ManoLayer

from .anchors import FINGER_JOINTS_21, SMPLX_SEGMENT_ORDER, SEGMENT_PARENTS, buildFlatHandAnchorTemplate, loadFlatHandMano
from ..common.frames import buildSegmentGlobalRotations
from ..common.geometry import axis_angle_to_rotmat, gram_schmidt_frame, normalize, project_to_plane, rotation_between_vectors, rotmat_to_axis_angle


AXIS_PRIOR_MAX_ROLL_RAD = 0.7
AXIS_PRIOR_POINT_WEIGHTS = {
    "ring_mid": 0.25,
    "ring_pos": 1.0,
    "ring_neg": 1.0,
    "prox_pos": 1.25,
    "prox_neg": 1.25,
    "dist_pos": 1.25,
    "dist_neg": 1.25,
}
KNOWN_SHAPE_SCALE_MIN = 0.85
KNOWN_SHAPE_SCALE_MAX = 1.35
JOINT_LINE_SEGMENT_WEIGHTS = {
    1: (1.0, 1.0, 0.35, 0.0),
    2: (0.35, 1.0, 1.0, 0.35),
    3: (0.0, 0.35, 1.0, 1.0),
}
# MANO kintree hand_pose[45] finger order: index -> middle -> pinky -> ring -> thumb
# SMPLX_SEGMENT_ORDER finger order: thumb -> index -> middle -> ring -> pinky
# The mapping below converts SMPLX_SEGMENT_ORDER indices to MANO kintree order.
_SMPLX_TO_MANO_KINTREE_REORDER = [3, 4, 5, 6, 7, 8, 12, 13, 14, 9, 10, 11, 0, 1, 2]


@dataclass
class ApproxManoEstimate:
    rootRot: torch.Tensor
    rootTrans: torch.Tensor
    handPose: torch.Tensor
    fullMano: torch.Tensor
    fallbackCount: int


@dataclass
class ApproxRestState:
    rootPoints: dict[str, torch.Tensor]
    rootPatchPoints: torch.Tensor
    segmentPoints: dict[str, dict[str, torch.Tensor]]
    jointPairs: dict[str, dict[str, torch.Tensor]]
    jointCenters: dict[str, torch.Tensor]
    fingerLengths: dict[str, torch.Tensor]
    wristJoint: torch.Tensor
    manoJoints: torch.Tensor


class ApproxForwardManoEstimator:
    """Approximate MANO estimator using fixed flat-hand ring semantics."""

    def __init__(
        self,
        *,
        manoPath: str = "assets/mano",
        handSide: str = "right",
        device: str = "cpu",
        jointWeightMap: dict[str, float] | None = None,
        axisPriorPath: str | None = None,
    ):
        verts, joints, faces = loadFlatHandMano(manoPath=manoPath, handSide=handSide)
        self.template = buildFlatHandAnchorTemplate(verts=verts, joints=joints, faces=faces)
        self.device = torch.device(device)
        self.manoLayer = ManoLayer(
            side=handSide,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            mano_assets_root=manoPath,
        ).to(self.device)
        manoFile = self._resolve_mano_model_path(manoPath=manoPath, handSide=handSide)
        with open(manoFile, "rb") as fileObj:
            manoData = pickle.load(fileObj, encoding="latin1")
        handsMean = manoData.get("hands_mean")
        if handsMean is None:
            raise ValueError(f"{manoFile} is missing hands_mean")
        self.handPoseMean = torch.tensor(handsMean, dtype=torch.float32, device=self.device).view(1, 45)
        self.jointWeightMap = jointWeightMap or {}
        self.jointWeightDenom = self._resolveJointWeightDenom()
        self.axisPrior = self._loadAxisPrior(axisPriorPath=axisPriorPath, handSide=handSide)
        self.indexOrder = torch.tensor(self.template.indexOrder, dtype=torch.long, device=self.device)
        self.indexToOffset = {int(index): offset for offset, index in enumerate(self.template.indexOrder)}
        self.restRootPoints = {
            name: torch.tensor(self.template.verts[index], dtype=torch.float32, device=self.device)
            for name, index in self.template.rootPointMap.items()
        }
        self.rootPatchVertexIndices = self._buildRootPatchVertexIndices()
        self.restRootPatchPoints = torch.tensor(
            self.template.verts[self.rootPatchVertexIndices],
            dtype=torch.float32,
            device=self.device,
        )
        self.restSegmentPoints = {
            name: {
                key: torch.tensor(self.template.verts[index], dtype=torch.float32, device=self.device)
                for key, index in ring.items()
            }
            for name, ring in self.template.segmentRingMap.items()
        }
        self.restJointPairs = {
            name: {
                key: torch.tensor(self.template.verts[index], dtype=torch.float32, device=self.device)
                for key, index in pair.items()
            }
            for name, pair in self.template.jointPairMap.items()
        }
        self.restJointCenters = self._buildRestJointCenters()
        self.restFingerLengths = self._buildRestFingerLengths()
        self.restWristJoint = torch.tensor(self.template.joints[0], dtype=torch.float32, device=self.device)
        self.restManoJoints = torch.tensor(self.template.joints, dtype=torch.float32, device=self.device)
        self.segmentEndpointMap = self._buildSegmentEndpointMap()
        self.meanRestState = ApproxRestState(
            rootPoints=self.restRootPoints,
            rootPatchPoints=self.restRootPatchPoints,
            segmentPoints=self.restSegmentPoints,
            jointPairs=self.restJointPairs,
            jointCenters=self.restJointCenters,
            fingerLengths=self.restFingerLengths,
            wristJoint=self.restWristJoint,
            manoJoints=self.restManoJoints,
        )

    def _resolveJointWeight(self, jointName: str) -> float:
        if not self.jointWeightMap:
            return 1.0
        defaultWeight = float(self.jointWeightMap.get("default", 1.0))
        return float(self.jointWeightMap.get(jointName, defaultWeight))

    def _resolveJointWeightDenom(self) -> float:
        if not self.jointWeightMap:
            return 1.0
        weights = [
            self._resolveJointWeight("joint_1"),
            self._resolveJointWeight("joint_2"),
            self._resolveJointWeight("joint_3"),
            self._resolveJointWeight("tip"),
            self._resolveJointWeight("default"),
        ]
        denom = max(weights)
        return denom if denom > 0.0 else 1.0

    def _resolveSegmentWeight(self, segmentName: str) -> float:
        if not self.jointWeightMap:
            return 1.0
        segmentIdx = int(segmentName.rsplit("_", 1)[-1])
        if segmentIdx == 1:
            proximal = "joint_1"
            distal = "joint_2"
        elif segmentIdx == 2:
            proximal = "joint_2"
            distal = "joint_3"
        else:
            proximal = "joint_3"
            distal = "tip"
        return 0.5 * (self._resolveJointWeight(proximal) + self._resolveJointWeight(distal))

    def _buildRootPatchVertexIndices(self) -> list[int]:
        ordered = (
            self.template.wristCuffMap["dorsal"]
            + self.template.wristCuffMap["palmar"]
            + self.template.palmSurfaceMap["dorsal"]
            + self.template.palmSurfaceMap["palmar"]
            + list(self.template.rootPointMap.values())
        )
        uniqueIndices: list[int] = []
        seen: set[int] = set()
        for index in ordered:
            if int(index) in seen:
                continue
            uniqueIndices.append(int(index))
            seen.add(int(index))
        return uniqueIndices

    def _pairCenter(self, pairPoints: dict[str, torch.Tensor]) -> torch.Tensor:
        return 0.5 * (pairPoints["pos"] + pairPoints["neg"])

    def _wristCenter(self, rootPoints: dict[str, torch.Tensor]) -> torch.Tensor:
        return 0.5 * (rootPoints["wrist_radial"] + rootPoints["wrist_ulnar"])

    def _buildSegmentEndpointMap(self) -> dict[str, tuple[str, str]]:
        endpointMap: dict[str, tuple[str, str]] = {}
        for segmentName in SMPLX_SEGMENT_ORDER:
            fingerName = segmentName.split("_")[0]
            segmentIdx = int(segmentName.rsplit("_", 1)[-1])
            proximalName = f"{fingerName}_joint_{segmentIdx}"
            distalName = f"{fingerName}_tip" if segmentIdx == 3 else f"{fingerName}_joint_{segmentIdx + 1}"
            endpointMap[segmentName] = (proximalName, distalName)
        return endpointMap

    def _getManoRestBoneAxis(self, restState: ApproxRestState, segmentName: str) -> torch.Tensor:
        fingerName = segmentName.split("_")[0]
        segIdx = int(segmentName.rsplit("_", 1)[-1]) - 1
        jointIds = FINGER_JOINTS_21[fingerName]
        proximal = restState.manoJoints[..., jointIds[segIdx], :]
        distal = restState.manoJoints[..., jointIds[segIdx + 1], :]
        return distal - proximal

    def _buildRestJointCenters(self) -> dict[str, torch.Tensor]:
        centers: dict[str, torch.Tensor] = {}
        for fingerName in FINGER_JOINTS_21.keys():
            for level in ("joint_1", "joint_2", "joint_3", "tip"):
                key = f"{fingerName}_{level}"
                centers[key] = self._pairCenter(self.restJointPairs[key])
        return centers

    def _buildRestFingerLengths(self) -> dict[str, torch.Tensor]:
        lengths: dict[str, torch.Tensor] = {}
        for fingerName in FINGER_JOINTS_21.keys():
            j1 = self.restJointCenters[f"{fingerName}_joint_1"]
            j2 = self.restJointCenters[f"{fingerName}_joint_2"]
            j3 = self.restJointCenters[f"{fingerName}_joint_3"]
            tip = self.restJointCenters[f"{fingerName}_tip"]
            segLens = torch.stack([
                torch.norm(j2 - j1),
                torch.norm(j3 - j2),
                torch.norm(tip - j3),
            ], dim=0)
            lengths[fingerName] = segLens
        return lengths

    def _buildFingerLengthsFromJointCenters(
        self,
        jointCenters: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        lengths: dict[str, torch.Tensor] = {}
        for fingerName in FINGER_JOINTS_21.keys():
            j1 = jointCenters[f"{fingerName}_joint_1"]
            j2 = jointCenters[f"{fingerName}_joint_2"]
            j3 = jointCenters[f"{fingerName}_joint_3"]
            tip = jointCenters[f"{fingerName}_tip"]
            lengths[fingerName] = torch.stack(
                [
                    torch.norm(j2 - j1, dim=-1),
                    torch.norm(j3 - j2, dim=-1),
                    torch.norm(tip - j3, dim=-1),
                ],
                dim=-1,
            )
        return lengths

    def _buildRestStateFromFlatOutputs(
        self,
        *,
        verts: torch.Tensor,
        joints: torch.Tensor,
    ) -> ApproxRestState:
        rootPoints = {
            name: verts[..., int(index), :]
            for name, index in self.template.rootPointMap.items()
        }
        rootPatchPoints = verts[..., self.rootPatchVertexIndices, :]
        segmentPoints = {
            name: {
                key: verts[..., int(index), :]
                for key, index in ring.items()
            }
            for name, ring in self.template.segmentRingMap.items()
        }
        jointPairs = {
            name: {
                key: verts[..., int(index), :]
                for key, index in pair.items()
            }
            for name, pair in self.template.jointPairMap.items()
        }
        jointCenters = {
            name: self._pairCenter(pair)
            for name, pair in jointPairs.items()
        }
        fingerLengths = self._buildFingerLengthsFromJointCenters(jointCenters)
        return ApproxRestState(
            rootPoints=rootPoints,
            rootPatchPoints=rootPatchPoints,
            segmentPoints=segmentPoints,
            jointPairs=jointPairs,
            jointCenters=jointCenters,
            fingerLengths=fingerLengths,
            wristJoint=joints[..., 0, :],
            manoJoints=joints,
        )

    def _buildRestStateFromShapeBetas(
        self,
        *,
        shapeBetas: torch.Tensor,
    ) -> ApproxRestState:
        flatBetas = shapeBetas.reshape(-1, shapeBetas.shape[-1]).to(device=self.device, dtype=torch.float32)
        flatPose = torch.zeros((flatBetas.shape[0], 48), dtype=flatBetas.dtype, device=flatBetas.device)
        output = self.manoLayer(flatPose, flatBetas)
        verts = output.verts.reshape(*shapeBetas.shape[:-1], 778, 3)
        joints = output.joints.reshape(*shapeBetas.shape[:-1], output.joints.shape[1], 3)
        return self._buildRestStateFromFlatOutputs(verts=verts, joints=joints)

    def _resolveRestState(
        self,
        *,
        shapeBetas: torch.Tensor | None,
    ) -> ApproxRestState:
        if shapeBetas is None:
            return self.meanRestState
        if shapeBetas.shape[-1] != 10:
            raise ValueError(f"knownShape must have shape [...,10], got {tuple(shapeBetas.shape)}")
        return self._buildRestStateFromShapeBetas(shapeBetas=shapeBetas)

    def _loadAxisPrior(
        self,
        *,
        axisPriorPath: str | None,
        handSide: str,
    ) -> dict[str, dict[str, float]] | None:
        if axisPriorPath is None:
            return None
        priorPath = Path(axisPriorPath)
        if not priorPath.is_file():
            raise FileNotFoundError(f"single_ik axis prior path does not exist: {priorPath}")
        data = np.load(str(priorPath), allow_pickle=True).item()
        if handSide not in data:
            raise ValueError(f"Axis prior is missing handSide={handSide}")
        prior = data[handSide]
        if not isinstance(prior, dict):
            raise ValueError("Invalid axis prior format: expected a dict")
        return prior

    def _build_axis_basis(
        self,
        *,
        axis: torch.Tensor,
        yHint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        yProj = project_to_plane(yHint, axis)
        yNorm = yProj.norm(dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=axis.dtype, device=axis.device)
        fallbackAlt = torch.tensor([0.0, 1.0, 0.0], dtype=axis.dtype, device=axis.device)
        dot = torch.sum(axis * fallback, dim=-1, keepdim=True).abs()
        fallbackVec = torch.where(dot > 0.9, fallbackAlt, fallback)
        fallbackProj = project_to_plane(fallbackVec, axis)
        yAxis = normalize(torch.where(yNorm > 1.0e-6, yProj, fallbackProj))
        zAxis = normalize(torch.cross(axis, yAxis, dim=-1))
        return yAxis, zAxis

    def _computeAxisPriorDelta(
        self,
        *,
        segmentName: str,
        observedSegmentPoints: dict[str, dict[str, torch.Tensor]],
        observedJointPairs: dict[str, dict[str, torch.Tensor]],
        observedAxis: torch.Tensor,
        baseRotation: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.axisPrior is None:
            return None
        if segmentName not in self.axisPrior:
            return None
        prior = self.axisPrior[segmentName]
        proximalName, distalName = self.segmentEndpointMap[segmentName]
        proximalCenter = self._pairCenter(observedJointPairs[proximalName])
        axis = normalize(observedAxis)
        if torch.any(torch.isnan(axis)):
            return None
        ring = observedSegmentPoints[segmentName]
        proxPair = observedJointPairs[proximalName]
        distPair = observedJointPairs[distalName]
        yAxis, zAxis = self._build_axis_basis(axis=axis, yHint=baseRotation[..., :, 1])
        pointMap = {
            "ring_mid": ring["mid"],
            "ring_pos": ring["pos"],
            "ring_neg": ring["neg"],
            "prox_pos": proxPair["pos"],
            "prox_neg": proxPair["neg"],
            "dist_pos": distPair["pos"],
            "dist_neg": distPair["neg"],
        }
        deltaSin = torch.zeros_like(axis[..., 0])
        deltaCos = torch.zeros_like(axis[..., 0])
        weightSum = torch.zeros_like(axis[..., 0])
        for name, point in pointMap.items():
            if name not in prior:
                continue
            rel = point - proximalCenter
            rel = project_to_plane(rel, axis)
            relNorm = rel.norm(dim=-1)
            valid = relNorm > 1.0e-6
            if not bool(torch.any(valid).item()):
                continue
            yVal = torch.sum(rel * yAxis, dim=-1)
            zVal = torch.sum(rel * zAxis, dim=-1)
            angleVal = torch.atan2(zVal, yVal)
            delta = angleVal - float(prior[name])
            pointWeight = float(AXIS_PRIOR_POINT_WEIGHTS.get(name, 1.0))
            weight = torch.where(valid, relNorm * pointWeight, torch.zeros_like(relNorm))
            deltaSin = deltaSin + weight * torch.sin(delta)
            deltaCos = deltaCos + weight * torch.cos(delta)
            weightSum = weightSum + weight
        if not bool(torch.any(weightSum > 1.0e-6).item()):
            return None
        delta = torch.atan2(deltaSin, deltaCos)
        return torch.where(delta.abs() <= AXIS_PRIOR_MAX_ROLL_RAD, delta, torch.zeros_like(delta))

    def _buildObservedJointCenters(
        self,
        *,
        observedJointPairs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        centers: dict[str, torch.Tensor] = {}
        for fingerName in FINGER_JOINTS_21.keys():
            for level in ("joint_1", "joint_2", "joint_3", "tip"):
                key = f"{fingerName}_{level}"
                centers[key] = self._pairCenter(observedJointPairs[key])
        return centers

    def _computeKnownShapePrescale(
        self,
        *,
        restState: ApproxRestState,
        observedJointPairs: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        observedCenters = self._buildObservedJointCenters(observedJointPairs=observedJointPairs)
        ratios: list[torch.Tensor] = []
        for fingerName in FINGER_JOINTS_21.keys():
            joint1 = observedCenters[f"{fingerName}_joint_1"]
            joint2 = observedCenters[f"{fingerName}_joint_2"]
            joint3 = observedCenters[f"{fingerName}_joint_3"]
            tip = observedCenters[f"{fingerName}_tip"]
            observedLengths = torch.stack(
                [
                    torch.norm(joint2 - joint1, dim=-1),
                    torch.norm(joint3 - joint2, dim=-1),
                    torch.norm(tip - joint3, dim=-1),
                ],
                dim=-1,
            )
            ratios.append(restState.fingerLengths[fingerName] / observedLengths.clamp_min(1.0e-6))
        scaleStack = torch.cat(ratios, dim=-1)
        return scaleStack.median(dim=-1).values.clamp(KNOWN_SHAPE_SCALE_MIN, KNOWN_SHAPE_SCALE_MAX)

    def _applyKnownShapePrescale(
        self,
        *,
        sampledPoints: torch.Tensor,
        restState: ApproxRestState,
        observedRootPatchPoints: torch.Tensor,
        observedJointPairs: dict[str, dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._computeKnownShapePrescale(
            restState=restState,
            observedJointPairs=observedJointPairs,
        )
        scaleCenter = observedRootPatchPoints.mean(dim=-2, keepdim=True)
        scaledPoints = scaleCenter + scale.unsqueeze(-1).unsqueeze(-1) * (sampledPoints - scaleCenter)
        return scaledPoints, scale

    def _fitWeightedLineDirection(
        self,
        *,
        points: torch.Tensor,
        weights: tuple[float, ...],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        weightTensor = torch.tensor(weights, dtype=points.dtype, device=points.device)
        viewShape = (*([1] * (points.ndim - 2)), points.shape[-2], 1)
        weightTensor = weightTensor.view(*viewShape)
        weightSum = weightTensor.sum(dim=-2, keepdim=True).clamp_min(1.0e-6)
        center = (points * weightTensor).sum(dim=-2, keepdim=True) / weightSum
        centered = points - center
        cov = torch.matmul((centered * weightTensor).transpose(-1, -2), centered)
        eigVals, eigVecs = torch.linalg.eigh(cov)
        direction = eigVecs[..., :, -1]
        direction = normalize(direction)
        refUnit = normalize(reference)
        sign = torch.sign(torch.sum(direction * refUnit, dim=-1, keepdim=True))
        sign = torch.where(sign >= 0.0, torch.ones_like(sign), -torch.ones_like(sign))
        direction = direction * sign
        valid = eigVals[..., -1] > 1.0e-8
        return torch.where(valid.unsqueeze(-1), direction, refUnit)

    def _buildSegmentAxesFromJointLine(
        self,
        *,
        observedJointPairs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        centers = self._buildObservedJointCenters(observedJointPairs=observedJointPairs)
        segmentAxes: dict[str, torch.Tensor] = {}
        for fingerName in FINGER_JOINTS_21.keys():
            chain = torch.stack(
                [
                    centers[f"{fingerName}_joint_1"],
                    centers[f"{fingerName}_joint_2"],
                    centers[f"{fingerName}_joint_3"],
                    centers[f"{fingerName}_tip"],
                ],
                dim=-2,
            )
            pairAxes = (
                chain[..., 1, :] - chain[..., 0, :],
                chain[..., 2, :] - chain[..., 1, :],
                chain[..., 3, :] - chain[..., 2, :],
            )
            for segIdx in range(3):
                weights = JOINT_LINE_SEGMENT_WEIGHTS[segIdx + 1]
                fittedAxis = self._fitWeightedLineDirection(
                    points=chain,
                    weights=weights,
                    reference=pairAxes[segIdx],
                )
                segmentAxes[f"{fingerName}_segment_{segIdx + 1}"] = fittedAxis
        return segmentAxes

    def _buildObservedSegmentAxes(
        self,
        *,
        observedJointPairs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return self._buildSegmentAxesFromJointLine(observedJointPairs=observedJointPairs)

    def _buildRootFrame(self, rootPoints: dict[str, torch.Tensor]) -> torch.Tensor:
        wristCenter = self._wristCenter(rootPoints)
        xAxis = rootPoints["middle_base"] - wristCenter
        yHint = rootPoints["index_base"] - rootPoints["pinky_base"]
        yAxis = project_to_plane(yHint, xAxis)
        return gram_schmidt_frame(xAxis, yAxis)

    def _resolve_mano_model_path(self, *, manoPath: str, handSide: str) -> Path:
        manoRoot = Path(manoPath)
        manoFile = manoRoot / f"MANO_{handSide.upper()}.pkl"
        if manoFile.is_file():
            return manoFile
        manoFile = manoRoot / "models" / f"MANO_{handSide.upper()}.pkl"
        if manoFile.is_file():
            return manoFile
        raise FileNotFoundError(f"MANO model files were not found: {manoRoot}")

    def _estimateRootRotation(
        self,
        *,
        restState: ApproxRestState,
        observedRootPoints: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        restRootFrame = self._buildRootFrame(restState.rootPoints)
        observedRootFrame = self._buildRootFrame(observedRootPoints)
        return torch.matmul(observedRootFrame, restRootFrame.transpose(-1, -2))

    def _decodeRootJointWithoutTranslation(
        self,
        *,
        rootRotAA: torch.Tensor,
        handPose: torch.Tensor,
        shapeBetas: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fullPose = torch.cat([rootRotAA, handPose], dim=-1)
        flatPose = fullPose.reshape(-1, 48)
        if shapeBetas is None:
            betas = torch.zeros((flatPose.shape[0], 10), dtype=flatPose.dtype, device=flatPose.device)
        else:
            betas = shapeBetas.reshape(-1, 10).to(device=flatPose.device, dtype=flatPose.dtype)
        output = self.manoLayer(flatPose, betas)
        rootJoint = output.joints[:, 0, :]
        verts = output.verts
        return (
            rootJoint.reshape(*fullPose.shape[:-1], 3),
            verts.reshape(*fullPose.shape[:-1], 778, 3),
        )

    def _estimateRootTranslation(
        self,
        *,
        sampledPoints: torch.Tensor,
        rootRotAA: torch.Tensor,
        handPose: torch.Tensor,
        shapeBetas: torch.Tensor | None,
    ) -> torch.Tensor:
        _, decodedVerts = self._decodeRootJointWithoutTranslation(
            rootRotAA=rootRotAA,
            handPose=handPose,
            shapeBetas=shapeBetas,
        )
        decodedSampledPoints = decodedVerts[..., self.indexOrder, :]
        return torch.mean(sampledPoints - decodedSampledPoints, dim=-2)

    def _correctSegmentGlobalRotations(
        self,
        *,
        restState: ApproxRestState,
        segmentGlobalRotations: dict[str, torch.Tensor],
        observedJointPairs: dict[str, dict[str, torch.Tensor]],
        observedSegmentPoints: dict[str, dict[str, torch.Tensor]],
        observedAxes: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        correctedRotations: dict[str, torch.Tensor] = {}
        for segmentName in SMPLX_SEGMENT_ORDER:
            proximalName, distalName = self.segmentEndpointMap[segmentName]
            restAxis = self._pairCenter(restState.jointPairs[distalName]) - self._pairCenter(restState.jointPairs[proximalName])
            observedAxis = observedAxes[segmentName]
            restAxisUnit = normalize(restAxis)
            predictedAxis = torch.matmul(segmentGlobalRotations[segmentName], restAxisUnit.unsqueeze(-1)).squeeze(-1)
            axisCorrection = rotation_between_vectors(predictedAxis, observedAxis)
            segmentWeight = self._resolveSegmentWeight(segmentName) / self.jointWeightDenom
            if segmentWeight != 1.0:
                axisAngle = rotmat_to_axis_angle(axisCorrection)
                axisCorrection = axis_angle_to_rotmat(axisAngle * segmentWeight)
            corrected = torch.matmul(axisCorrection, segmentGlobalRotations[segmentName])
            axisPriorDelta = self._computeAxisPriorDelta(
                segmentName=segmentName,
                observedSegmentPoints=observedSegmentPoints,
                observedJointPairs=observedJointPairs,
                observedAxis=observedAxis,
                baseRotation=corrected,
            )
            if axisPriorDelta is not None:
                axisUnit = normalize(observedAxis)
                axisAngle = axisUnit * axisPriorDelta.unsqueeze(-1)
                rollRot = axis_angle_to_rotmat(axisAngle)
                corrected = torch.matmul(rollRot, corrected)
            correctedRotations[segmentName] = corrected
        return correctedRotations

    def _applyAxisPriorRollCorrection(
        self,
        *,
        segmentName: str,
        baseRotation: torch.Tensor,
        observedAxis: torch.Tensor,
        observedSegmentPoints: dict[str, dict[str, torch.Tensor]],
        observedJointPairs: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        axisPriorDelta = self._computeAxisPriorDelta(
            segmentName=segmentName,
            observedSegmentPoints=observedSegmentPoints,
            observedJointPairs=observedJointPairs,
            observedAxis=observedAxis,
            baseRotation=baseRotation,
        )
        if axisPriorDelta is None:
            return baseRotation
        axisUnit = normalize(observedAxis)
        axisAngle = axisUnit * axisPriorDelta.unsqueeze(-1)
        rollRot = axis_angle_to_rotmat(axisAngle)
        return torch.matmul(rollRot, baseRotation)

    def _solveLocalRotationsFromObservedAxes(
        self,
        *,
        restState: ApproxRestState,
        rootRotation: torch.Tensor,
        observedAxes: dict[str, torch.Tensor],
        observedSegmentPoints: dict[str, dict[str, torch.Tensor]],
        observedJointPairs: dict[str, dict[str, torch.Tensor]],
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        localRotations: list[torch.Tensor] = []
        segmentGlobalRotations: dict[str, torch.Tensor] = {}
        for segmentName in SMPLX_SEGMENT_ORDER:
            parentName = SEGMENT_PARENTS[segmentName]
            parentRotation = rootRotation if parentName == "root" else segmentGlobalRotations[parentName]
            restAxis = self._getManoRestBoneAxis(restState, segmentName)
            restAxisUnit = normalize(restAxis)
            observedAxisUnit = normalize(observedAxes[segmentName])
            targetAxisLocal = torch.matmul(parentRotation.transpose(-1, -2), observedAxisUnit.unsqueeze(-1)).squeeze(-1)
            while restAxisUnit.ndim < targetAxisLocal.ndim:
                restAxisUnit = restAxisUnit.unsqueeze(0)
            restAxisUnit = torch.broadcast_to(restAxisUnit, targetAxisLocal.shape)
            localRotation = rotation_between_vectors(restAxisUnit, targetAxisLocal)
            segmentGlobal = torch.matmul(parentRotation, localRotation)
            segmentGlobal = self._applyAxisPriorRollCorrection(
                segmentName=segmentName,
                baseRotation=segmentGlobal,
                observedAxis=observedAxisUnit,
                observedSegmentPoints=observedSegmentPoints,
                observedJointPairs=observedJointPairs,
            )
            localRotation = torch.matmul(parentRotation.transpose(-1, -2), segmentGlobal)
            localRotations.append(localRotation)
            segmentGlobalRotations[segmentName] = segmentGlobal
        return localRotations, segmentGlobalRotations

    def _gatherObservedPoints(
        self,
        sampledPoints: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
        rootPoints = {
            name: sampledPoints[..., self.indexToOffset[index], :]
            for name, index in self.template.rootPointMap.items()
        }
        rootPatchPoints = sampledPoints[
            ...,
            [self.indexToOffset[int(index)] for index in self.rootPatchVertexIndices],
            :,
        ]
        segmentPoints = {
            name: {
                key: sampledPoints[..., self.indexToOffset[index], :]
                for key, index in ring.items()
            }
            for name, ring in self.template.segmentRingMap.items()
        }
        jointPairs = {
            name: {
                key: sampledPoints[..., self.indexToOffset[index], :]
                for key, index in pair.items()
            }
            for name, pair in self.template.jointPairMap.items()
        }
        return rootPoints, rootPatchPoints, segmentPoints, jointPairs

    def estimate(
        self,
        sampledPoints: torch.Tensor,
        *,
        knownShape: torch.Tensor | None = None,
    ) -> ApproxManoEstimate:
        if sampledPoints.shape[-2] != len(self.template.indexOrder) or sampledPoints.shape[-1] != 3:
            raise ValueError(f"sampledPoints must have shape [..., {len(self.template.indexOrder)}, 3], got {tuple(sampledPoints.shape)}")
        sampled = sampledPoints.to(self.device)
        shapeBetas = None if knownShape is None else knownShape.to(device=self.device, dtype=sampled.dtype)
        if shapeBetas is not None:
            while shapeBetas.ndim < sampled.ndim - 1:
                shapeBetas = shapeBetas.unsqueeze(-2)
            targetShape = (*sampled.shape[:-2], shapeBetas.shape[-1])
            shapeBetas = torch.broadcast_to(shapeBetas, targetShape)
        poseRestState = self._resolveRestState(shapeBetas=shapeBetas)
        observedRootPoints, observedRootPatchPoints, observedSegmentPoints, observedJointPairs = self._gatherObservedPoints(sampled)
        if shapeBetas is not None:
            sampled, _ = self._applyKnownShapePrescale(
                sampledPoints=sampled,
                restState=poseRestState,
                observedRootPatchPoints=observedRootPatchPoints,
                observedJointPairs=observedJointPairs,
            )
            observedRootPoints, observedRootPatchPoints, observedSegmentPoints, observedJointPairs = self._gatherObservedPoints(sampled)
        observedAxes = self._buildObservedSegmentAxes(
            observedJointPairs=observedJointPairs,
        )
        rootRotation = self._estimateRootRotation(
            restState=poseRestState,
            observedRootPoints=observedRootPoints,
        )
        _, segmentFallbackCount = buildSegmentGlobalRotations(
            restSegmentPoints=poseRestState.segmentPoints,
            observedSegmentPoints=observedSegmentPoints,
        )
        localRotations, _ = self._solveLocalRotationsFromObservedAxes(
            restState=poseRestState,
            rootRotation=rootRotation,
            observedAxes=observedAxes,
            observedSegmentPoints=observedSegmentPoints,
            observedJointPairs=observedJointPairs,
        )
        reorderedLocalRotations = [localRotations[i] for i in _SMPLX_TO_MANO_KINTREE_REORDER]
        localRotmat = torch.stack(reorderedLocalRotations, dim=-3)
        handPoseAbs = rotmat_to_axis_angle(localRotmat).reshape(*localRotmat.shape[:-3], 45)
        rootRotAA = rotmat_to_axis_angle(rootRotation)
        rootTrans = self._estimateRootTranslation(
            sampledPoints=sampled,
            rootRotAA=rootRotAA,
            handPose=handPoseAbs,
            shapeBetas=shapeBetas,
        )
        handPoseRel = handPoseAbs - self.handPoseMean.view(*([1] * (handPoseAbs.ndim - 1)), 45)
        outputBetas = (
            torch.zeros(*handPoseRel.shape[:-1], 10, dtype=handPoseRel.dtype, device=handPoseRel.device)
            if shapeBetas is None
            else shapeBetas.to(device=handPoseRel.device, dtype=handPoseRel.dtype)
        )
        fullMano = torch.cat([rootRotAA, handPoseRel, rootTrans, outputBetas], dim=-1)
        return ApproxManoEstimate(
            rootRot=rootRotAA,
            rootTrans=rootTrans,
            handPose=handPoseRel,
            fullMano=fullMano,
            fallbackCount=segmentFallbackCount,
        )
