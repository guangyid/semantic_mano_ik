"""Build root and segment frames from fixed MANO anchors."""
from __future__ import annotations

from typing import Dict

import torch

from .geometry import gram_schmidt_frame, kabsch, normalize, project_to_plane, rotmat_to_axis_angle


RING_KEYS = ("mid", "pos", "neg")
PAIR_KEYS = ("pos", "neg")
# Three-point semantic rings are better constrained by the frame rule than by a loose Kabsch fallback.
KABSCH_ACCEPT_ANGLE_RAD = 0.08
ROOT_KABSCH_ACCEPT_ANGLE_RAD = 0.02
SEGMENT_KABSCH_ACCEPT_ANGLE_RAD = 0.02


def buildRootFrame(anchorPoints: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Build root frame from semantic root anchors.

    anchorPoints[name]: [..., 3]
    return: [..., 3, 3]
    """
    wristCenter = anchorPoints["wrist_center"]
    middleMcp = anchorPoints["middle_mcp_center"]
    indexMcp = anchorPoints["index_mcp_center"]
    pinkyMcp = anchorPoints["pinky_mcp_center"]
    xRoot = normalize(middleMcp - wristCenter)
    yHint = indexMcp - pinkyMcp
    return gram_schmidt_frame(xRoot, yHint)


def _buildRingFrame(ringPoints: torch.Tensor) -> torch.Tensor:
    """
    Build frame from ring anchors.

        ringPoints: [..., 3, 3] in (mid, pos, neg)
    return: [..., 3, 3]
    """
    midPoint, posPoint, negPoint = ringPoints.unbind(dim=-2)
    yHint = posPoint - negPoint
    zHint = midPoint - 0.5 * (posPoint + negPoint)
    xHint = torch.cross(yHint, zHint, dim=-1)
    return gram_schmidt_frame(xHint, yHint)


def _expandRestPoints(restPoints: torch.Tensor, observedPoints: torch.Tensor) -> torch.Tensor:
    expanded = restPoints
    while expanded.ndim < observedPoints.ndim:
        expanded = expanded.unsqueeze(0)
    return expanded.expand(*observedPoints.shape[:-2], observedPoints.shape[-2], observedPoints.shape[-1])


def _selectRigidRotation(
    *,
    restPoints: torch.Tensor,
    observedPoints: torch.Tensor,
    frameRotation: torch.Tensor,
    acceptAngleRad: float,
) -> tuple[torch.Tensor, int]:
    alignedRest = _expandRestPoints(restPoints, observedPoints)
    try:
        rigidRotation = kabsch(alignedRest, observedPoints)
        deltaRotation = torch.matmul(frameRotation.transpose(-1, -2), rigidRotation)
        deltaAngle = rotmat_to_axis_angle(deltaRotation).norm(dim=-1)
        useRigid = deltaAngle < acceptAngleRad
        chosenRotation = torch.where(useRigid[..., None, None], rigidRotation, frameRotation)
        fallbackCount = int((~useRigid).sum().item())
        return chosenRotation, fallbackCount
    except torch.linalg.LinAlgError:
        batchCount = int(frameRotation.reshape(-1, 3, 3).shape[0])
        return frameRotation, batchCount


def _pairCenter(pairPoints: Dict[str, torch.Tensor]) -> torch.Tensor:
    return 0.5 * (pairPoints["pos"] + pairPoints["neg"])


def _buildSegmentFrameFromAnchors(
    *,
    proximalCenter: torch.Tensor,
    distalCenter: torch.Tensor,
    ringPoints: Dict[str, torch.Tensor],
    proximalPair: Dict[str, torch.Tensor],
    distalPair: Dict[str, torch.Tensor],
) -> torch.Tensor:
    xAxis = distalCenter - proximalCenter
    ringHint = project_to_plane(ringPoints["pos"] - ringPoints["neg"], xAxis)
    proximalHint = project_to_plane(proximalPair["pos"] - proximalPair["neg"], xAxis)
    distalHint = project_to_plane(distalPair["pos"] - distalPair["neg"], xAxis)
    yHint = ringHint + 0.5 * (proximalHint + distalHint)
    yHintNorm = yHint.norm(dim=-1, keepdim=True)
    zHint = ringPoints["mid"] - 0.5 * (ringPoints["pos"] + ringPoints["neg"])
    fallbackYHint = torch.cross(zHint, xAxis, dim=-1)
    safeYHint = torch.where(yHintNorm > 1.0e-6, yHint, fallbackYHint)
    return gram_schmidt_frame(xAxis, safeYHint)


def buildRootTransformFromAnchors(
    *,
    restRootPoints: Dict[str, torch.Tensor],
    observedRootPoints: Dict[str, torch.Tensor],
    restJointPairs: Dict[str, Dict[str, torch.Tensor]],
    observedJointPairs: Dict[str, Dict[str, torch.Tensor]],
    restPalmPoints: Dict[str, torch.Tensor],
    observedPalmPoints: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    restFramePoints = {
        "wrist_center": 0.5 * (restRootPoints["wrist_radial"] + restRootPoints["wrist_ulnar"]),
        "index_mcp_center": _pairCenter(restJointPairs["index_joint_1"]),
        "middle_mcp_center": _pairCenter(restJointPairs["middle_joint_1"]),
        "pinky_mcp_center": _pairCenter(restJointPairs["pinky_joint_1"]),
    }
    observedFramePoints = {
        "wrist_center": 0.5 * (observedRootPoints["wrist_radial"] + observedRootPoints["wrist_ulnar"]),
        "index_mcp_center": _pairCenter(observedJointPairs["index_joint_1"]),
        "middle_mcp_center": _pairCenter(observedJointPairs["middle_joint_1"]),
        "pinky_mcp_center": _pairCenter(observedJointPairs["pinky_joint_1"]),
    }
    restRootFrame = buildRootFrame(restFramePoints)
    observedRootFrame = buildRootFrame(observedFramePoints)
    frameRotation = torch.matmul(observedRootFrame, restRootFrame.transpose(-1, -2))
    restFitPoints = torch.cat(
        [
            restPalmPoints["dorsal"],
            restPalmPoints["palmar"],
            _pairCenter(restJointPairs["thumb_joint_1"]).unsqueeze(-2),
            _pairCenter(restJointPairs["index_joint_1"]).unsqueeze(-2),
            _pairCenter(restJointPairs["middle_joint_1"]).unsqueeze(-2),
            _pairCenter(restJointPairs["ring_joint_1"]).unsqueeze(-2),
            _pairCenter(restJointPairs["pinky_joint_1"]).unsqueeze(-2),
            restFramePoints["wrist_center"].unsqueeze(-2),
        ],
        dim=-2,
    )
    observedFitPoints = torch.cat(
        [
            observedPalmPoints["dorsal"],
            observedPalmPoints["palmar"],
            _pairCenter(observedJointPairs["thumb_joint_1"]).unsqueeze(-2),
            _pairCenter(observedJointPairs["index_joint_1"]).unsqueeze(-2),
            _pairCenter(observedJointPairs["middle_joint_1"]).unsqueeze(-2),
            _pairCenter(observedJointPairs["ring_joint_1"]).unsqueeze(-2),
            _pairCenter(observedJointPairs["pinky_joint_1"]).unsqueeze(-2),
            observedFramePoints["wrist_center"].unsqueeze(-2),
        ],
        dim=-2,
    )
    rootRotation, fallbackCount = _selectRigidRotation(
        restPoints=restFitPoints,
        observedPoints=observedFitPoints,
        frameRotation=frameRotation,
        acceptAngleRad=ROOT_KABSCH_ACCEPT_ANGLE_RAD,
    )
    restCenter = _expandRestPoints(restFitPoints, observedFitPoints).mean(dim=-2)
    observedCenter = observedFitPoints.mean(dim=-2)
    rootTranslation = observedCenter - torch.matmul(rootRotation, restCenter.unsqueeze(-1)).squeeze(-1)
    return rootRotation, rootTranslation, fallbackCount


def buildSegmentGlobalRotationsFromAnchors(
    *,
    restSegmentPoints: Dict[str, Dict[str, torch.Tensor]],
    observedSegmentPoints: Dict[str, Dict[str, torch.Tensor]],
    restJointPairs: Dict[str, Dict[str, torch.Tensor]],
    observedJointPairs: Dict[str, Dict[str, torch.Tensor]],
    segmentEndpointMap: Dict[str, tuple[str, str]],
) -> tuple[Dict[str, torch.Tensor], int]:
    segmentRotations: Dict[str, torch.Tensor] = {}
    fallbackCount = 0
    for segmentName, restRingMap in restSegmentPoints.items():
        proximalName, distalName = segmentEndpointMap[segmentName]
        observedRingMap = observedSegmentPoints[segmentName]
        restProximalPair = restJointPairs[proximalName]
        restDistalPair = restJointPairs[distalName]
        observedProximalPair = observedJointPairs[proximalName]
        observedDistalPair = observedJointPairs[distalName]
        restFrame = _buildSegmentFrameFromAnchors(
            proximalCenter=_pairCenter(restProximalPair),
            distalCenter=_pairCenter(restDistalPair),
            ringPoints=restRingMap,
            proximalPair=restProximalPair,
            distalPair=restDistalPair,
        )
        observedFrame = _buildSegmentFrameFromAnchors(
            proximalCenter=_pairCenter(observedProximalPair),
            distalCenter=_pairCenter(observedDistalPair),
            ringPoints=observedRingMap,
            proximalPair=observedProximalPair,
            distalPair=observedDistalPair,
        )
        frameRotation = torch.matmul(observedFrame, restFrame.transpose(-1, -2))
        restFitPoints = torch.stack(
            [
                _pairCenter(restProximalPair),
                _pairCenter(restDistalPair),
                restRingMap["mid"],
                restRingMap["pos"],
                restRingMap["neg"],
            ],
            dim=-2,
        )
        observedFitPoints = torch.stack(
            [
                _pairCenter(observedProximalPair),
                _pairCenter(observedDistalPair),
                observedRingMap["mid"],
                observedRingMap["pos"],
                observedRingMap["neg"],
            ],
            dim=-2,
        )
        segmentRotation, segmentFallback = _selectRigidRotation(
            restPoints=restFitPoints,
            observedPoints=observedFitPoints,
            frameRotation=frameRotation,
            acceptAngleRad=SEGMENT_KABSCH_ACCEPT_ANGLE_RAD,
        )
        segmentRotations[segmentName] = segmentRotation
        fallbackCount += segmentFallback
    return segmentRotations, fallbackCount


def buildSegmentGlobalRotations(
    *,
    restSegmentPoints: Dict[str, Dict[str, torch.Tensor]],
    observedSegmentPoints: Dict[str, Dict[str, torch.Tensor]],
) -> tuple[Dict[str, torch.Tensor], int]:
    """
    Estimate segment global rotations.

    return:
      segmentRotations[name]: [..., 3, 3]
      fallbackCount: number of segments that used frame fallback instead of Kabsch
    """
    segmentRotations: Dict[str, torch.Tensor] = {}
    fallbackCount = 0
    for segmentName in restSegmentPoints.keys():
        restRing = torch.stack([restSegmentPoints[segmentName][key] for key in RING_KEYS], dim=-2)
        observedRing = torch.stack([observedSegmentPoints[segmentName][key] for key in RING_KEYS], dim=-2)
        restFrame = _buildRingFrame(restRing)
        observedFrame = _buildRingFrame(observedRing)
        frameRotation = torch.matmul(observedFrame, restFrame.transpose(-1, -2))
        if restRing.ndim < observedRing.ndim:
            for _ in range(observedRing.ndim - restRing.ndim):
                restRing = restRing.unsqueeze(0)
            restRing = restRing.expand_as(observedRing)
        try:
            kabschRotation = kabsch(restRing, observedRing)
            deltaRotation = torch.matmul(frameRotation.transpose(-1, -2), kabschRotation)
            deltaAngle = rotmat_to_axis_angle(deltaRotation).norm(dim=-1)
            useKabsch = bool(torch.all(deltaAngle < KABSCH_ACCEPT_ANGLE_RAD))
            segmentRotations[segmentName] = kabschRotation if useKabsch else frameRotation
            fallbackCount += 0 if useKabsch else 1
        except torch.linalg.LinAlgError:
            segmentRotations[segmentName] = frameRotation
            fallbackCount += 1
    return segmentRotations, fallbackCount
