"""Runtime reorder helpers for ApproxForwardManoEstimator input semantics."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _buildGroupOffsets(template) -> Dict[str, List[int]]:
    indexOffsetMap: Dict[int, int] = {}
    offsetsByGroup: Dict[str, List[int]] = {}
    for group in template.groups:
        groupOffsets: List[int] = []
        for index in group.indices:
            indexInt = int(index)
            if indexInt not in indexOffsetMap:
                indexOffsetMap[indexInt] = len(indexOffsetMap)
            groupOffsets.append(indexOffsetMap[indexInt])
        offsetsByGroup[group.name] = groupOffsets
    return offsetsByGroup


def _mapSourceGroupName(*, targetGroupName: str, sourceHandSide: str, targetHandSide: str) -> str:
    if sourceHandSide == targetHandSide:
        return targetGroupName
    if targetGroupName == "wrist_cuff_dorsal":
        return "wrist_cuff_palmar"
    if targetGroupName == "wrist_cuff_palmar":
        return "wrist_cuff_dorsal"
    if targetGroupName == "palm_dorsal_surface":
        return "palm_palmar_surface"
    if targetGroupName == "palm_palmar_surface":
        return "palm_dorsal_surface"
    return targetGroupName


def _mapSourceSlotOrder(
    *,
    targetGroupName: str,
    slotCount: int,
    sourceHandSide: str,
    targetHandSide: str,
) -> List[int]:
    if sourceHandSide == targetHandSide:
        return list(range(slotCount))
    if targetGroupName.endswith("_ring"):
        if slotCount != 3:
            raise ValueError(f"{targetHandSide} hand ring group must contain 3 points, got {slotCount}")
        return [0, 2, 1]
    if targetGroupName.endswith("_updown"):
        if slotCount != 2:
            raise ValueError(f"{targetHandSide} hand updown group must contain 2 points, got {slotCount}")
        return [1, 0]
    return list(range(slotCount))


def buildApproxIkInputOrder(
    *,
    sourceTemplate,
    targetTemplate,
    sourceHandSide: str,
    targetHandSide: str,
) -> torch.Tensor:
    sourceOffsetsByGroup = _buildGroupOffsets(sourceTemplate)
    targetOffsetsByGroup = _buildGroupOffsets(targetTemplate)
    reorderIndex: List[Optional[int]] = [None] * len(targetTemplate.indexOrder)
    for targetGroupName, targetOffsets in targetOffsetsByGroup.items():
        sourceGroupName = _mapSourceGroupName(
            targetGroupName=targetGroupName,
            sourceHandSide=sourceHandSide,
            targetHandSide=targetHandSide,
        )
        if sourceGroupName not in sourceOffsetsByGroup:
            raise ValueError(f"{sourceHandSide}->{targetHandSide} IK reorder is missing group: {sourceGroupName}")
        sourceOffsets = sourceOffsetsByGroup[sourceGroupName]
        sourceSlotOrder = _mapSourceSlotOrder(
            targetGroupName=targetGroupName,
            slotCount=len(targetOffsets),
            sourceHandSide=sourceHandSide,
            targetHandSide=targetHandSide,
        )
        if len(sourceOffsets) != len(targetOffsets):
            raise ValueError(
                f"{sourceHandSide}->{targetHandSide} IK group length mismatch: "
                f"{sourceGroupName}={len(sourceOffsets)} vs {targetGroupName}={len(targetOffsets)}"
            )
        for targetOffset, sourceSlot in zip(targetOffsets, sourceSlotOrder):
            reorderIndex[targetOffset] = sourceOffsets[sourceSlot]
    if any(offset is None for offset in reorderIndex):
        raise ValueError(f"{sourceHandSide}->{targetHandSide} IK reorder did not cover all {len(reorderIndex)} sample points")
    return torch.tensor([int(offset) for offset in reorderIndex], dtype=torch.long)


def resolveApproxIkInputOrders(
    *,
    sampleIndices: np.ndarray,
    leftEstimator,
    rightEstimator,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    leftIndexOrder = np.asarray(leftEstimator.template.indexOrder, dtype=np.int64)
    rightIndexOrder = np.asarray(rightEstimator.template.indexOrder, dtype=np.int64)
    sourceTemplate = None
    sourceHandSide = None
    if np.array_equal(sampleIndices, rightIndexOrder):
        sourceTemplate = rightEstimator.template
        sourceHandSide = "right"
    elif np.array_equal(sampleIndices, leftIndexOrder):
        sourceTemplate = leftEstimator.template
        sourceHandSide = "left"
    else:
        raise ValueError("The current sample index order does not match either the left or right Approx IK template")
    leftInputOrder = buildApproxIkInputOrder(
        sourceTemplate=sourceTemplate,
        targetTemplate=leftEstimator.template,
        sourceHandSide=sourceHandSide,
        targetHandSide="left",
    )
    rightInputOrder = buildApproxIkInputOrder(
        sourceTemplate=sourceTemplate,
        targetTemplate=rightEstimator.template,
        sourceHandSide=sourceHandSide,
        targetHandSide="right",
    )
    return leftInputOrder, rightInputOrder, sourceHandSide
