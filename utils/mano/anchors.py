"""Anchor selection utilities for flat-hand MANO visualization."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import colorsys

import numpy as np
import torch
import trimesh
from manotorch.manolayer import ManoLayer


FINGER_JOINTS_21 = OrderedDict([
    ("thumb", (1, 2, 3, 4)),
    ("index", (5, 6, 7, 8)),
    ("middle", (9, 10, 11, 12)),
    ("ring", (13, 14, 15, 16)),
    ("pinky", (17, 18, 19, 20)),
])
SMPLX_SEGMENT_ORDER = [
    "thumb_segment_1",
    "thumb_segment_2",
    "thumb_segment_3",
    "index_segment_1",
    "index_segment_2",
    "index_segment_3",
    "middle_segment_1",
    "middle_segment_2",
    "middle_segment_3",
    "ring_segment_1",
    "ring_segment_2",
    "ring_segment_3",
    "pinky_segment_1",
    "pinky_segment_2",
    "pinky_segment_3",
]
SEGMENT_PARENTS = {
    "index_segment_1": "root",
    "index_segment_2": "index_segment_1",
    "index_segment_3": "index_segment_2",
    "middle_segment_1": "root",
    "middle_segment_2": "middle_segment_1",
    "middle_segment_3": "middle_segment_2",
    "pinky_segment_1": "root",
    "pinky_segment_2": "pinky_segment_1",
    "pinky_segment_3": "pinky_segment_2",
    "ring_segment_1": "root",
    "ring_segment_2": "ring_segment_1",
    "ring_segment_3": "ring_segment_2",
    "thumb_segment_1": "root",
    "thumb_segment_2": "thumb_segment_1",
    "thumb_segment_3": "thumb_segment_2",
}
RING_TRIANGLE_TOPK = 12
PALM_SURFACE_COUNT = 5
TOTAL_HAND_ANCHOR_COUNT = 100
PALM_PART_ASSIGNMENT_PATH = Path(__file__).resolve().parents[2] / "assets" / "merged_vertex_assignment.txt"
WRIST_CUFF_TOTAL_COUNT = 6
WRIST_CUFF_POOL_MULTIPLIER = 6
DEFAULT_RING_SLICE_PRESET = {
    "target_alpha": 0.50,
    "scan_half_width": 0.34,
    "scan_steps": 17,
    "alpha_tolerance": 0.08,
    "alpha_tolerance_fallback": 0.14,
}
MANUAL_RING_SLICE_PRESETS = {
    "middle_segment_1": {
        "target_alpha": 0.16,
        "scan_half_width": 0.02,
        "scan_steps": 3,
    },
}
PALM_ROOT_TARGET_NAMES = ("wrist_radial", "wrist_ulnar", "index_base", "middle_base", "pinky_base")


@dataclass
class AnchorGroup:
    name: str
    indices: list[int]
    color: np.ndarray


@dataclass
class FlatHandAnchorTemplate:
    verts: np.ndarray
    joints: np.ndarray
    faces: np.ndarray
    groups: list[AnchorGroup]
    segmentRingMap: dict[str, dict[str, int]]
    jointPairMap: dict[str, dict[str, int]]
    rootPointMap: dict[str, int]
    palmSurfaceMap: dict[str, list[int]]
    wristCuffMap: dict[str, list[int]]
    indexOrder: list[int]
    debugInfo: dict[str, dict[str, np.ndarray]]


def _normalize(vector: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def _project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    unitNormal = _normalize(normal)
    return vector - np.dot(vector, unitNormal) * unitNormal


def _build_palette(count: int) -> list[np.ndarray]:
    colors = []
    for idx in range(count):
        hue = idx / max(count, 1)
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors.append(np.array([int(255 * red), int(255 * green), int(255 * blue), 255], dtype=np.uint8))
    return colors


def loadFlatHandMano(*, manoPath: str, handSide: str = "right") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer = ManoLayer(
        side=handSide,
        use_pca=False,
        flat_hand_mean=True,
        ncomps=45,
        mano_assets_root=manoPath,
    )
    output = layer(
        torch.zeros((1, 48), dtype=torch.float32),
        torch.zeros((1, 10), dtype=torch.float32),
    )
    verts = output.verts[0].detach().cpu().numpy().astype(np.float32)
    joints = output.joints[0].detach().cpu().numpy().astype(np.float32)
    faces = layer.th_faces.detach().cpu().numpy().astype(np.int64)
    return verts, joints, faces


def _point_to_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = end - start
    axisNormSq = float(np.dot(axis, axis))
    if axisNormSq < 1.0e-10:
        diff = points - start[None, :]
        return np.linalg.norm(diff, axis=1), np.zeros((points.shape[0],), dtype=np.float32)
    rel = points - start[None, :]
    alpha = np.clip((rel @ axis) / axisNormSq, 0.0, 1.0)
    closest = start[None, :] + alpha[:, None] * axis[None, :]
    return np.linalg.norm(points - closest, axis=1), alpha.astype(np.float32)


def _buildSegmentRegistry(joints: np.ndarray) -> list[dict]:
    registry = []
    for fingerName, jointIds in FINGER_JOINTS_21.items():
        fingerCenters = joints[np.asarray(jointIds, dtype=np.int64)]
        for segmentIdx in range(3):
            start = fingerCenters[segmentIdx]
            end = fingerCenters[segmentIdx + 1]
            registry.append(
                {
                    "name": f"{fingerName}_segment_{segmentIdx + 1}",
                    "finger": fingerName,
                    "segment_idx": segmentIdx,
                    "start": start,
                    "end": end,
                }
            )
    return registry


def _assignVerticesToSegments(verts: np.ndarray, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    segmentRegistry = _buildSegmentRegistry(joints)
    distances = []
    for segment in segmentRegistry:
        dist, _ = _point_to_segment_distance(verts, segment["start"], segment["end"])
        distances.append(dist)
    distanceMatrix = np.stack(distances, axis=1)
    assignment = np.argmin(distanceMatrix, axis=1)
    return assignment, segmentRegistry


def _fingerAxes(*, fingerName: str, joints: np.ndarray) -> dict:
    fingerCenters = joints[np.asarray(FINGER_JOINTS_21[fingerName], dtype=np.int64)]
    wrist = joints[0]
    palmBases = np.stack([joints[jointIds[0]] for jointIds in FINGER_JOINTS_21.values()], axis=0)
    palmCenter = np.mean(np.concatenate([palmBases, wrist[None, :]], axis=0), axis=0)
    palmNormal = _normalize(np.cross(joints[5] - joints[17], joints[9] - wrist))
    radial = _normalize(fingerCenters[0] - palmCenter)
    if np.linalg.norm(radial) < 1.0e-6:
        radial = _normalize(joints[9] - palmCenter)
    return {
        "centers": fingerCenters,
        "palm_center": palmCenter,
        "palm_normal": palmNormal,
        "radial": radial,
    }


def _pick_extreme(
    *,
    verts: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
    candidateIndices: np.ndarray,
    used: set[int],
) -> int:
    if candidateIndices.size == 0:
        raise ValueError("Candidate vertex set is empty; cannot select a point")
    scores = (verts[candidateIndices] - center[None, :]) @ _normalize(direction)
    for idx in candidateIndices[np.argsort(-scores)].tolist():
        if int(idx) not in used:
            used.add(int(idx))
            return int(idx)
    bestIdx = int(candidateIndices[int(np.argmax(scores))])
    used.add(bestIdx)
    return bestIdx


def _segment_candidates(
    *,
    verts: np.ndarray,
    segmentMask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    alphaWindow: float,
) -> np.ndarray:
    axis = end - start
    axisNorm = np.linalg.norm(axis)
    axisUnit = _normalize(axis)
    rel = verts - start[None, :]
    proj = rel @ axisUnit
    closest = start[None, :] + proj[:, None] * axisUnit[None, :]
    radial = np.linalg.norm(verts - closest, axis=1)
    alpha = proj / max(axisNorm, 1.0e-8)
    return np.where(
        segmentMask
        & (alpha >= 0.5 - alphaWindow)
        & (alpha <= 0.5 + alphaWindow)
        & (radial <= radius)
    )[0]


def _joint_candidates(
    *,
    verts: np.ndarray,
    fingerMask: np.ndarray,
    segmentMask: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    dist = np.linalg.norm(verts - center[None, :], axis=1)
    return np.where(fingerMask & segmentMask & (dist <= radius))[0]


def _fallback_candidates(*, verts: np.ndarray, fingerMask: np.ndarray, center: np.ndarray, topk: int = 48) -> np.ndarray:
    fingerIndices = np.where(fingerMask)[0]
    distances = np.linalg.norm(verts[fingerIndices] - center[None, :], axis=1)
    order = np.argsort(distances)[: min(topk, fingerIndices.shape[0])]
    return fingerIndices[order]


def _segmentSelectionParams(segmentName: str) -> dict[str, float]:
    if segmentName == "middle_segment_1":
        return {"radius": 0.016, "alpha_window": 0.055}
    if segmentName.endswith("_segment_1"):
        return {"radius": 0.018, "alpha_window": 0.070}
    return {"radius": 0.016, "alpha_window": 0.080}


def _segment_frame_axes(*, start: np.ndarray, end: np.ndarray, fingerInfo: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xAxis = _normalize(end - start)
    zAxis = _normalize(np.cross(xAxis, fingerInfo["radial"]))
    if np.linalg.norm(zAxis) < 1.0e-6:
        zAxis = fingerInfo["palm_normal"]
    yAxis = _normalize(np.cross(zAxis, xAxis))
    zAxis = _normalize(np.cross(xAxis, yAxis))
    return xAxis, yAxis, zAxis


def _unique_available_indices(*, pools: list[np.ndarray], used: set[int]) -> np.ndarray:
    searchPool = []
    seen = set()
    for pool in pools:
        for idx in pool.tolist():
            index = int(idx)
            if index not in used and index not in seen:
                searchPool.append(index)
                seen.add(index)
    return np.asarray(searchPool, dtype=np.int64)


def _best_spread_ring_triangle(
    *,
    verts: np.ndarray,
    candidateIndices: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    yAxis: np.ndarray,
    zAxis: np.ndarray,
) -> tuple[tuple[float, float, float, float], np.ndarray] | None:
    if candidateIndices.size < 3:
        return None
    rel = verts[candidateIndices] - center[None, :]
    axial = np.abs(rel @ axis)
    yCoord = rel @ yAxis
    zCoord = rel @ zAxis
    radial = np.sqrt(np.maximum(yCoord ** 2 + zCoord ** 2, 0.0))
    topOrder = np.argsort(-radial)[: min(RING_TRIANGLE_TOPK, candidateIndices.shape[0])]
    topIndices = candidateIndices[topOrder]
    topY = yCoord[topOrder]
    topZ = zCoord[topOrder]
    topAxial = axial[topOrder]
    bestKey = None
    bestTriple = None
    for idxA in range(topIndices.shape[0] - 2):
        for idxB in range(idxA + 1, topIndices.shape[0] - 1):
            for idxC in range(idxB + 1, topIndices.shape[0]):
                yz = np.array(
                    [
                        [topY[idxA], topZ[idxA]],
                        [topY[idxB], topZ[idxB]],
                        [topY[idxC], topZ[idxC]],
                    ],
                    dtype=np.float32,
                )
                area = 0.5 * abs(np.cross(yz[1] - yz[0], yz[2] - yz[0]))
                pair01 = np.linalg.norm(yz[0] - yz[1])
                pair12 = np.linalg.norm(yz[1] - yz[2])
                pair02 = np.linalg.norm(yz[0] - yz[2])
                minPair = min(pair01, pair12, pair02)
                pairSum = pair01 + pair12 + pair02
                meanAxial = float(np.mean([topAxial[idxA], topAxial[idxB], topAxial[idxC]]))
                scoreKey = (float(minPair), float(pairSum), float(area), -meanAxial)
                if bestKey is None or scoreKey > bestKey:
                    bestKey = scoreKey
                    bestTriple = np.asarray(
                        [
                            int(topIndices[idxA]),
                            int(topIndices[idxB]),
                            int(topIndices[idxC]),
                        ],
                        dtype=np.int64,
                    )
    if bestTriple is None or bestKey is None:
        return None
    return bestKey, bestTriple


def _scan_ring_slice(
    *,
    verts: np.ndarray,
    segmentIndices: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    axis: np.ndarray,
    yAxis: np.ndarray,
    zAxis: np.ndarray,
    preset: dict[str, float],
    used: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    availableIndices = np.asarray([int(index) for index in segmentIndices.tolist() if int(index) not in used], dtype=np.int64)
    if availableIndices.size < 3:
        return None
    axisNorm = max(np.linalg.norm(end - start), 1.0e-8)
    alpha = ((verts[availableIndices] - start[None, :]) @ axis) / axisNorm
    bestKey = None
    bestTriple = None
    bestCandidates = None
    bestCenter = None
    bestAlpha = None
    alphaMin = max(0.0, float(preset["target_alpha"]) - float(preset["scan_half_width"]))
    alphaMax = min(1.0, float(preset["target_alpha"]) + float(preset["scan_half_width"]))
    for alphaCenter in np.linspace(alphaMin, alphaMax, int(preset["scan_steps"])):
        center = start + float(alphaCenter) * (end - start)
        sliceIndices = availableIndices[np.abs(alpha - float(alphaCenter)) <= float(preset["alpha_tolerance"])]
        if sliceIndices.size < 3:
            sliceIndices = availableIndices[np.abs(alpha - float(alphaCenter)) <= float(preset["alpha_tolerance_fallback"])]
        triangle = _best_spread_ring_triangle(
            verts=verts,
            candidateIndices=sliceIndices,
            center=center,
            axis=axis,
            yAxis=yAxis,
            zAxis=zAxis,
        )
        if triangle is None:
            continue
        triangleKey, triple = triangle
        scoreKey = (*triangleKey, -abs(float(alphaCenter) - float(preset["target_alpha"])), sliceIndices.size)
        if bestKey is None or scoreKey > bestKey:
            bestKey = scoreKey
            bestTriple = triple
            bestCandidates = sliceIndices
            bestCenter = center
            bestAlpha = float(alphaCenter)
    if bestTriple is None or bestCandidates is None or bestCenter is None or bestAlpha is None:
        return None
    return bestTriple, bestCandidates, bestCenter, bestAlpha


def _order_ring_points(
    *,
    verts: np.ndarray,
    triple: np.ndarray,
    center: np.ndarray,
    yAxis: np.ndarray,
    zAxis: np.ndarray,
) -> list[int]:
    tripleRel = verts[triple] - center[None, :]
    tripleY = tripleRel @ yAxis
    tripleZ = tripleRel @ zAxis
    posLocal = int(np.argmax(tripleZ))
    negLocal = int(np.argmin(tripleZ))
    midLocal = next(idx for idx in range(3) if idx not in (posLocal, negLocal))
    return [int(triple[midLocal]), int(triple[posLocal]), int(triple[negLocal])]


def _selectThreePointRing(
    *,
    verts: np.ndarray,
    center: np.ndarray,
    candidates: np.ndarray,
    fallbackCandidates: np.ndarray,
    xAxis: np.ndarray,
    yAxis: np.ndarray,
    zAxis: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    segmentName: str,
    segmentIndices: np.ndarray,
    used: set[int],
) -> tuple[list[int], dict[str, np.ndarray]]:
    axis = _normalize(end - start)
    preset = dict(DEFAULT_RING_SLICE_PRESET)
    preset.update(MANUAL_RING_SLICE_PRESETS.get(segmentName, {}))
    scanResult = _scan_ring_slice(
        verts=verts,
        segmentIndices=segmentIndices,
        start=start,
        end=end,
        axis=axis,
        yAxis=yAxis,
        zAxis=zAxis,
        preset=preset,
        used=used,
    )
    if scanResult is not None:
        triple, sliceCandidates, sliceCenter, sliceAlpha = scanResult
        ordered = _order_ring_points(verts=verts, triple=triple, center=sliceCenter, yAxis=yAxis, zAxis=zAxis)
        for index in ordered:
            used.add(index)
        return ordered, {
            "candidates": sliceCandidates.astype(np.int64),
            "selected": np.asarray(ordered, dtype=np.int64),
            "center": sliceCenter.astype(np.float32),
            "slice_alpha": np.asarray([sliceAlpha], dtype=np.float32),
        }
    searchArray = _unique_available_indices(pools=[candidates, fallbackCandidates], used=used)
    if searchArray.size < 3:
        raise ValueError("Fewer than 3 candidate points are available for the ring group")
    triangle = _best_spread_ring_triangle(
        verts=verts,
        candidateIndices=searchArray,
        center=center,
        axis=axis,
        yAxis=yAxis,
        zAxis=zAxis,
    )
    if triangle is None:
        raise ValueError("Could not build a valid triangle from the ring candidates")
    _, triple = triangle
    ordered = _order_ring_points(verts=verts, triple=triple, center=center, yAxis=yAxis, zAxis=zAxis)
    for index in ordered:
        used.add(index)
    return ordered, {
        "candidates": candidates.astype(np.int64),
        "selected": np.asarray(ordered, dtype=np.int64),
        "center": center.astype(np.float32),
    }


def _center_seed_farthest_point_sample_2d(*, coords: np.ndarray, indices: np.ndarray, count: int, used: set[int]) -> list[int]:
    available = [int(index) for index in indices.tolist() if int(index) not in used]
    if not available:
        return []
    availableArray = np.asarray(available, dtype=np.int64)
    availableCoords = coords[availableArray]
    center = np.mean(availableCoords, axis=0, keepdims=True)
    firstIdx = int(availableArray[np.argmin(np.linalg.norm(availableCoords - center, axis=1))])
    selected = [firstIdx]
    used.add(firstIdx)
    while len(selected) < min(count, len(available)):
        selectedCoords = coords[np.asarray(selected, dtype=np.int64)]
        candidates = np.asarray([idx for idx in available if idx not in selected], dtype=np.int64)
        if candidates.size == 0:
            break
        candCoords = coords[candidates]
        dist = np.linalg.norm(candCoords[:, None, :] - selectedCoords[None, :, :], axis=-1)
        nextIdx = int(candidates[np.argmax(np.min(dist, axis=1))])
        selected.append(nextIdx)
        used.add(nextIdx)
    return selected


def _select_wrist_cuff_points(
    *,
    verts: np.ndarray,
    wristPoint: np.ndarray,
    indices: np.ndarray,
    count: int,
    used: set[int],
) -> list[int]:
    available = np.asarray([int(index) for index in indices.tolist() if int(index) not in used], dtype=np.int64)
    if available.size == 0 or count <= 0:
        return []
    poolSize = min(available.size, max(count, count * WRIST_CUFF_POOL_MULTIPLIER))
    wristDist = np.linalg.norm(verts[available] - wristPoint[None, :], axis=1)
    cuffPool = available[np.argsort(wristDist)[:poolSize]]
    selected = [int(index) for index in cuffPool[: min(count, cuffPool.shape[0])].tolist()]
    for index in selected:
        used.add(index)
    return selected


def _buildFingerMasks(*, segmentAssign: np.ndarray, segmentRegistry: list[dict]) -> dict[str, np.ndarray]:
    fingerMasks = {}
    for fingerName in FINGER_JOINTS_21.keys():
        validSegments = [idx for idx, segment in enumerate(segmentRegistry) if segment["finger"] == fingerName]
        fingerMasks[fingerName] = np.isin(segmentAssign, np.asarray(validSegments, dtype=np.int64))
    return fingerMasks


def _load_palm_part_assignment(*, vertexCount: int) -> np.ndarray:
    if not PALM_PART_ASSIGNMENT_PATH.is_file():
        raise FileNotFoundError(f"Merged vertex assignment file does not exist: {PALM_PART_ASSIGNMENT_PATH}")
    assignment = np.asarray(
        [int(text.strip()) for text in PALM_PART_ASSIGNMENT_PATH.read_text().splitlines() if text.strip()],
        dtype=np.int64,
    )
    if assignment.shape[0] != vertexCount:
        raise ValueError(f"Merged vertex assignment length must be {vertexCount}, got {assignment.shape[0]}")
    return assignment


def _resolve_palm_surface_parts(
    *,
    partAssignment: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    palmNormal: np.ndarray,
) -> tuple[int, int]:
    counts = np.bincount(partAssignment, minlength=int(np.max(partAssignment)) + 1)
    topParts = np.argsort(-counts)[:2]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertexNormals = np.asarray(mesh.vertex_normals)
    firstScore = float(np.mean(vertexNormals[partAssignment == int(topParts[0])] @ palmNormal))
    secondScore = float(np.mean(vertexNormals[partAssignment == int(topParts[1])] @ palmNormal))
    if firstScore >= secondScore:
        return int(topParts[0]), int(topParts[1])
    return int(topParts[1]), int(topParts[0])


def _assign_root_points_from_selected(
    *,
    selectedIndices: np.ndarray,
    verts: np.ndarray,
    targetMap: dict[str, np.ndarray],
) -> dict[str, int]:
    rootPointMap: dict[str, int] = {}
    assigned: set[int] = set()
    for targetName in PALM_ROOT_TARGET_NAMES:
        center = targetMap[targetName]
        distances = np.linalg.norm(verts[selectedIndices] - center[None, :], axis=1)
        for localIdx in np.argsort(distances).tolist():
            index = int(selectedIndices[localIdx])
            if index not in assigned:
                rootPointMap[targetName] = index
                assigned.add(index)
                break
        if targetName not in rootPointMap:
            raise ValueError(f"Could not assign a unique sample point to root semantic target: {targetName}")
    return rootPointMap


def buildFlatHandAnchorTemplate(*, verts: np.ndarray, joints: np.ndarray, faces: np.ndarray) -> FlatHandAnchorTemplate:
    segmentAssign, segmentRegistry = _assignVerticesToSegments(verts, joints)
    fingerMasks = _buildFingerMasks(segmentAssign=segmentAssign, segmentRegistry=segmentRegistry)
    palette = _build_palette(len(FINGER_JOINTS_21) * 7 + 15)
    groups: list[AnchorGroup] = []
    used: set[int] = set()
    colorIdx = 0
    segmentRingMap: dict[str, dict[str, int]] = {}
    jointPairMap: dict[str, dict[str, int]] = {}
    debugInfo: dict[str, dict[str, np.ndarray]] = {}

    for fingerName in FINGER_JOINTS_21.keys():
        fingerInfo = _fingerAxes(fingerName=fingerName, joints=joints)
        fingerCenters = fingerInfo["centers"]
        fingerMask = fingerMasks[fingerName]

        for segmentIdx in range(3):
            start = fingerCenters[segmentIdx]
            end = fingerCenters[segmentIdx + 1]
            center = 0.5 * (start + end)
            segmentName = f"{fingerName}_segment_{segmentIdx + 1}"
            segmentId = next(idx for idx, segment in enumerate(segmentRegistry) if segment["name"] == segmentName)
            segmentMask = segmentAssign == segmentId
            segmentIndices = np.where(segmentMask)[0]
            xAxis, yAxis, zAxis = _segment_frame_axes(start=start, end=end, fingerInfo=fingerInfo)
            params = _segmentSelectionParams(segmentName)
            candidates = _segment_candidates(
                verts=verts,
                segmentMask=segmentMask,
                start=start,
                end=end,
                radius=float(params["radius"]),
                alphaWindow=float(params["alpha_window"]),
            )
            fallbackCandidates = _fallback_candidates(verts=verts, fingerMask=fingerMask, center=center, topk=64)
            if candidates.size == 0:
                candidates = fallbackCandidates
            groupIndices, ringDebug = _selectThreePointRing(
                verts=verts,
                center=center,
                candidates=candidates,
                fallbackCandidates=fallbackCandidates,
                xAxis=xAxis,
                yAxis=yAxis,
                zAxis=zAxis,
                start=start,
                end=end,
                segmentName=segmentName,
                segmentIndices=segmentIndices,
                used=used,
            )
            groups.append(AnchorGroup(name=f"{segmentName}_ring", indices=groupIndices, color=palette[colorIdx]))
            segmentRingMap[segmentName] = {
                "mid": int(groupIndices[0]),
                "pos": int(groupIndices[1]),
                "neg": int(groupIndices[2]),
            }
            debugInfo[segmentName] = ringDebug
            colorIdx += 1

        for pointIdx, pointName in enumerate(("joint_1", "joint_2", "joint_3", "tip")):
            center = fingerCenters[pointIdx]
            if pointName == "joint_1":
                segmentMask = np.isin(
                    segmentAssign,
                    np.asarray(
                        [
                            next(idx for idx, segment in enumerate(segmentRegistry) if segment["name"] == f"{fingerName}_segment_1"),
                            next(idx for idx, segment in enumerate(segmentRegistry) if segment["name"] == f"{fingerName}_segment_2"),
                        ],
                        dtype=np.int64,
                    ),
                )
            elif pointName in ("joint_2", "joint_3"):
                localIdx = 2 if pointName == "joint_3" else 1
                segmentMask = np.isin(
                    segmentAssign,
                    np.asarray(
                        [
                            next(idx for idx, segment in enumerate(segmentRegistry) if segment["name"] == f"{fingerName}_segment_{localIdx}"),
                            next(idx for idx, segment in enumerate(segmentRegistry) if segment["name"] == f"{fingerName}_segment_{localIdx + 1}"),
                        ],
                        dtype=np.int64,
                    ),
                )
            else:
                tipSegment = next(idx for idx, segment in enumerate(segmentRegistry) if segment["name"] == f"{fingerName}_segment_3")
                segmentMask = segmentAssign == tipSegment
            candidates = _joint_candidates(
                verts=verts,
                fingerMask=fingerMask,
                segmentMask=segmentMask,
                center=center,
                radius=0.018 if pointName != "tip" else 0.014,
            )
            if candidates.size == 0:
                candidates = _fallback_candidates(verts=verts, fingerMask=fingerMask, center=center, topk=32)
            zAxis = fingerInfo["palm_normal"]
            groupIndices = [
                _pick_extreme(verts=verts, center=center, direction=zAxis, candidateIndices=candidates, used=used),
                _pick_extreme(verts=verts, center=center, direction=-zAxis, candidateIndices=candidates, used=used),
            ]
            groups.append(AnchorGroup(name=f"{fingerName}_{pointName}_updown", indices=groupIndices, color=palette[colorIdx]))
            jointPairMap[f"{fingerName}_{pointName}"] = {
                "pos": int(groupIndices[0]),
                "neg": int(groupIndices[1]),
            }
            colorIdx += 1

    wrist = joints[0]
    palmBases = {fingerName: joints[jointIds[0]] for fingerName, jointIds in FINGER_JOINTS_21.items()}
    palmCenter = np.mean(np.stack([wrist, *palmBases.values()], axis=0), axis=0)
    palmNormal = _normalize(np.cross(joints[5] - joints[17], joints[9] - wrist))
    palmSide = _normalize(joints[5] - joints[17])
    wristCenter = 0.5 * ((wrist + 0.020 * palmSide) + (wrist - 0.020 * palmSide))
    xPalm = _normalize(palmBases["middle"] - wristCenter)
    yPalm = _normalize(_project_to_plane(palmBases["index"] - palmBases["pinky"], xPalm))
    zPalm = _normalize(np.cross(xPalm, yPalm))
    yPalm = _normalize(np.cross(zPalm, xPalm))
    relPalm = verts - palmCenter[None, :]
    palm2d = np.stack([relPalm @ yPalm, relPalm @ xPalm], axis=1)
    partAssignment = _load_palm_part_assignment(vertexCount=verts.shape[0])
    dorsalPart, palmarPart = _resolve_palm_surface_parts(
        partAssignment=partAssignment,
        verts=verts,
        faces=faces,
        palmNormal=palmNormal,
    )
    fingerPointCount = sum(len(group.indices) for group in groups)
    palmPointBudget = TOTAL_HAND_ANCHOR_COUNT - fingerPointCount
    dorsalPartCount = int(np.sum(partAssignment == dorsalPart))
    palmarPartCount = int(np.sum(partAssignment == palmarPart))
    if palmPointBudget < WRIST_CUFF_TOTAL_COUNT:
        raise ValueError(f"Palm-point budget is too small to include the wrist cuff: budget={palmPointBudget}")
    dorsalIndices = np.where(partAssignment == dorsalPart)[0]
    palmarIndices = np.where(partAssignment == palmarPart)[0]
    palmUnionIndices = np.concatenate([dorsalIndices, palmarIndices], axis=0)
    cuffSelected = _select_wrist_cuff_points(
        verts=verts,
        wristPoint=wrist,
        indices=palmUnionIndices,
        count=WRIST_CUFF_TOTAL_COUNT,
        used=used,
    )
    dorsalCuffSelected = [int(index) for index in cuffSelected if int(partAssignment[int(index)]) == dorsalPart]
    palmarCuffSelected = [int(index) for index in cuffSelected if int(partAssignment[int(index)]) == palmarPart]
    surfacePointBudget = palmPointBudget - len(dorsalCuffSelected) - len(palmarCuffSelected)
    dorsalSurfaceCount = surfacePointBudget // 2
    palmarSurfaceCount = surfacePointBudget - dorsalSurfaceCount
    if dorsalPartCount >= palmarPartCount:
        dorsalSurfaceCount, palmarSurfaceCount = palmarSurfaceCount, dorsalSurfaceCount
    dorsalSelected = _center_seed_farthest_point_sample_2d(
        coords=palm2d,
        indices=dorsalIndices,
        count=dorsalSurfaceCount,
        used=used,
    )
    palmarSelected = _center_seed_farthest_point_sample_2d(
        coords=palm2d,
        indices=palmarIndices,
        count=palmarSurfaceCount,
        used=used,
    )
    groups.append(AnchorGroup(name="wrist_cuff_dorsal", indices=dorsalCuffSelected, color=palette[colorIdx]))
    colorIdx += 1
    groups.append(AnchorGroup(name="wrist_cuff_palmar", indices=palmarCuffSelected, color=palette[colorIdx]))
    colorIdx += 1
    groups.append(AnchorGroup(name="palm_dorsal_surface", indices=dorsalSelected, color=palette[colorIdx]))
    colorIdx += 1
    groups.append(AnchorGroup(name="palm_palmar_surface", indices=palmarSelected, color=palette[colorIdx]))
    colorIdx += 1
    palmSurfaceMap = {
        "dorsal": [int(index) for index in dorsalSelected],
        "palmar": [int(index) for index in palmarSelected],
    }
    wristCuffMap = {
        "dorsal": [int(index) for index in dorsalCuffSelected],
        "palmar": [int(index) for index in palmarCuffSelected],
    }

    rootTargetMap = {
        "wrist_radial": wrist + 0.020 * palmSide,
        "wrist_ulnar": wrist - 0.020 * palmSide,
        "index_base": palmBases["index"],
        "middle_base": palmBases["middle"],
        "pinky_base": palmBases["pinky"],
    }
    cuffPool = np.asarray(dorsalCuffSelected + palmarCuffSelected, dtype=np.int64)
    palmPool = np.asarray(dorsalCuffSelected + palmarCuffSelected + dorsalSelected + palmarSelected, dtype=np.int64)
    rootPointMap: dict[str, int] = {}
    assignedRoot: set[int] = set()
    for targetName in ("wrist_radial", "wrist_ulnar"):
        distances = np.linalg.norm(verts[cuffPool] - rootTargetMap[targetName][None, :], axis=1)
        for localIdx in np.argsort(distances).tolist():
            index = int(cuffPool[localIdx])
            if index not in assignedRoot:
                rootPointMap[targetName] = index
                assignedRoot.add(index)
                break
    for targetName in ("index_base", "middle_base", "pinky_base"):
        distances = np.linalg.norm(verts[palmPool] - rootTargetMap[targetName][None, :], axis=1)
        for localIdx in np.argsort(distances).tolist():
            index = int(palmPool[localIdx])
            if index not in assignedRoot:
                rootPointMap[targetName] = index
                assignedRoot.add(index)
                break
    if set(rootTargetMap.keys()) != set(rootPointMap.keys()):
        raise ValueError(f"Could not build a complete rootPointMap: expected={sorted(rootTargetMap.keys())}, actual={sorted(rootPointMap.keys())}")
    indexOrder: list[int] = []
    seen = set()
    for group in groups:
        for index in group.indices:
            if int(index) not in seen:
                indexOrder.append(int(index))
                seen.add(int(index))
    return FlatHandAnchorTemplate(
        verts=verts,
        joints=joints,
        faces=faces,
        groups=groups,
        segmentRingMap=segmentRingMap,
        jointPairMap=jointPairMap,
        rootPointMap=rootPointMap,
        palmSurfaceMap=palmSurfaceMap,
        wristCuffMap=wristCuffMap,
        indexOrder=indexOrder,
        debugInfo=debugInfo,
    )


def selectFingerAndPalmGroups(*, verts: np.ndarray, joints: np.ndarray, faces: np.ndarray | None = None) -> list[AnchorGroup]:
    if faces is None:
        faces = np.zeros((0, 3), dtype=np.int64)
    return buildFlatHandAnchorTemplate(verts=verts, joints=joints, faces=faces).groups


def buildAnchorScene(*, verts: np.ndarray, faces: np.ndarray, groups: List[AnchorGroup], pointRadius: float = 0.0038) -> trimesh.Scene:
    scene = trimesh.Scene()
    handMesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    handMesh.visual.vertex_colors = np.tile(np.array([[185, 185, 185, 210]], dtype=np.uint8), (verts.shape[0], 1))
    scene.add_geometry(handMesh, geom_name="flat_hand_mesh")
    for group in groups:
        for localIdx, index in enumerate(group.indices):
            sphere = trimesh.creation.uv_sphere(radius=pointRadius)
            sphere.apply_translation(verts[int(index)])
            sphere.visual.vertex_colors = np.tile(group.color[None, :], (sphere.vertices.shape[0], 1))
            scene.add_geometry(sphere, geom_name=f"{group.name}_{localIdx:02d}_v{int(index):03d}")
    return scene
