"""Geometry helpers for forward MANO IK."""
from __future__ import annotations

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


def normalize(vector: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize [..., 3] vectors."""
    return vector / vector.norm(dim=-1, keepdim=True).clamp(min=eps)


def project_to_plane(vector: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Project [..., 3] vector onto plane orthogonal to normal."""
    unitNormal = normalize(normal)
    return vector - torch.sum(vector * unitNormal, dim=-1, keepdim=True) * unitNormal


def gram_schmidt_frame(xHint: torch.Tensor, yHint: torch.Tensor) -> torch.Tensor:
    """
    Build right-handed frame with columns [x, y, z].

    xHint/yHint: [..., 3]
    return: [..., 3, 3]
    """
    xAxis = normalize(xHint)
    yProj = project_to_plane(yHint, xAxis)
    yAxis = normalize(yProj)
    zAxis = normalize(torch.cross(xAxis, yAxis, dim=-1))
    yAxis = normalize(torch.cross(zAxis, xAxis, dim=-1))
    return torch.stack([xAxis, yAxis, zAxis], dim=-1)


def rotmat_to_axis_angle(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert [..., 3, 3] rotation matrices to axis-angle [..., 3]."""
    return matrix_to_axis_angle(rotmat)


def axis_angle_to_rotmat(axisAngle: torch.Tensor) -> torch.Tensor:
    """Convert [..., 3] axis-angle vectors to rotation matrices [..., 3, 3]."""
    return axis_angle_to_matrix(axisAngle)


def rotation_between_vectors(source: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Build the shortest rotation matrix that maps source to target."""
    srcUnit = normalize(source, eps=eps)
    tgtUnit = normalize(target, eps=eps)
    crossVal = torch.cross(srcUnit, tgtUnit, dim=-1)
    dotVal = torch.sum(srcUnit * tgtUnit, dim=-1).clamp(-1.0, 1.0)
    sinVal = crossVal.norm(dim=-1)
    eye = torch.eye(3, device=source.device, dtype=source.dtype).expand(*source.shape[:-1], 3, 3).clone()
    skew = torch.zeros_like(eye)
    skew[..., 0, 1] = -crossVal[..., 2]
    skew[..., 0, 2] = crossVal[..., 1]
    skew[..., 1, 0] = crossVal[..., 2]
    skew[..., 1, 2] = -crossVal[..., 0]
    skew[..., 2, 0] = -crossVal[..., 1]
    skew[..., 2, 1] = crossVal[..., 0]
    factor = ((1.0 - dotVal) / (sinVal.square().clamp(min=eps)))[..., None, None]
    rotation = eye + skew + torch.matmul(skew, skew) * factor
    stableMask = sinVal > eps
    return torch.where(stableMask[..., None, None], rotation, eye)


def kabsch(pointsRest: torch.Tensor, pointsObserved: torch.Tensor) -> torch.Tensor:
    """
    Solve rigid rotation with batched Kabsch.

    pointsRest/pointsObserved: [..., N, 3]
    return: [..., 3, 3]
    """
    if pointsRest.shape != pointsObserved.shape:
        raise ValueError(f"Kabsch input shapes do not match: {pointsRest.shape} vs {pointsObserved.shape}")
    restCentered = pointsRest - pointsRest.mean(dim=-2, keepdim=True)
    obsCentered = pointsObserved - pointsObserved.mean(dim=-2, keepdim=True)
    cov = torch.matmul(restCentered.transpose(-1, -2), obsCentered)
    uVal, _, vVal = torch.linalg.svd(cov)
    detVal = torch.det(torch.matmul(vVal, uVal.transpose(-1, -2)))
    correction = torch.eye(3, device=pointsRest.device, dtype=pointsRest.dtype).expand(*detVal.shape, 3, 3).clone()
    correction[..., 2, 2] = torch.where(detVal < 0, -1.0, 1.0)
    return torch.matmul(torch.matmul(vVal, correction), uVal.transpose(-1, -2))


def compose_local_rotation(parentGlobal: torch.Tensor, childGlobal: torch.Tensor) -> torch.Tensor:
    """Compose local rotation as R_parent^T @ R_child."""
    return torch.matmul(parentGlobal.transpose(-1, -2), childGlobal)
