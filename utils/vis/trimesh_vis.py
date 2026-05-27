"""Small trimesh visualization helpers reused across scripts."""
from __future__ import annotations

import math

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix


def normalize_vector(vec: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return np.zeros_like(arr)
    return arr / norm


def add_sphere(scene: trimesh.Scene, center: np.ndarray, radius: float, color: np.ndarray, name: str) -> None:
    sphere = trimesh.creation.uv_sphere(radius=float(radius))
    sphere.apply_translation(np.asarray(center, dtype=np.float32))
    sphere.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8)[None, :], (sphere.vertices.shape[0], 1))
    scene.add_geometry(sphere, node_name=name)


def _rotation_from_z(axis: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-8:
        return np.eye(4, dtype=np.float32)
    unit = axis / axis_norm
    dot_val = float(np.clip(np.dot(z_axis, unit), -1.0, 1.0))
    if math.isclose(dot_val, 1.0):
        return np.eye(4, dtype=np.float32)
    if math.isclose(dot_val, -1.0):
        return rotation_matrix(math.pi, [1.0, 0.0, 0.0]).astype(np.float32)
    rot_axis = np.cross(z_axis, unit)
    angle = math.acos(dot_val)
    return rotation_matrix(angle, rot_axis).astype(np.float32)


def add_cylinder(
    scene: trimesh.Scene,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: np.ndarray,
    name: str,
) -> None:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1.0e-8:
        return
    cylinder = trimesh.creation.cylinder(radius=float(radius), height=length, sections=24)
    cylinder.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8)[None, :], (cylinder.vertices.shape[0], 1))
    transform = _rotation_from_z(vec)
    transform[:3, 3] = 0.5 * (start + end)
    cylinder.apply_transform(transform)
    scene.add_geometry(cylinder, node_name=name)


def add_axes(
    scene: trimesh.Scene,
    *,
    origin: np.ndarray,
    axes: dict[str, np.ndarray],
    axisLength: float,
    radius: float,
    prefix: str,
) -> None:
    colors = {
        "x": np.array([230, 70, 70, 255], dtype=np.uint8),
        "y": np.array([70, 185, 110, 255], dtype=np.uint8),
        "z": np.array([70, 125, 230, 255], dtype=np.uint8),
    }
    origin = np.asarray(origin, dtype=np.float32)
    add_sphere(scene, origin, radius * 1.35, np.array([240, 220, 130, 255], dtype=np.uint8), f"{prefix}_origin")
    for axis_name, direction in axes.items():
        end = origin + float(axisLength) * normalize_vector(direction)
        add_cylinder(scene, origin, end, radius, colors[axis_name], f"{prefix}_{axis_name}")
        add_sphere(scene, end, radius * 1.45, colors[axis_name], f"{prefix}_{axis_name}_tip")


def build_hand_mesh(verts: np.ndarray, faces: np.ndarray, *, color: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces.astype(np.int64), process=False)
    mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8)[None, :], (mesh.vertices.shape[0], 1))
    return mesh
