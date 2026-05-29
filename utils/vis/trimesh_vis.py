"""Small visualization helpers reused across scripts."""
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


def set_axes_equal(ax, points: np.ndarray, *, zoom: float = 1.0) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.52 * float(np.max(maxs - mins))
    radius = max(radius * float(zoom), 0.022)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def style_3d_axes(ax, title: str | None = None) -> None:
    if title:
        ax.set_title(title, pad=8)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    try:
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    except Exception:
        pass


def plot_mesh(ax, verts: np.ndarray, faces: np.ndarray, *, color: str, alpha: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = verts[faces]
    poly = Poly3DCollection(tris, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(poly)


def plot_trimesh(ax, mesh: trimesh.Trimesh, *, color: str, alpha: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    poly = Poly3DCollection(tris, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(poly)


def plot_round_arrow(
    ax,
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    color: str,
    alpha: float = 0.96,
    arrow_ratio: float = 0.24,
) -> None:
    origin = np.asarray(origin, dtype=np.float32)
    unit = np.asarray(direction, dtype=np.float32)
    unit_norm = float(np.linalg.norm(unit))
    if unit_norm < 1.0e-8 or length <= 0.0:
        return
    unit = unit / unit_norm
    tip_length = max(length * float(arrow_ratio), radius * 6.0)
    tip_length = min(tip_length, length * 0.45)
    shaft_length = max(length - tip_length, length * 0.55)

    shaft = trimesh.creation.cylinder(radius=float(radius), height=float(shaft_length), sections=32)
    shaft.apply_transform(_rotation_from_z(unit * shaft_length))
    shaft.apply_translation(origin + unit * (0.5 * shaft_length))
    plot_trimesh(ax, shaft, color=color, alpha=alpha)

    cone = trimesh.creation.cone(radius=float(radius * 2.15), height=float(tip_length), sections=32)
    cone.apply_transform(_rotation_from_z(unit * tip_length))
    cone.apply_translation(origin + unit * (shaft_length - radius * 0.12))
    plot_trimesh(ax, cone, color=color, alpha=alpha)
