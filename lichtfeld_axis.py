#!/usr/bin/env python3
"""Axis conversion helpers for Metashape -> LichtFeld datasets.

The tested global world-axis conversion is:
    X' = X
    Y' = -Y
    Z' = -Z

For COLMAP image poses (world-to-camera rotation), the equivalent operation is:
    R'_w2c = R_w2c @ S
where S = diag(1, -1, -1).

Translations in COLMAP images.txt stay unchanged because they are part of the
world-to-camera representation rather than camera centers directly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


S = np.diag([1.0, -1.0, -1.0])


def quaternion_wxyz_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a COLMAP WXYZ quaternion to a 3x3 rotation matrix."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        raise ValueError("Zero-length quaternion")

    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)


def rotation_to_quaternion_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a WXYZ quaternion."""
    R = np.asarray(R, dtype=float)
    trace = float(np.trace(R))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    return tuple(float(x) for x in q)


def transform_colmap_images_txt(path: Path) -> int:
    """Apply the LichtFeld axis conversion to COLMAP images.txt.

    Camera/image names and 2D observations remain unchanged. Only the pose
    quaternion is changed. Translation is intentionally preserved.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    output: List[str] = []
    transformed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue

        fields = stripped.split()
        # COLMAP image pose line has 10 fields. The following line with 2D
        # observations has a different structure and must remain untouched.
        if len(fields) >= 10 and fields[0].isdigit():
            try:
                image_id = int(fields[0])
                qw, qx, qy, qz = map(float, fields[1:5])
                tx, ty, tz = map(float, fields[5:8])
                camera_id = int(fields[8])
            except ValueError:
                output.append(line)
                continue

            R = quaternion_wxyz_to_rotation(qw, qx, qy, qz)
            R_new = R @ S
            qw2, qx2, qy2, qz2 = rotation_to_quaternion_wxyz(R_new)

            name = " ".join(fields[9:])
            output.append(
                f"{image_id} {qw2:.17g} {qx2:.17g} {qy2:.17g} {qz2:.17g} "
                f"{tx:.17g} {ty:.17g} {tz:.17g} {camera_id} {name}"
            )
            transformed += 1
        else:
            output.append(line)

    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return transformed


def transform_points3d_txt(path: Path) -> int:
    """Apply X,Y,Z -> X,-Y,-Z to COLMAP points3D.txt."""
    lines = path.read_text(encoding="utf-8").splitlines()
    output: List[str] = []
    transformed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue

        fields = stripped.split()
        if len(fields) >= 7 and fields[0].isdigit():
            try:
                float(fields[1])
                float(fields[2])
                float(fields[3])
            except ValueError:
                output.append(line)
                continue

            x = float(fields[1])
            y = float(fields[2])
            z = float(fields[3])
            fields[1] = f"{x:.17g}"
            fields[2] = f"{-y:.17g}"
            fields[3] = f"{-z:.17g}"
            output.append(" ".join(fields))
            transformed += 1
        else:
            output.append(line)

    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return transformed


def transform_ply_xyz(path: Path) -> int:
    """Transform PLY vertex XYZ coordinates while preserving other properties.

    This supports common ASCII PLY files. Binary PLY files are deliberately not
    modified here because silently rewriting binary layouts would be unsafe.
    """
    raw = path.read_bytes()
    header_end = raw.find(b"end_header\n")
    if header_end < 0:
        raise ValueError(f"PLY header not found in {path}")
    header_end += len(b"end_header\n")

    header = raw[:header_end].decode("utf-8")
    if "format ascii" not in header:
        raise ValueError("Only ASCII PLY is supported by transform_ply_xyz")

    lines = raw[header_end:].decode("utf-8").splitlines()
    vertex_count: Optional[int] = None
    for line in header.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            break

    if vertex_count is None:
        raise ValueError("PLY vertex count not found")

    output = []
    for index, line in enumerate(lines):
        if index >= vertex_count:
            output.append(line)
            continue
        fields = line.split()
        if len(fields) < 3:
            output.append(line)
            continue
        fields[1] = f"{-float(fields[1]):.17g}"
        fields[2] = f"{-float(fields[2]):.17g}"
        # Intentionally keep X unchanged.
        output.append(" ".join(fields))

    path.write_text(header + "\n".join(output) + "\n", encoding="utf-8")
    return min(vertex_count, len(lines))
