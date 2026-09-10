#!/usr/bin/env python3
"""Metashape 360 -> COLMAP converter with LichtFeld axis conversion.

This wrapper deliberately keeps the vanilla converter as the single source of
truth for conversion. Every vanilla CLI argument is passed through unchanged.
After vanilla finishes, the generated COLMAP scene is adapted for LichtFeld
with the tested global world-axis conversion:

    X' = X
    Y' = -Y
    Z' = -Z

For COLMAP world-to-camera poses:

    R'_w2c = R_w2c @ diag(1,-1,-1)

Images and masks are not modified.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from lichtfeld_axis import transform_colmap_images_txt, transform_points3d_txt


THIS_DIR = Path(__file__).resolve().parent
VANILLA = THIS_DIR / "metashape_360_to_colmap.py"


def parse_wrapper_args() -> tuple[Path, bool, list[str]]:
    """Extract only wrapper-specific arguments.

    All other arguments are forwarded byte-for-byte to vanilla. This prevents
    the adapter from drifting when the upstream CLI gains or changes options.
    """
    parser = argparse.ArgumentParser(
        description="Run vanilla Metashape 360 -> COLMAP and adapt the result for LichtFeld."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-lichtfeld-axis",
        action="store_true",
        help="Run vanilla only; useful for A/B validation.",
    )

    args, unknown = parser.parse_known_args()

    # argparse consumes --output. Put it back so vanilla receives the exact
    # same option. --no-lichtfeld-axis is wrapper-only and is not forwarded.
    vanilla_args = list(unknown)
    vanilla_args += ["--output", str(args.output)]

    return args.output, args.no_lichtfeld_axis, vanilla_args


def run_vanilla(vanilla_args: list[str]) -> int:
    cmd = [sys.executable, str(VANILLA), *vanilla_args]
    print("Running vanilla converter:")
    print(" ".join(cmd))
    print()
    return subprocess.call(cmd, cwd=str(THIS_DIR))


def transform_ply_with_open3d(ply: Path) -> int:
    """Apply X,Y,Z -> X,-Y,-Z to the vanilla-generated PLY.

    Vanilla writes the PLY through Open3D and normally produces a binary PLY.
    Therefore the old ASCII-only helper was not suitable here. We use Open3D
    to read/write the point cloud, preserving its point attributes such as
    colors.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required to transform points3D.ply") from exc

    pc = o3d.io.read_point_cloud(str(ply))
    points = np.asarray(pc.points)
    if points.size == 0:
        raise RuntimeError(f"Open3D loaded zero points from {ply}")

    points = points.copy()
    points[:, 1] *= -1.0
    points[:, 2] *= -1.0
    pc.points = o3d.utility.Vector3dVector(points)

    backup = ply.with_suffix(ply.suffix + ".vanilla")
    if not backup.exists():
        import shutil
        shutil.copy2(ply, backup)

    if not o3d.io.write_point_cloud(str(ply), pc):
        raise RuntimeError(f"Failed to write transformed PLY: {ply}")

    return len(points)


def apply_lichtfeld_conversion(output_dir: Path, quiet: bool = False) -> None:
    images_txt = output_dir / "images.txt"
    points_txt = output_dir / "points3D.txt"
    ply = output_dir / "points3D.ply"

    if images_txt.exists():
        n = transform_colmap_images_txt(images_txt)
        if not quiet:
            print(f"LichtFeld axis: transformed {n} camera poses in {images_txt}")

    if points_txt.exists():
        n = transform_points3d_txt(points_txt)
        if not quiet:
            print(f"LichtFeld axis: transformed {n} points in {points_txt}")

    if ply.exists():
        n = transform_ply_with_open3d(ply)
        if not quiet:
            print(f"LichtFeld axis: transformed {n} points in {ply}")


def main() -> int:
    output_dir, no_axis, vanilla_args = parse_wrapper_args()

    if not VANILLA.is_file():
        print(f"Error: vanilla converter not found: {VANILLA}")
        return 1

    code = run_vanilla(vanilla_args)
    if code != 0:
        return code

    if not no_axis:
        apply_lichtfeld_conversion(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
