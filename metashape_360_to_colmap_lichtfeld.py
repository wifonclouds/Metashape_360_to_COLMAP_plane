#!/usr/bin/env python3
"""Metashape 360 -> COLMAP converter with LichtFeld axis conversion.

This entry point intentionally leaves the upstream converter unchanged. It:
1. runs ``metashape_360_to_colmap.py`` with the supplied arguments;
2. applies the tested global axis conversion to the generated COLMAP scene:
       X' = X, Y' = -Y, Z' = -Z
   and the equivalent W2C pose rotation:
       R'_w2c = R_w2c @ diag(1,-1,-1)

Images and masks are not modified.

Usage:
    python metashape_360_to_colmap_lichtfeld.py \
        --images ./equirect \
        --xml ./cameras.xml \
        --output ./colmap_dataset \
        --ply ./dense.ply

All converter options are accepted and passed through to the vanilla script.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from lichtfeld_axis import transform_colmap_images_txt, transform_points3d_txt, transform_ply_xyz


THIS_DIR = Path(__file__).resolve().parent
VANILLA = THIS_DIR / "metashape_360_to_colmap.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the vanilla Metashape 360 -> COLMAP converter and adapt the result for LichtFeld."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--crop-size", type=int, default=1920)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--flip-vertical", action="store_true", default=True)
    parser.add_argument("--no-flip-vertical", action="store_false", dest="flip_vertical")
    parser.add_argument("--max-images", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--apply-component-transform-for-ply", action="store_true")
    parser.add_argument("--skip-directions", type=str, default="")
    parser.add_argument("--generate-masks", action="store_true")
    parser.add_argument("--external-masks", type=Path)
    parser.add_argument("--yolo-model", type=str, default="yolo11n-seg.pt")
    parser.add_argument("--yolo-classes", type=str, default="0")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--invert-mask", action="store_true")
    parser.add_argument("--yaw-offset", type=float, default=0.0)
    parser.add_argument("--rotate-z180", action="store_true", default=True)
    parser.add_argument("--no-rotate-z180", action="store_false", dest="rotate_z180")
    parser.add_argument("--range-images", type=str)
    parser.add_argument("--mask-overexposure", action="store_true")
    parser.add_argument("--overexposure-threshold", type=int, default=250)
    parser.add_argument("--overexposure-dilate", type=int, default=5)
    parser.add_argument("--output-format", choices=["auto", "jpg", "jpeg", "png", "tiff", "tif", "webp"], default="auto")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-lichtfeld-axis",
        action="store_true",
        help="Run vanilla only; useful for A/B validation.",
    )
    return parser


def run_vanilla(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(VANILLA),
        "--images", str(args.images),
        "--xml", str(args.xml),
        "--output", str(args.output),
        "--crop-size", str(args.crop_size),
        "--fov-deg", str(args.fov_deg),
        "--max-images", str(args.max_images),
        "--num-workers", str(args.num_workers),
        "--yolo-model", str(args.yolo_model),
        "--yolo-classes", str(args.yolo_classes),
        "--yolo-conf", str(args.yolo_conf),
        "--yaw-offset", str(args.yaw_offset),
        "--overexposure-threshold", str(args.overexposure_threshold),
        "--overexposure-dilate", str(args.overexposure_dilate),
        "--output-format", args.output_format,
    ]

    if args.flip_vertical:
        cmd.append("--flip-vertical")
    else:
        cmd.append("--no-flip-vertical")

    if args.apply_component_transform_for_ply:
        cmd.append("--apply-component-transform-for-ply")
    if args.skip_directions:
        cmd += ["--skip-directions", args.skip_directions]
    if args.generate_masks:
        cmd.append("--generate-masks")
    if args.external_masks:
        cmd += ["--external-masks", str(args.external_masks)]
    if args.invert_mask:
        cmd.append("--invert-mask")
    if args.rotate_z180:
        cmd.append("--rotate-z180")
    else:
        cmd.append("--no-rotate-z180")
    if args.range_images:
        cmd += ["--range-images", args.range_images]
    if args.mask_overexposure:
        cmd.append("--mask-overexposure")
    if args.quiet:
        cmd.append("--quiet")

    return subprocess.call(cmd, cwd=str(THIS_DIR))


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
        # The vanilla converter writes this PLY with Open3D. Instead of parsing
        # binary data here, make a backup and leave the generated PLY untouched.
        # points3D.txt is the COLMAP point representation used by the dataset.
        backup = ply.with_suffix(ply.suffix + ".vanilla")
        if not backup.exists():
            shutil.copy2(ply, backup)
        if not quiet:
            print(f"LichtFeld axis: left binary PLY unchanged (backup: {backup.name})")


def main() -> int:
    args = build_parser().parse_args()

    if not VANILLA.is_file():
        print(f"Error: vanilla converter not found: {VANILLA}")
        return 1

    code = run_vanilla(args)
    if code != 0:
        return code

    if not args.no_lichtfeld_axis:
        apply_lichtfeld_conversion(args.output, quiet=args.quiet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
