#!/usr/bin/env python3
"""Align two 2D trajectories using both position and yaw constraints.

The script estimates a rigid-body transform in SE(2):

    p_target ~= R(theta) @ p_source + t
    yaw_target ~= yaw_source + theta

Rotation is solved analytically from a weighted objective that combines:
1. Position correspondences in the x-y plane.
2. Heading correspondences derived from yaw angles.

Translation is then computed from the aligned position centroids.

Input CSV files must contain columns named: x, y, yaw
Rows are assumed to be corresponding poses in the same order.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Transform2D:
    theta: float
    tx: float
    ty: float


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def read_trajectory(csv_path: Path, degrees: bool) -> List[Pose2D]:
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is missing a CSV header.")

        required = {"x", "y", "yaw"}
        missing = required - set(reader.fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required columns: {missing_text}")

        poses: List[Pose2D] = []
        for row_index, row in enumerate(reader, start=2):
            try:
                x = float(row["x"])
                y = float(row["y"])
                yaw = float(row["yaw"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row {row_index} in {csv_path}: {row}"
                ) from exc

            if degrees:
                yaw = math.radians(yaw)

            poses.append(Pose2D(x=x, y=y, yaw=yaw))

    if not poses:
        raise ValueError(f"{csv_path} does not contain any poses.")

    return poses


def mean_xy(poses: Sequence[Pose2D]) -> tuple[float, float]:
    x_mean = sum(pose.x for pose in poses) / len(poses)
    y_mean = sum(pose.y for pose in poses) / len(poses)
    return x_mean, y_mean


def estimate_transform(
    source: Sequence[Pose2D],
    target: Sequence[Pose2D],
    position_weight: float,
    yaw_weight: float,
) -> Transform2D:
    if len(source) != len(target):
        raise ValueError(
            "Source and target trajectories must contain the same number of poses. "
            "Rows are matched by index."
        )

    if not source:
        raise ValueError("At least one pose is required.")

    if position_weight < 0.0 or yaw_weight < 0.0:
        raise ValueError("Weights must be non-negative.")

    if position_weight == 0.0 and yaw_weight == 0.0:
        raise ValueError("At least one of position_weight or yaw_weight must be positive.")

    source_mean_x, source_mean_y = mean_xy(source)
    target_mean_x, target_mean_y = mean_xy(target)

    cos_term = 0.0
    sin_term = 0.0

    if position_weight > 0.0:
        for source_pose, target_pose in zip(source, target):
            src_x = source_pose.x - source_mean_x
            src_y = source_pose.y - source_mean_y
            tgt_x = target_pose.x - target_mean_x
            tgt_y = target_pose.y - target_mean_y

            cos_term += position_weight * (src_x * tgt_x + src_y * tgt_y)
            sin_term += position_weight * (src_x * tgt_y - src_y * tgt_x)

    if yaw_weight > 0.0:
        for source_pose, target_pose in zip(source, target):
            src_hx = math.cos(source_pose.yaw)
            src_hy = math.sin(source_pose.yaw)
            tgt_hx = math.cos(target_pose.yaw)
            tgt_hy = math.sin(target_pose.yaw)

            cos_term += yaw_weight * (src_hx * tgt_hx + src_hy * tgt_hy)
            sin_term += yaw_weight * (src_hx * tgt_hy - src_hy * tgt_hx)

    if math.isclose(cos_term, 0.0, abs_tol=1e-12) and math.isclose(
        sin_term, 0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Rotation is not observable from the provided data. Add more geometric "
            "variation, or use a positive yaw_weight."
        )

    theta = math.atan2(sin_term, cos_term)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    tx = target_mean_x - (cos_theta * source_mean_x - sin_theta * source_mean_y)
    ty = target_mean_y - (sin_theta * source_mean_x + cos_theta * source_mean_y)

    return Transform2D(theta=theta, tx=tx, ty=ty)


def apply_transform(poses: Iterable[Pose2D], transform: Transform2D) -> List[Pose2D]:
    cos_theta = math.cos(transform.theta)
    sin_theta = math.sin(transform.theta)

    aligned: List[Pose2D] = []
    for pose in poses:
        x = cos_theta * pose.x - sin_theta * pose.y + transform.tx
        y = sin_theta * pose.x + cos_theta * pose.y + transform.ty
        yaw = wrap_to_pi(pose.yaw + transform.theta)
        aligned.append(Pose2D(x=x, y=y, yaw=yaw))

    return aligned


def compute_metrics(
    aligned_source: Sequence[Pose2D],
    target: Sequence[Pose2D],
    degrees: bool,
) -> dict[str, float]:
    if len(aligned_source) != len(target):
        raise ValueError("Metric inputs must have the same number of poses.")

    position_error_sq = 0.0
    yaw_error_sq = 0.0

    for aligned_pose, target_pose in zip(aligned_source, target):
        dx = aligned_pose.x - target_pose.x
        dy = aligned_pose.y - target_pose.y
        dyaw = wrap_to_pi(aligned_pose.yaw - target_pose.yaw)

        position_error_sq += dx * dx + dy * dy
        yaw_error_sq += dyaw * dyaw

    count = len(aligned_source)
    position_rmse = math.sqrt(position_error_sq / count)
    yaw_rmse = math.sqrt(yaw_error_sq / count)

    if degrees:
        yaw_rmse = math.degrees(yaw_rmse)

    return {
        "position_rmse": position_rmse,
        "yaw_rmse": yaw_rmse,
    }


def write_trajectory(csv_path: Path, poses: Sequence[Pose2D], degrees: bool) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["x", "y", "yaw"])
        writer.writeheader()
        for pose in poses:
            yaw = math.degrees(pose.yaw) if degrees else pose.yaw
            writer.writerow({"x": pose.x, "y": pose.y, "yaw": yaw})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align a source trajectory to a target trajectory using both x/y positions "
            "and yaw angles to constrain a 2D rigid transform."
        )
    )
    parser.add_argument("source_csv", type=Path, help="CSV file with source trajectory")
    parser.add_argument("target_csv", type=Path, help="CSV file with target trajectory")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the aligned source trajectory as CSV",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret the yaw column as degrees instead of radians",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=1.0,
        help="Weight of the x/y position constraints (default: 1.0)",
    )
    parser.add_argument(
        "--yaw-weight",
        type=float,
        default=1.0,
        help="Weight of the yaw constraints (default: 1.0)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source = read_trajectory(args.source_csv, degrees=args.degrees)
    target = read_trajectory(args.target_csv, degrees=args.degrees)

    transform = estimate_transform(
        source=source,
        target=target,
        position_weight=args.position_weight,
        yaw_weight=args.yaw_weight,
    )
    aligned_source = apply_transform(source, transform)
    metrics = compute_metrics(aligned_source, target, degrees=args.degrees)

    angle_unit = "deg" if args.degrees else "rad"
    print("Estimated transform:")
    print(f"  theta: {transform.theta:.12f} rad")
    if args.degrees:
        print(f"         {math.degrees(transform.theta):.12f} deg")
    print(f"  tx:    {transform.tx:.12f}")
    print(f"  ty:    {transform.ty:.12f}")
    print("Alignment error:")
    print(f"  position RMSE: {metrics['position_rmse']:.12f}")
    print(f"  yaw RMSE:      {metrics['yaw_rmse']:.12f} {angle_unit}")

    if args.output is not None:
        write_trajectory(args.output, aligned_source, degrees=args.degrees)
        print(f"Aligned trajectory written to: {args.output}")


if __name__ == "__main__":
    main()
