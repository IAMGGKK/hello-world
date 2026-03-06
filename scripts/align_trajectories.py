#!/usr/bin/env python3
"""Align two 2D trajectories with a numpy/scipy least-squares optimizer.

The current model estimates a rigid-body transform in SE(2):

    p_target ~= R(theta) @ p_source + t
    yaw_target ~= yaw_source + theta

Residual blocks are assembled explicitly and passed to
``scipy.optimize.least_squares``:

1. Position residuals for x/y correspondences.
2. Angle residuals for yaw correspondences.

The code keeps parameter packing/unpacking separate from residual assembly so
future parameters (for example scale, sensor extrinsics, or time offsets) can
be added without rewriting the optimization pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult, least_squares


@dataclass(frozen=True)
class Trajectory2D:
    xy: np.ndarray
    yaw: np.ndarray

    def __post_init__(self) -> None:
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2).")
        if self.yaw.ndim != 1 or self.yaw.shape[0] != self.xy.shape[0]:
            raise ValueError("yaw must have shape (N,) and match xy.")

    @property
    def size(self) -> int:
        return int(self.yaw.shape[0])


@dataclass(frozen=True)
class Transform2D:
    theta: float
    tx: float
    ty: float


@dataclass(frozen=True)
class ParameterLayout:
    names: tuple[str, ...]

    def pack_transform(self, transform: Transform2D) -> np.ndarray:
        return np.array([transform.theta, transform.tx, transform.ty], dtype=float)

    def unpack_transform(self, params: np.ndarray) -> Transform2D:
        if params.shape != (len(self.names),):
            raise ValueError(
                f"Expected parameter vector of shape ({len(self.names)},), "
                f"got {params.shape}."
            )
        return Transform2D(theta=float(params[0]), tx=float(params[1]), ty=float(params[2]))


@dataclass(frozen=True)
class AlignmentConfig:
    position_weight: float
    yaw_weight: float
    loss: str
    f_scale: float
    max_nfev: int | None


DEFAULT_LAYOUT = ParameterLayout(names=("theta", "tx", "ty"))


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    angle_array = np.asarray(angle, dtype=float)
    return (angle_array + np.pi) % (2.0 * np.pi) - np.pi


def read_trajectory(csv_path: Path, degrees: bool) -> Trajectory2D:
    rows: list[tuple[float, float, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is missing a CSV header.")

        required = {"x", "y", "yaw"}
        missing = required - set(reader.fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required columns: {missing_text}")

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

            rows.append((x, y, yaw))

    if not rows:
        raise ValueError(f"{csv_path} does not contain any poses.")

    xy = np.asarray([(x, y) for x, y, _ in rows], dtype=float)
    yaw = np.asarray([yaw_value for _, _, yaw_value in rows], dtype=float)
    return Trajectory2D(xy=xy, yaw=yaw)


def validate_problem(
    source: Trajectory2D,
    target: Trajectory2D,
    config: AlignmentConfig,
) -> None:
    if source.size != target.size:
        raise ValueError(
            "Source and target trajectories must contain the same number of poses. "
            "Rows are matched by index."
        )

    if source.size == 0:
        raise ValueError("At least one pose is required.")

    if config.position_weight < 0.0 or config.yaw_weight < 0.0:
        raise ValueError("Weights must be non-negative.")

    if config.position_weight == 0.0 and config.yaw_weight == 0.0:
        raise ValueError("At least one residual block must be enabled.")

    if config.position_weight <= 0.0:
        raise ValueError(
            "position_weight must be positive because tx and ty are optimized "
            "variables in the current model."
        )

    source_centered = source.xy - source.xy.mean(axis=0, keepdims=True)
    target_centered = target.xy - target.xy.mean(axis=0, keepdims=True)
    spread = np.linalg.norm(source_centered) + np.linalg.norm(target_centered)

    if config.yaw_weight == 0.0 and spread < 1e-12:
        raise ValueError(
            "Rotation is not observable from position alone. Add geometric "
            "variation or use a positive yaw_weight."
        )

    if config.f_scale <= 0.0:
        raise ValueError("f_scale must be positive.")

    if config.max_nfev is not None and config.max_nfev <= 0:
        raise ValueError("max_nfev must be positive when provided.")


def estimate_initial_transform(
    source: Trajectory2D,
    target: Trajectory2D,
    config: AlignmentConfig,
) -> Transform2D:
    source_centered = source.xy - source.xy.mean(axis=0, keepdims=True)
    target_centered = target.xy - target.xy.mean(axis=0, keepdims=True)

    cos_term = 0.0
    sin_term = 0.0

    if config.position_weight > 0.0:
        cos_term += config.position_weight * float(np.sum(source_centered * target_centered))
        sin_term += config.position_weight * float(
            np.sum(source_centered[:, 0] * target_centered[:, 1])
            - np.sum(source_centered[:, 1] * target_centered[:, 0])
        )

    if config.yaw_weight > 0.0:
        yaw_delta = target.yaw - source.yaw
        cos_term += config.yaw_weight * float(np.sum(np.cos(yaw_delta)))
        sin_term += config.yaw_weight * float(np.sum(np.sin(yaw_delta)))

    theta = math.atan2(sin_term, cos_term)
    rotation = rotation_matrix(theta)
    translation = target.xy.mean(axis=0) - (source.xy @ rotation.T).mean(axis=0)

    return Transform2D(theta=theta, tx=float(translation[0]), ty=float(translation[1]))


def rotation_matrix(theta: float) -> np.ndarray:
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=float)


def apply_transform(source: Trajectory2D, transform: Transform2D) -> Trajectory2D:
    rotation = rotation_matrix(transform.theta)
    translation = np.array([transform.tx, transform.ty], dtype=float)
    aligned_xy = source.xy @ rotation.T + translation
    aligned_yaw = wrap_to_pi(source.yaw + transform.theta)
    return Trajectory2D(xy=aligned_xy, yaw=np.asarray(aligned_yaw, dtype=float))


def build_residual_vector(
    params: np.ndarray,
    layout: ParameterLayout,
    source: Trajectory2D,
    target: Trajectory2D,
    config: AlignmentConfig,
) -> np.ndarray:
    transform = layout.unpack_transform(params)
    aligned_source = apply_transform(source, transform)
    residual_blocks: list[np.ndarray] = []

    if config.position_weight > 0.0:
        position_residual = math.sqrt(config.position_weight) * (
            aligned_source.xy - target.xy
        )
        residual_blocks.append(position_residual.reshape(-1))

    if config.yaw_weight > 0.0:
        yaw_residual = math.sqrt(config.yaw_weight) * wrap_to_pi(
            aligned_source.yaw - target.yaw
        )
        residual_blocks.append(np.asarray(yaw_residual, dtype=float))

    return np.concatenate(residual_blocks)


def optimize_transform(
    source: Trajectory2D,
    target: Trajectory2D,
    config: AlignmentConfig,
    layout: ParameterLayout = DEFAULT_LAYOUT,
) -> tuple[Transform2D, OptimizeResult]:
    validate_problem(source, target, config)
    initial_transform = estimate_initial_transform(source, target, config)
    initial_params = layout.pack_transform(initial_transform)

    result = least_squares(
        build_residual_vector,
        x0=initial_params,
        args=(layout, source, target, config),
        method="trf",
        loss=config.loss,
        f_scale=config.f_scale,
        max_nfev=config.max_nfev,
    )
    if not result.success:
        raise RuntimeError(f"Least-squares optimization failed: {result.message}")

    return layout.unpack_transform(result.x), result


def compute_metrics(
    aligned_source: Trajectory2D,
    target: Trajectory2D,
    degrees: bool,
) -> dict[str, float]:
    if aligned_source.size != target.size:
        raise ValueError("Metric inputs must have the same number of poses.")

    position_rmse = float(
        np.sqrt(np.mean(np.sum((aligned_source.xy - target.xy) ** 2, axis=1)))
    )
    yaw_rmse = float(np.sqrt(np.mean(wrap_to_pi(aligned_source.yaw - target.yaw) ** 2)))

    if degrees:
        yaw_rmse = math.degrees(yaw_rmse)

    return {"position_rmse": position_rmse, "yaw_rmse": yaw_rmse}


def write_trajectory(csv_path: Path, trajectory: Trajectory2D, degrees: bool) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["x", "y", "yaw"])
        writer.writeheader()
        for xy, yaw in zip(trajectory.xy, trajectory.yaw):
            yaw_value = math.degrees(float(yaw)) if degrees else float(yaw)
            writer.writerow({"x": float(xy[0]), "y": float(xy[1]), "yaw": yaw_value})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align a source trajectory to a target trajectory using weighted "
            "x/y and yaw residuals solved by scipy.optimize.least_squares."
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
        help="Weight of the x/y position residual block (default: 1.0)",
    )
    parser.add_argument(
        "--yaw-weight",
        type=float,
        default=1.0,
        help="Weight of the yaw residual block (default: 1.0)",
    )
    parser.add_argument(
        "--loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="linear",
        help="Robust loss passed to scipy.optimize.least_squares (default: linear)",
    )
    parser.add_argument(
        "--f-scale",
        type=float,
        default=1.0,
        help="f_scale argument for the chosen robust loss (default: 1.0)",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="Optional maximum number of function evaluations",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source = read_trajectory(args.source_csv, degrees=args.degrees)
    target = read_trajectory(args.target_csv, degrees=args.degrees)
    config = AlignmentConfig(
        position_weight=args.position_weight,
        yaw_weight=args.yaw_weight,
        loss=args.loss,
        f_scale=args.f_scale,
        max_nfev=args.max_nfev,
    )

    transform, optimizer_result = optimize_transform(source=source, target=target, config=config)
    aligned_source = apply_transform(source, transform)
    metrics = compute_metrics(aligned_source, target, degrees=args.degrees)

    angle_unit = "deg" if args.degrees else "rad"
    print("Estimated transform:")
    print(f"  theta: {transform.theta:.12f} rad")
    print(f"         {math.degrees(transform.theta):.12f} deg")
    print(f"  tx:    {transform.tx:.12f}")
    print(f"  ty:    {transform.ty:.12f}")
    print("Optimizer:")
    print("  solver: scipy.optimize.least_squares")
    print(f"  loss:   {args.loss}")
    print(f"  nfev:   {optimizer_result.nfev}")
    print(f"  cost:   {optimizer_result.cost:.12f}")
    print("Alignment error:")
    print(f"  position RMSE: {metrics['position_rmse']:.12f}")
    print(f"  yaw RMSE:      {metrics['yaw_rmse']:.12f} {angle_unit}")

    if args.output is not None:
        write_trajectory(args.output, aligned_source, degrees=args.degrees)
        print(f"Aligned trajectory written to: {args.output}")


if __name__ == "__main__":
    main()
