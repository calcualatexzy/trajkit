"""Validation helpers for trajectory schemas."""

from __future__ import annotations

import polars as pl

REQUIRED_COLUMNS = ("trajectory_id", "t", "x", "y")


def ensure_required_columns(columns: list[str] | tuple[str, ...]) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def maybe_add_z(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    if "z" in (frame.columns if isinstance(frame, pl.DataFrame) else frame.collect_schema().names()):
        return frame
    return frame.with_columns(pl.lit(0.0).alias("z"))
