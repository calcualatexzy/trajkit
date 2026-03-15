"""Shared typing aliases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PathLike = str | Path
StatsDict = dict[str, Any]
