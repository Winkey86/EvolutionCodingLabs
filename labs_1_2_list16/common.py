"""Общие утилиты: статистика, CSV и простые SVG-графики."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Нельзя вычислить статистику пустой выборки")
    return {
        "best": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "worst": max(values),
    }


def _escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def line_chart_svg(
    path: Path,
    series: dict[str, Sequence[float]],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Записывает компактный линейный SVG-график без внешних библиотек."""
    if not series or any(not values for values in series.values()):
        return
    ensure_dir(path.parent)
    width, height = 960, 560
    left, right, top, bottom = 85, 25, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_values = [float(v) for values in series.values() for v in values if math.isfinite(v)]
    if not all_values:
        return
    y_min, y_max = min(all_values), max(all_values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    padding = 0.05 * (y_max - y_min)
    y_min -= padding
    y_max += padding
    n = max(len(values) for values in series.values())

    def x_pos(index: int) -> float:
        return left + (index / max(1, n - 1)) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = ["#1565c0", "#ef6c00", "#2e7d32", "#6a1b9a", "#c62828"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{_escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="monospace" font-size="12">{value:.4g}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
            f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">{_escape(x_label)}</text>',
            f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{_escape(y_label)}</text>',
        ]
    )
    for idx, (name, values) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{x_pos(i):.2f},{y_pos(float(v)):.2f}" for i, v in enumerate(values))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>')
        legend_x = left + idx * 180
        parts.append(f'<line x1="{legend_x}" y1="{height-48}" x2="{legend_x+28}" y2="{height-48}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+34}" y="{height-43}" font-family="sans-serif" font-size="13">{_escape(name)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
