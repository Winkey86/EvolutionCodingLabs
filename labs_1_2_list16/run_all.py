"""Запуск обеих лабораторных для одной группы."""

from __future__ import annotations

import argparse
from pathlib import Path

import lab1
import lab2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, choices=(507, 517, 527), required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    print("ЛР1:", lab1.experiment(args.group, args.runs, args.output / "lab1"))
    print("ЛР2:", lab2.experiment(args.group, args.runs, args.output / "lab2"))


if __name__ == "__main__":
    main()
