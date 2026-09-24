"""Автоматическая проверка комплектности и ключевых требований ЛР1–ЛР2."""

from __future__ import annotations

import csv
import importlib.util
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_links() -> None:
    for document in ROOT.rglob("*.md"):
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]+)?\)", text):
            if "://" not in target:
                require((document.parent / target).exists(),
                        f"Нерабочая ссылка: {document}: {target}")


def verify_lab1() -> None:
    directory = ROOT / "lab_1_hgbat"
    module = load_module("lab1_hgbat", directory / "main.py")
    require(module.DIMENSION == 7, "ЛР1: для V=20 должна быть размерность d=7")
    require(module.LOWER_BOUND == -15 and module.UPPER_BOUND == 15,
            "ЛР1: неверная область HGBat")
    require(math.isclose(module.hgbat([-1.0] * 7), 0.0, abs_tol=1e-12),
            "ЛР1: HGBat(-1,…,-1) должна равняться 0")

    results = directory / "results"
    runs = rows(results / "runs.csv")
    require(len(runs) == 60, "ЛР1: ожидаются 3 метода × 20 запусков")
    counts = Counter(row["method"] for row in runs)
    require(counts == {"GA_sigma_2pct": 20, "GA_sigma_8pct": 20, "random_search": 20},
            "ЛР1: неверное число независимых запусков")
    require(all(int(row["evaluations"]) == 12_000 for row in runs),
            "ЛР1: методы должны иметь одинаковый бюджет 12000")
    require(len(rows(results / "summary.csv")) == 3, "ЛР1: неполная сводка")
    require(len(rows(results / "parameters.csv")) == 3, "ЛР1: нет таблицы параметров")
    ET.parse(results / "convergence.svg")


def verify_lab2() -> None:
    directory = ROOT / "lab_2_weekly_menu"
    module = load_module("lab2_menu", directory / "main.py")
    dishes = module.generate_instance()
    require(len(dishes) == 36, "ЛР2: для меню требуется не менее 30 блюд")
    require(Counter(dish.meal for dish in dishes) == {meal: 12 for meal in module.MEALS},
            "ЛР2: неверное распределение блюд по типам")
    allowed = module.allowed_by_slot(dishes)
    rng = random.Random(51_720)
    for _ in range(100):
        repaired = module.repair(module.random_individual(allowed, rng), dishes, allowed, rng)
        require(module.evaluate(repaired, dishes).feasible,
                "ЛР2: оператор восстановления вернул недопустимое меню")

    results = directory / "results"
    runs = rows(results / "runs.csv")
    require(len(runs) == 80, "ЛР2: ожидаются 4 метода × 20 запусков")
    counts = Counter(row["method"] for row in runs)
    require(all(count == 20 for count in counts.values()) and len(counts) == 4,
            "ЛР2: неверное число независимых запусков")
    require(all(int(row["evaluations"]) == 14_300 for row in runs),
            "ЛР2: методы должны иметь сопоставимый бюджет 14300")
    ga_methods = {"repair_uniform", "penalty_uniform", "repair_one_point"}
    require(all(row["feasible"] == "True" for row in runs if row["method"] in ga_methods),
            "ЛР2: финальные решения ГА должны быть допустимыми")
    require(len(rows(results / "parameters.csv")) == 3, "ЛР2: нет таблицы параметров")

    menu = rows(results / "best_menu.csv")
    require(len(menu) == 21, "ЛР2: меню должно содержать 7 × 3 приёма пищи")
    require(sum(int(row["cost"]) for row in menu) <= module.WEEKLY_BUDGET,
            "ЛР2: лучшее меню превышает бюджет")
    require(len({row["dish"] for row in menu}) >= module.MIN_UNIQUE_DISHES,
            "ЛР2: недостаточное разнообразие")
    require(max(Counter(row["dish"] for row in menu).values()) <= module.MAX_REPETITIONS,
            "ЛР2: блюдо повторяется слишком часто")
    for day in range(7):
        calories = sum(int(row["calories"]) for row in menu[day * 3:day * 3 + 3])
        require(module.MIN_DAILY_CALORIES <= calories <= module.MAX_DAILY_CALORIES,
                f"ЛР2: нарушена калорийность дня {day + 1}")

    examples = rows(results / "feasibility_examples.csv")
    require([row["feasible"] for row in examples].count("True") == 2,
            "ЛР2: нужны два допустимых примера")
    require([row["feasible"] for row in examples].count("False") == 2,
            "ЛР2: нужны два недопустимых примера")
    ET.parse(results / "convergence.svg")


def main() -> None:
    verify_links()
    verify_lab1()
    verify_lab2()
    print("OK: вариант 20 и все проверяемые требования ЛР1–ЛР2 выполнены")


if __name__ == "__main__":
    main()
