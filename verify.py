"""Проверка сдаваемых материалов ЛР1–ЛР4 и пересчёт результатов ЛР3–ЛР4."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import re
import statistics
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
        if ".git" in document.parts or "graphify-out" in document.parts:
            continue
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

    upper_bound = module.utility_upper_bound(dishes)
    certificate = json.loads((results / "optimality.json").read_text(encoding="utf-8"))
    require(certificate["upper_bound"] == upper_bound, "ЛР2: неверная верхняя граница")
    best_utility = sum(int(row["score"]) for row in menu)
    require(certificate["certified_optimal"] and best_utility == upper_bound,
            "ЛР2: оптимальность сохранённого меню не подтверждена")
    for record in rows(results / "summary.csv"):
        group = [row for row in runs if row["method"] == record["method"]]
        fitness = [float(row["fitness"]) for row in group]
        expected = {"best_fitness": min(fitness), "mean_fitness": statistics.fmean(fitness),
                    "median_fitness": statistics.median(fitness), "std_fitness": statistics.stdev(fitness),
                    "worst_fitness": max(fitness),
                    "mean_seconds": statistics.fmean(float(row["seconds"]) for row in group),
                    "mean_evaluations": statistics.fmean(int(row["evaluations"]) for row in group),
                    "optimal_rate": sum(row["feasible"] == "True" and int(row["utility"]) == upper_bound
                                        for row in group) / len(group)}
        for key, value in expected.items():
            require(math.isclose(float(record[key]), value, abs_tol=1e-9),
                    f"ЛР2: неверная сводная метрика {key}")

    examples = rows(results / "feasibility_examples.csv")
    require([row["feasible"] for row in examples].count("True") == 2,
            "ЛР2: нужны два допустимых примера")
    require([row["feasible"] for row in examples].count("False") == 2,
            "ЛР2: нужны два недопустимых примера")
    ET.parse(results / "convergence.svg")


def verify_dispatch(lab: int) -> None:
    import dispatch_core as core
    folder = "lab_3_dispatch_gp" if lab == 3 else "lab_4_robust_dispatch"
    directory = ROOT / folder
    results = directory / "results"
    config = json.loads((results / "config.json").read_text(encoding="utf-8"))
    require((config["group"], config["list_number"], config["variant"]) == (517, 18, 20),
            f"ЛР{lab}: неверный вариант")
    dataset = core.load_dataset(directory / "data/scenarios.json")
    generated = core.make_dataset(config["task_count"], config["uncertainty"], config["data_seed"])
    require(dataset == generated, f"ЛР{lab}: данные не совпали с генератором и seed")
    require({k: len(v) for k,v in dataset.items()} == {"train": 4, "validation": 4, "test": 10},
            f"ЛР{lab}: неверное разделение данных")
    split_seeds = [{s.seed for s in split} for split in dataset.values()]
    require(all(not a & b for i,a in enumerate(split_seeds) for b in split_seeds[i+1:]),
            f"ЛР{lab}: пересечение seed сценариев")
    runs = rows(results / "runs.csv")
    expected = 20 if lab == 3 else 30
    methods = {m["name"]: m for m in config["methods"]}
    require(Counter(row["method"] for row in runs) == {name: expected for name in methods},
            f"ЛР{lab}: требуется {expected} запусков каждого метода")
    models = json.loads((results / "models.json").read_text(encoding="utf-8"))
    require(len(models) == len(runs), f"ЛР{lab}: неполный набор моделей")
    for model, row in zip(models, runs):
        require((model["method"], model["seed"]) == (row["method"], int(row["seed"])),
                f"ЛР{lab}: модель и результат не соответствуют друг другу")
        tree = core.freeze(model["tree"])
        core.validate(tree, methods[row["method"]]["depth"])
        require(core.size(tree) == int(row["size"]), f"ЛР{lab}: неверная сложность")
        require(int(row["evaluations"]) == config["budget"], f"ЛР{lab}: нарушен бюджет")
        require(int(row["training_simulations"]) == config["budget"]*4,
                f"ЛР{lab}: неверное число обучающих симуляций")
        recalculated = core.assess(tree, dataset["test"], config["risk"])
        for metric, column in (("quality", "test_quality"), ("loss", "test_loss"),
                               ("makespan", "test_makespan"), ("cost", "test_cost"),
                               ("mean_tardiness", "test_tardiness"),
                               ("late_fraction", "test_late_fraction")):
            require(math.isclose(recalculated[metric], float(row[column]), abs_tol=1e-9),
                    f"ЛР{lab}: тестовая метрика {metric} не воспроизведена")
        require(math.isclose(core.assess(tree, dataset["train"], config["risk"])["quality"],
                             float(row["train_quality"]), abs_tol=1e-9),
                f"ЛР{lab}: обучающая оценка не воспроизведена")
        require(math.isclose(core.assess(tree, dataset["validation"], config["risk"])["quality"],
                             model["validation_quality"], abs_tol=1e-9),
                f"ЛР{lab}: validation не воспроизведена")
    for item in rows(results / "summary.csv"):
        values = [float(row["test_quality"]) for row in runs if row["method"] == item["method"]]
        for name, value in {"best": min(values), "mean": statistics.fmean(values),
                            "median": statistics.median(values), "std": statistics.stdev(values),
                            "worst": max(values)}.items():
            require(math.isclose(float(item[name]), value, abs_tol=1e-9),
                    f"ЛР{lab}: неверная статистика {name}")
    representatives = json.loads((results / "representatives.json").read_text(encoding="utf-8"))
    require(len(representatives) == 3 and len({r["size"] for r in representatives}) == 3,
            f"ЛР{lab}: нужны три эволюционировавшие программы разных сложностей")
    for item in representatives:
        tree = core.freeze(item["tree"])
        core.validate(tree)
        require(core.size(tree) == item["size"], f"ЛР{lab}: неверный размер представителя")
        require(math.isclose(core.assess(tree, dataset["test"], config["risk"])["quality"],
                             item["test_quality"], abs_tol=1e-9),
                f"ЛР{lab}: представитель не воспроизведён")
    for item in rows(results / "baselines.csv"):
        result = core.assess(item["tree"], dataset["test"], config["risk"])
        for metric, value in result.items():
            require(math.isclose(float(item[metric]), value, abs_tol=1e-9),
                    f"ЛР{lab}: baseline {item['method']} не воспроизведён")
    selected = json.loads((results / "selected_model.json").read_text(encoding="utf-8"))
    chosen = min(models, key=lambda m: m["validation_quality"]+config["penalty"]*m["size"])
    require(selected == chosen, f"ЛР{lab}: модель выбрана не по validation")
    schedule = core.simulate(core.freeze(selected["tree"]), dataset["test"][0], True)["schedule"]
    if lab == 3:
        decoded = rows(results / "decoding_example.csv")
        selected_rows = [row for row in decoded if row["selected"] == "True"]
        require(len(selected_rows) == 1 and int(selected_rows[0]["agent"]) == schedule[0]["agent"],
                "ЛР3: пример декодирования не соответствует расписанию")
        require(float(selected_rows[0]["priority"]) == min(float(row["priority"]) for row in decoded),
                "ЛР3: в примере выбран не минимальный приоритет")
    stored = rows(results / "schedule.csv")
    require(len(stored) == config["task_count"], f"ЛР{lab}: неполное расписание")
    for actual, saved in zip(schedule, stored):
        for key, value in actual.items():
            require(math.isclose(value, float(saved[key]), abs_tol=1e-9),
                    f"ЛР{lab}: расписание не воспроизведено")
    for graphic in results.glob("*.svg"):
        ET.parse(graphic)
    histories = rows(results / "convergence.csv")
    require(len(histories) == len(methods)*config["budget"], f"ЛР{lab}: неполные траектории")
    for method in methods:
        series = [r for r in histories if r["method"] == method]
        require(all(float(r["minimum"]) <= float(r["mean"]) <= float(r["maximum"])
                    for r in series), f"ЛР{lab}: неверные диапазоны сходимости")
        require(all(float(b["mean"]) <= float(a["mean"])+1e-10
                    for a,b in zip(series, series[1:])), f"ЛР{lab}: нарушен элитизм")


def main() -> None:
    verify_links()
    verify_lab1()
    verify_lab2()
    verify_dispatch(3)
    verify_dispatch(4)
    print("OK: ЛР1–ЛР4, вариант 20 — программные проверки пройдены")
    print("ЛР4: утверждение темы преподавателем и устная защита остаются внешними условиями")


if __name__ == "__main__":
    main()
