"""ЛР2, вариант 6: генетический алгоритм для месячного плана закупок."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DATA_SEED = 6_160_026
ITEM_COUNT = 30


@dataclass(frozen=True)
class Item:
    number: int
    name: str
    unit_cost: int
    unit_utility: int
    minimum: int
    maximum: int


@dataclass(frozen=True)
class Config:
    name: str
    constraint_mode: str
    crossover: str
    population_size: int = 80
    crossover_probability: float = 0.9
    mutation_probability: float = 0.25
    gene_mutation_probability: float = 0.08


@dataclass(frozen=True)
class Evaluation:
    fitness: float
    utility: int
    cost: int
    feasible: bool
    shortage: int
    over_maximum: int
    overspend: int


@dataclass
class RunResult:
    method: str
    seed: int
    evaluation: Evaluation
    individual: list[int]
    evaluations: int
    seconds: float
    trajectory: list[float]


def generate_instance() -> tuple[list[Item], int]:
    rng = random.Random(DATA_SEED)
    items = []
    for index in range(ITEM_COUNT):
        minimum = rng.randint(1, 4)
        maximum = minimum + rng.randint(5, 14)
        items.append(
            Item(
                number=index + 1,
                name=f"Материал {index + 1:02d}",
                unit_cost=rng.randint(6, 55),
                unit_utility=rng.randint(8, 95),
                minimum=minimum,
                maximum=maximum,
            )
        )
    minimum_cost = sum(item.minimum * item.unit_cost for item in items)
    optional_cost = sum((item.maximum - item.minimum) * item.unit_cost for item in items)
    budget = minimum_cost + int(optional_cost * 0.38)
    return items, budget


def random_individual(items: Sequence[Item], rng: random.Random) -> list[int]:
    return [rng.randint(item.minimum, item.maximum) for item in items]


def evaluate(individual: Sequence[int], items: Sequence[Item], budget: int) -> Evaluation:
    cost = sum(quantity * item.unit_cost for quantity, item in zip(individual, items))
    utility = sum(quantity * item.unit_utility for quantity, item in zip(individual, items))
    shortage = sum(max(0, item.minimum - quantity) for quantity, item in zip(individual, items))
    over_maximum = sum(max(0, quantity - item.maximum) for quantity, item in zip(individual, items))
    overspend = max(0, cost - budget)
    penalty = 3_000 * (shortage + over_maximum) + 50 * overspend
    feasible = shortage == 0 and over_maximum == 0 and overspend == 0
    return Evaluation(-utility + penalty, utility, cost, feasible, shortage, over_maximum, overspend)


def repair(individual: Sequence[int], items: Sequence[Item], budget: int) -> list[int]:
    result = [
        min(item.maximum, max(item.minimum, quantity))
        for quantity, item in zip(individual, items)
    ]
    cost = sum(quantity * item.unit_cost for quantity, item in zip(result, items))
    removable = sorted(
        range(len(items)),
        key=lambda index: items[index].unit_utility / items[index].unit_cost,
    )
    while cost > budget:
        changed = False
        for index in removable:
            if result[index] > items[index].minimum:
                result[index] -= 1
                cost -= items[index].unit_cost
                changed = True
                if cost <= budget:
                    break
        if not changed:
            break
    return result


def uniform_crossover(left: Sequence[int], right: Sequence[int], rng: random.Random) -> list[int]:
    return [a if rng.random() < 0.5 else b for a, b in zip(left, right)]


def one_point_crossover(left: Sequence[int], right: Sequence[int], rng: random.Random) -> list[int]:
    point = rng.randrange(1, len(left))
    return [*left[:point], *right[point:]]


def mutate(individual: list[int], items: Sequence[Item], probability: float, rng: random.Random) -> None:
    for index, item in enumerate(items):
        if rng.random() < probability:
            individual[index] += rng.choice((-2, -1, 1, 2))
            individual[index] = min(item.maximum + 2, max(0, individual[index]))


def tournament(
    population: Sequence[list[int]], evaluations: Sequence[Evaluation], rng: random.Random
) -> list[int]:
    indexes = rng.sample(range(len(population)), 3)
    return population[min(indexes, key=lambda index: evaluations[index].fitness)]


def genetic_algorithm(
    config: Config,
    items: Sequence[Item],
    budget: int,
    generations: int,
    seed: int,
) -> RunResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    population = [random_individual(items, rng) for _ in range(config.population_size)]
    if config.constraint_mode == "repair":
        population = [repair(individual, items, budget) for individual in population]
    evaluations = [evaluate(individual, items, budget) for individual in population]
    evaluation_count = len(population)
    trajectory = [min(result.fitness for result in evaluations)]

    for _ in range(generations):
        elite_index = min(range(len(population)), key=lambda index: evaluations[index].fitness)
        next_population = [population[elite_index][:]]
        while len(next_population) < config.population_size:
            parent_a = tournament(population, evaluations, rng)
            parent_b = tournament(population, evaluations, rng)
            if rng.random() < config.crossover_probability:
                child = (
                    uniform_crossover(parent_a, parent_b, rng)
                    if config.crossover == "uniform"
                    else one_point_crossover(parent_a, parent_b, rng)
                )
            else:
                child = parent_a[:]
            if rng.random() < config.mutation_probability:
                mutate(child, items, config.gene_mutation_probability, rng)
            if config.constraint_mode == "repair":
                child = repair(child, items, budget)
            next_population.append(child)
        population = next_population
        evaluations = [evaluate(individual, items, budget) for individual in population]
        evaluation_count += len(population) - 1  # элита уже была оценена
        trajectory.append(min(result.fitness for result in evaluations))

    best_index = min(range(len(population)), key=lambda index: evaluations[index].fitness)
    return RunResult(
        config.name,
        seed,
        evaluations[best_index],
        population[best_index],
        evaluation_count,
        time.perf_counter() - started,
        trajectory,
    )


def random_feasible_search(
    items: Sequence[Item], budget: int, evaluations_limit: int, seed: int
) -> RunResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    best_individual: list[int] = []
    best_evaluation: Evaluation | None = None
    for _ in range(evaluations_limit):
        individual = repair(random_individual(items, rng), items, budget)
        result = evaluate(individual, items, budget)
        if best_evaluation is None or result.fitness < best_evaluation.fitness:
            best_individual, best_evaluation = individual, result
    assert best_evaluation is not None
    return RunResult(
        "random_feasible_search",
        seed,
        best_evaluation,
        best_individual,
        evaluations_limit,
        time.perf_counter() - started,
        [best_evaluation.fitness],
    )


def greedy_baseline(items: Sequence[Item], budget: int) -> tuple[list[int], Evaluation]:
    plan = [item.minimum for item in items]
    remaining = budget - sum(quantity * item.unit_cost for quantity, item in zip(plan, items))
    order = sorted(
        range(len(items)),
        key=lambda index: items[index].unit_utility / items[index].unit_cost,
        reverse=True,
    )
    for index in order:
        amount = min(items[index].maximum - plan[index], remaining // items[index].unit_cost)
        plan[index] += amount
        remaining -= amount * items[index].unit_cost
    return plan, evaluate(plan, items, budget)


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def save_instance(data_dir: Path, items: Sequence[Item], budget: int) -> None:
    write_csv(
        data_dir / "materials.csv",
        ["number", "name", "unit_cost", "unit_utility", "minimum", "maximum"],
        [[item.number, item.name, item.unit_cost, item.unit_utility, item.minimum, item.maximum] for item in items],
    )
    (data_dir / "config.json").write_text(
        json.dumps({"seed": DATA_SEED, "budget": budget, "item_count": len(items)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_svg(path: Path, series: dict[str, Sequence[float]]) -> None:
    width, height = 960, 560
    left, right, top, bottom = 90, 25, 55, 80
    all_values = [value for values in series.values() for value in values]
    y_min, y_max = min(all_values), max(all_values)
    padding = max(1.0, 0.05 * (y_max - y_min))
    y_min, y_max = y_min - padding, y_max + padding
    length = max(len(values) for values in series.values())

    def x(index: int) -> float:
        return left + index / max(1, length - 1) * (width - left - right)

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    colors = ("#1565c0", "#ef6c00", "#2e7d32")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">Средняя сходимость: план закупок</text>',
    ]
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.2f}" text-anchor="end" font-family="monospace" font-size="12">{value:.5g}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>')
    for index, (name, values) in enumerate(series.items()):
        points = " ".join(f"{x(i):.2f},{y(value):.2f}" for i, value in enumerate(values))
        color = colors[index % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        legend_x = left + index * 260
        parts.append(f'<line x1="{legend_x}" y1="{height-48}" x2="{legend_x+28}" y2="{height-48}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+34}" y="{height-43}" font-family="sans-serif" font-size="12">{name}</text>')
    parts.append('<text x="480" y="540" text-anchor="middle" font-family="sans-serif" font-size="14">Поколение</text>')
    parts.append('<text x="20" y="280" transform="rotate(-90 20 280)" text-anchor="middle" font-family="sans-serif" font-size="14">Фитнес (меньше лучше)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def run_experiment(runs: int, generations: int, output: Path, data_dir: Path) -> None:
    if runs < 20:
        raise ValueError("По ТЗ требуется не менее 20 независимых запусков")
    items, budget = generate_instance()
    save_instance(data_dir, items, budget)
    output.mkdir(parents=True, exist_ok=True)
    configs = (
        Config("repair_uniform", "repair", "uniform"),
        Config("penalty_uniform", "penalty", "uniform"),
        Config("repair_one_point", "repair", "one_point"),
    )
    by_method: dict[str, list[RunResult]] = {}
    for config_index, config in enumerate(configs):
        by_method[config.name] = [
            genetic_algorithm(config, items, budget, generations, 600_000 + config_index * 10_000 + run)
            for run in range(runs)
        ]
    equal_budget = configs[0].population_size + generations * (configs[0].population_size - 1)
    by_method["random_feasible_search"] = [
        random_feasible_search(items, budget, equal_budget, 900_000 + run)
        for run in range(runs)
    ]

    greedy_plan, greedy_evaluation = greedy_baseline(items, budget)
    run_rows = []
    for method, results in by_method.items():
        for result in results:
            ev = result.evaluation
            run_rows.append([method, result.seed, ev.fitness, ev.utility, ev.cost, ev.feasible, ev.shortage, ev.over_maximum, ev.overspend, result.evaluations, result.seconds, json.dumps(result.individual)])
    write_csv(
        output / "runs.csv",
        ["method", "seed", "fitness", "utility", "cost", "feasible", "shortage", "over_maximum", "overspend", "evaluations", "seconds", "individual"],
        run_rows,
    )

    summary_rows = []
    for method, results in by_method.items():
        fitness = [result.evaluation.fitness for result in results]
        utilities = [result.evaluation.utility for result in results if result.evaluation.feasible]
        summary_rows.append([
            method,
            min(fitness), statistics.fmean(fitness), statistics.median(fitness), statistics.stdev(fitness), max(fitness),
            max(utilities) if utilities else "", statistics.fmean(utilities) if utilities else "",
            sum(result.evaluation.feasible for result in results) / len(results),
        ])
    summary_rows.append(["greedy_constructive", greedy_evaluation.fitness, greedy_evaluation.fitness, greedy_evaluation.fitness, 0.0, greedy_evaluation.fitness, greedy_evaluation.utility, greedy_evaluation.utility, 1.0])
    write_csv(output / "summary.csv", ["method", "best_fitness", "mean_fitness", "median_fitness", "std_fitness", "worst_fitness", "best_utility", "mean_utility", "feasible_rate"], summary_rows)

    ga_results = [result for config in configs for result in by_method[config.name]]
    feasible_results = [result for result in ga_results if result.evaluation.feasible]
    best = min(feasible_results or ga_results, key=lambda result: result.evaluation.fitness)
    write_csv(
        output / "best_plan.csv",
        ["material", "quantity", "minimum", "maximum", "unit_cost", "unit_utility", "line_cost", "line_utility"],
        [[item.name, quantity, item.minimum, item.maximum, item.unit_cost, item.unit_utility, quantity * item.unit_cost, quantity * item.unit_utility] for quantity, item in zip(best.individual, items)],
    )

    valid_a = greedy_plan
    valid_b = repair([item.maximum for item in items], items, budget)
    invalid_a = [0] * len(items)
    invalid_b = [item.maximum for item in items]
    examples = [
        ("допустимый: жадный план", valid_a),
        ("допустимый: восстановленный максимум", valid_b),
        ("недопустимый: отсутствуют минимальные запасы", invalid_a),
        ("недопустимый: превышен бюджет", invalid_b),
    ]
    write_csv(
        output / "feasibility_examples.csv",
        ["description", "feasible", "fitness", "utility", "cost", "shortage", "over_maximum", "overspend", "individual"],
        [[description, (ev := evaluate(individual, items, budget)).feasible, ev.fitness, ev.utility, ev.cost, ev.shortage, ev.over_maximum, ev.overspend, json.dumps(individual)] for description, individual in examples],
    )

    chart = {}
    for config in configs:
        results = by_method[config.name]
        length = min(len(result.trajectory) for result in results)
        chart[config.name] = [statistics.fmean(result.trajectory[index] for result in results) for index in range(length)]
    write_svg(output / "convergence.svg", chart)

    table = "\n".join(
        f"| {row[0]} | {float(row[1]):.8g} | {float(row[2]):.8g} | {float(row[3]):.8g} | {float(row[4]):.8g} | {float(row[5]):.8g} | {row[6]} | {float(row[8]):.0%} |"
        for row in summary_rows
    )
    report = f"""# Отчёт по лабораторной работе №2

## Вариант и постановка

Вариант 6, группа 517, номер в списке 16: месячный план закупки расходных материалов при лимите бюджета и минимальных запасах. Сгенерировано {len(items)} материалов с фиксированным `seed={DATA_SEED}`. Бюджет — {budget} условных единиц.

## Представление и ограничения

Особь — целочисленный вектор `x=(x₁,…,x₃₀)`, где `xᵢ` — закупаемое количество материала. Ограничения: `minimumᵢ ≤ xᵢ ≤ maximumᵢ` и `Σ costᵢ·xᵢ ≤ {budget}`. Цель — максимизировать `Σ utilityᵢ·xᵢ`; в минимизируемом фитнесе используется отрицательная полезность и штраф за нарушения.

## Алгоритм

Популяция — 80, поколений — {generations}, турнирная селекция размера 3, вероятность кроссовера — 0.9, вероятность применения мутации — 0.25, элитизм — одна особь. Сравниваются:

1. ремонт + равномерный кроссовер;
2. штрафы + равномерный кроссовер — меняется только работа с ограничениями;
3. ремонт + одноточечный кроссовер — меняется только кроссовер.

```mermaid
flowchart TD
    A[Генерация целочисленной популяции] --> B[Ремонт или штраф]
    B --> C[Оценка стоимости и полезности]
    C --> D[Турнирная селекция]
    D --> E[Кроссовер]
    E --> F[Целочисленная мутация]
    F --> G[Новая популяция с элитой]
    G --> H{{180 поколений?}}
    H -- нет --> B
    H -- да --> I[Лучший допустимый план]
```

## Обязательные проверки

В [feasibility_examples.csv](feasibility_examples.csv) приведены два допустимых и два недопустимых решения. Случайный допустимый поиск использует тот же бюджет — {equal_budget} оценок. Жадная конструктивная эвристика приведена как дополнительная понятная базовая линия.

## Результаты {runs} независимых запусков

Фитнес минимизируется; для допустимых решений он равен отрицательной полезности.

| Метод | Лучший фитнес | Средний | Медиана | Ст. отклонение | Худший | Лучшая полезность | Допустимые запуски |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

Лучший план: полезность `{best.evaluation.utility}`, стоимость `{best.evaluation.cost}` из бюджета `{budget}`. Подробности находятся в [best_plan.csv](best_plan.csv).

## Вывод

Целочисленное кодирование непосредственно соответствует количествам материалов. Ремонт быстро возвращает решение в допустимую область, а штрафной подход допускает исследование недопустимых промежуточных решений. Контролируемая замена кроссовера показывает влияние оператора без одновременной смены метода ограничений. Серия запусков позволяет судить об устойчивости результатов.

Файлы: [runs.csv](runs.csv), [summary.csv](summary.csv), [convergence.svg](convergence.svg), [best_plan.csv](best_plan.csv).
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--generations", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.runs, args.generations, args.output, args.data)
    print(f"Готово: {args.output.resolve()}")
