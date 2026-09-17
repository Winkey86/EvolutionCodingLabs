"""Лабораторная №1: вещественный генетический алгоритм.

Для номера в списке 16 поддерживаются все три возможные группы:
507 -> V19 HappyCat, 517 -> V6 Schwefel, 527 -> V13 Salomon.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Sequence

from common import ensure_dir, line_chart_svg, summary, write_csv
from variant import calculate_variant


Vector = list[float]


@dataclass(frozen=True)
class Problem:
    variant: int
    name: str
    dimension: int
    lower: float
    upper: float
    optimum_value: float
    optimum_point: float
    objective: Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class GAConfig:
    name: str
    population: int = 60
    evaluations: int = 12_000
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    mutation_scale: float = 0.05
    tournament_size: int = 3
    blx_alpha: float = 0.35


@dataclass
class RunResult:
    seed: int
    best_value: float
    best_vector: Vector
    trajectory: list[float]
    evaluations: int
    elapsed_seconds: float


def happy_cat(x: Sequence[float]) -> float:
    d = len(x)
    sum_sq = sum(value * value for value in x)
    return abs(sum_sq - d) ** 0.25 + (0.5 * sum_sq + sum(x)) / d + 0.5


def schwefel(x: Sequence[float]) -> float:
    return 418.9829 * len(x) - sum(value * math.sin(math.sqrt(abs(value))) for value in x)


def salomon(x: Sequence[float]) -> float:
    radius = math.sqrt(sum(value * value for value in x))
    return 1.0 - math.cos(2.0 * math.pi * radius) + 0.1 * radius


def make_problem(group: int) -> Problem:
    info = calculate_variant(16, group)
    if info.number == 19:
        return Problem(19, "HappyCat (счастливая кошка)", info.dimension, -20.0, 20.0, 0.0, -1.0, happy_cat)
    if info.number == 6:
        return Problem(6, "Schwefel (Швефеля)", info.dimension, -500.0, 500.0, 0.0, 420.968746, schwefel)
    if info.number == 13:
        return Problem(13, "Salomon (Саломона)", info.dimension, -100.0, 100.0, 0.0, 0.0, salomon)
    raise AssertionError(f"Для номера 16 не ожидался вариант {info.number}")


def tournament(population: Sequence[Vector], scores: Sequence[float], size: int, rng: random.Random) -> Vector:
    candidates = rng.sample(range(len(population)), size)
    winner = min(candidates, key=scores.__getitem__)
    return population[winner]


def blx_alpha(parent_a: Sequence[float], parent_b: Sequence[float], alpha: float, problem: Problem, rng: random.Random) -> Vector:
    child: Vector = []
    for left, right in zip(parent_a, parent_b):
        low, high = sorted((left, right))
        span = high - low
        value = rng.uniform(low - alpha * span, high + alpha * span)
        child.append(min(problem.upper, max(problem.lower, value)))
    return child


def mutate(vector: Vector, config: GAConfig, problem: Problem, rng: random.Random) -> None:
    sigma = config.mutation_scale * (problem.upper - problem.lower)
    for index in range(len(vector)):
        if rng.random() < config.mutation_probability:
            vector[index] += rng.gauss(0.0, sigma)
            vector[index] = min(problem.upper, max(problem.lower, vector[index]))


def run_ga(problem: Problem, config: GAConfig, seed: int) -> RunResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    population = [
        [rng.uniform(problem.lower, problem.upper) for _ in range(problem.dimension)]
        for _ in range(config.population)
    ]
    scores = [problem.objective(individual) for individual in population]
    evaluations = len(population)
    best_index = min(range(len(population)), key=scores.__getitem__)
    best_vector = population[best_index][:]
    best_value = scores[best_index]
    trajectory = [best_value]

    while evaluations < config.evaluations:
        order = sorted(range(len(population)), key=scores.__getitem__)
        elite_score = scores[order[0]]
        next_population = [population[order[0]][:]]  # элитизм: одна особь
        while len(next_population) < config.population and evaluations + len(next_population) - 1 < config.evaluations:
            parent_a = tournament(population, scores, config.tournament_size, rng)
            parent_b = tournament(population, scores, config.tournament_size, rng)
            if rng.random() < config.crossover_probability:
                child = blx_alpha(parent_a, parent_b, config.blx_alpha, problem, rng)
            else:
                child = parent_a[:]
            mutate(child, config, problem, rng)
            next_population.append(child)

        next_scores = [problem.objective(individual) for individual in next_population[1:]]
        evaluations += len(next_scores)
        population = next_population
        scores = [elite_score, *next_scores]
        current_index = min(range(len(population)), key=scores.__getitem__)
        if scores[current_index] < best_value:
            best_value = scores[current_index]
            best_vector = population[current_index][:]
        trajectory.append(best_value)

    return RunResult(seed, best_value, best_vector, trajectory, evaluations, time.perf_counter() - started)


def random_search(problem: Problem, evaluations: int, seed: int) -> RunResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    best_vector: Vector = []
    best_value = math.inf
    trajectory: list[float] = []
    checkpoint = 60
    for index in range(evaluations):
        vector = [rng.uniform(problem.lower, problem.upper) for _ in range(problem.dimension)]
        value = problem.objective(vector)
        if value < best_value:
            best_value, best_vector = value, vector
        if index % checkpoint == 0:
            trajectory.append(best_value)
    trajectory.append(best_value)
    return RunResult(seed, best_value, best_vector, trajectory, evaluations, time.perf_counter() - started)


def align_trajectories(results: Sequence[RunResult]) -> tuple[list[float], list[float], list[float]]:
    length = min(len(item.trajectory) for item in results)
    columns = [[item.trajectory[index] for item in results] for index in range(length)]
    return (
        [min(column) for column in columns],
        [fmean(column) for column in columns],
        [max(column) for column in columns],
    )


def experiment(group: int, runs: int, output_root: Path) -> Path:
    if runs < 20:
        raise ValueError("По заданию требуется не менее 20 независимых запусков")
    problem = make_problem(group)
    output = ensure_dir(output_root / f"group_{group}_v{problem.variant}")
    configs = [
        GAConfig("GA_sigma_5pct", mutation_scale=0.05),
        GAConfig("GA_sigma_15pct", mutation_scale=0.15),
    ]
    all_results: dict[str, list[RunResult]] = {}
    for config_index, config in enumerate(configs):
        all_results[config.name] = [
            run_ga(problem, config, 10_000 * problem.variant + config_index * 1_000 + run)
            for run in range(runs)
        ]
    all_results["random_search"] = [
        random_search(problem, configs[0].evaluations, 90_000 + problem.variant * 100 + run)
        for run in range(runs)
    ]

    rows = []
    for method, results in all_results.items():
        for item in results:
            rows.append(
                [method, item.seed, item.best_value, item.evaluations, item.elapsed_seconds, " ".join(f"{x:.10g}" for x in item.best_vector)]
            )
    write_csv(output / "runs.csv", ["method", "seed", "best_value", "evaluations", "elapsed_seconds", "best_vector"], rows)

    stats_rows = []
    for method, results in all_results.items():
        stats = summary([item.best_value for item in results])
        stats_rows.append([method, *(stats[key] for key in ("best", "mean", "median", "std", "worst"))])
    write_csv(output / "summary.csv", ["method", "best", "mean", "median", "std", "worst"], stats_rows)

    best_curve, mean_curve, worst_curve = align_trajectories(all_results[configs[0].name])
    write_csv(output / "convergence.csv", ["generation", "minimum", "mean", "maximum"], zip(range(len(mean_curve)), best_curve, mean_curve, worst_curve))
    line_chart_svg(
        output / "convergence.svg",
        {"минимум": best_curve, "среднее": mean_curve, "максимум": worst_curve},
        title=f"ЛР1, группа {group}, V={problem.variant}: сходимость {configs[0].name}",
        x_label="Поколение",
        y_label="Лучшее f(x)",
    )

    best_method, best_run = min(
        ((method, result) for method, results in all_results.items() if method != "random_search" for result in results),
        key=lambda pair: pair[1].best_value,
    )
    stats_text = "\n".join(
        f"| {row[0]} | " + " | ".join(f"{float(value):.8g}" for value in row[1:]) + " |"
        for row in stats_rows
    )
    report = f"""# Лабораторная работа №1

## Вариант

- Номер в списке: 16.
- Группа: {group}.
- Формула: `V = 1 + ((17·16 + 7·Q + 26) mod 20)`.
- Вариант: **V={problem.variant}**, функция **{problem.name}**.
- Размерность: `d = 5 + (V mod 6) = {problem.dimension}`.
- Область поиска: `[{problem.lower}, {problem.upper}]^{problem.dimension}`.
- Известный глобальный минимум: `f(x*) = {problem.optimum_value}` при `x_i = {problem.optimum_point}`.

## Алгоритм

Особь — вещественный вектор длины {problem.dimension}. Использованы турнирная селекция (размер 3), BLX-α-кроссовер (`α=0.35`), гауссовская мутация, отсечение координат по границам и элитизм одной особи. Бюджет каждой попытки — {configs[0].evaluations} вычислений функции, популяция — {configs[0].population}.

Сравниваются две конфигурации, отличающиеся только масштабом мутации: 5% и 15% ширины области. Базовый метод — равномерный случайный поиск при том же бюджете.

## Результаты {runs} независимых запусков

| Метод | Лучшее | Среднее | Медиана | Ст. отклонение | Худшее |
|---|---:|---:|---:|---:|---:|
{stats_text}

Лучший результат: `{best_run.best_value:.12g}` методом `{best_method}` при векторе `{[round(x, 8) for x in best_run.best_vector]}`.

## Вывод

Функция {problem.name} нетривиальна для локального/случайного поиска: её ландшафт имеет выраженную нелинейность и/или множество локальных структур, а объём пространства экспоненциально растёт с размерностью. Генетический алгоритм использует накопленную популяцией информацию; сравнение с `random_search` показывает, насколько это полезно при одинаковом числе обращений к функции. Различие между масштабами мутации демонстрирует компромисс между локальным уточнением и выходом из областей притяжения.

График: [convergence.svg](convergence.svg). Полные данные: [runs.csv](runs.csv), [summary.csv](summary.csv), [convergence.csv](convergence.csv).
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, choices=(507, 517, 527), required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/lab1"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_path = experiment(args.group, args.runs, args.output)
    print(f"Готово: {result_path}")
