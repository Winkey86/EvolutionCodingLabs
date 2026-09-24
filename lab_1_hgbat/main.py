"""ЛР1, вариант 20: минимизация функции HGBat генетическим алгоритмом."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DIMENSION = 5 + 20 % 6  # d = 7
LOWER_BOUND = -15.0
UPPER_BOUND = 15.0
KNOWN_OPTIMUM_X = -1.0


@dataclass(frozen=True)
class Config:
    name: str
    mutation_scale: float
    population_size: int = 60
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    tournament_size: int = 3
    blx_alpha: float = 0.35


@dataclass
class RunResult:
    method: str
    seed: int
    value: float
    vector: list[float]
    evaluations: int
    seconds: float
    trajectory: list[float]


def hgbat(vector: Sequence[float]) -> float:
    """Целевая функция HGBat варианта 20 (минимизация)."""
    sum_squares = sum(value * value for value in vector)
    sum_values = sum(vector)
    return math.sqrt(abs(sum_squares**2 - sum_values**2)) + (
        0.5 * sum_squares + sum_values
    ) / len(vector) + 0.5


def tournament(
    population: Sequence[list[float]],
    scores: Sequence[float],
    size: int,
    rng: random.Random,
) -> list[float]:
    indexes = rng.sample(range(len(population)), size)
    return population[min(indexes, key=scores.__getitem__)]


def blx_alpha(
    left: Sequence[float], right: Sequence[float], alpha: float, rng: random.Random
) -> list[float]:
    child = []
    for value_a, value_b in zip(left, right):
        low, high = sorted((value_a, value_b))
        extension = alpha * (high - low)
        value = rng.uniform(low - extension, high + extension)
        child.append(min(UPPER_BOUND, max(LOWER_BOUND, value)))
    return child


def mutate(vector: list[float], config: Config, rng: random.Random) -> None:
    sigma = config.mutation_scale * (UPPER_BOUND - LOWER_BOUND)
    for index in range(len(vector)):
        if rng.random() < config.mutation_probability:
            vector[index] += rng.gauss(0.0, sigma)
            vector[index] = min(UPPER_BOUND, max(LOWER_BOUND, vector[index]))


def genetic_algorithm(config: Config, evaluations_limit: int, seed: int) -> RunResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    population = [
        [rng.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(DIMENSION)]
        for _ in range(config.population_size)
    ]
    scores = [hgbat(individual) for individual in population]
    evaluations = len(population)
    best_index = min(range(len(population)), key=scores.__getitem__)
    best_vector = population[best_index][:]
    best_value = scores[best_index]
    trajectory = [best_value]

    while evaluations < evaluations_limit:
        elite_index = min(range(len(population)), key=scores.__getitem__)
        elite = population[elite_index][:]
        elite_score = scores[elite_index]
        child_count = min(config.population_size - 1, evaluations_limit - evaluations)
        children: list[list[float]] = []
        for _ in range(child_count):
            parent_a = tournament(population, scores, config.tournament_size, rng)
            parent_b = tournament(population, scores, config.tournament_size, rng)
            child = (
                blx_alpha(parent_a, parent_b, config.blx_alpha, rng)
                if rng.random() < config.crossover_probability
                else parent_a[:]
            )
            mutate(child, config, rng)
            children.append(child)

        child_scores = [hgbat(child) for child in children]
        evaluations += len(children)
        population = [elite, *children]
        scores = [elite_score, *child_scores]
        generation_best = min(range(len(population)), key=scores.__getitem__)
        if scores[generation_best] < best_value:
            best_value = scores[generation_best]
            best_vector = population[generation_best][:]
        trajectory.append(best_value)

    return RunResult(
        config.name,
        seed,
        best_value,
        best_vector,
        evaluations,
        time.perf_counter() - started,
        trajectory,
    )


def random_search(evaluations_limit: int, seed: int) -> RunResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    best_value = math.inf
    best_vector: list[float] = []
    for _ in range(evaluations_limit):
        vector = [rng.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(DIMENSION)]
        value = hgbat(vector)
        if value < best_value:
            best_value, best_vector = value, vector
    return RunResult(
        "random_search", seed, best_value, best_vector, evaluations_limit,
        time.perf_counter() - started, [best_value]
    )


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_svg(path: Path, minimum: Sequence[float], mean: Sequence[float], maximum: Sequence[float]) -> None:
    width, height = 960, 560
    left, right, top, bottom = 85, 25, 55, 70
    values = [*minimum, *mean, *maximum]
    y_min, y_max = min(values), max(values)
    padding = max(1e-9, 0.05 * (y_max - y_min))
    y_min, y_max = y_min - padding, y_max + padding
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(index: int) -> float:
        return left + index / max(1, len(mean) - 1) * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">HGBat: GA_sigma_2pct, серия из 20 запусков</text>',
    ]
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.2f}" text-anchor="end" font-family="monospace" font-size="12">{value:.4g}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>')
    for index, (label, series, color) in enumerate((
        ("минимум", minimum, "#2e7d32"),
        ("среднее", mean, "#1565c0"),
        ("максимум", maximum, "#c62828"),
    )):
        points = " ".join(f"{x(i):.2f},{y(value):.2f}" for i, value in enumerate(series))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        legend_x = left + index * 180
        parts.append(f'<line x1="{legend_x}" y1="{height-42}" x2="{legend_x+28}" y2="{height-42}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+34}" y="{height-37}" font-family="sans-serif" font-size="13">{label}</text>')
    parts.append('<text x="480" y="535" text-anchor="middle" font-family="sans-serif" font-size="14">Поколение</text>')
    parts.append('<text x="20" y="280" transform="rotate(-90 20 280)" text-anchor="middle" font-family="sans-serif" font-size="14">Лучшее f(x)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def summarize(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    return min(values), statistics.fmean(values), statistics.median(values), statistics.stdev(values), max(values)


def run_experiment(runs: int, evaluations_limit: int, output: Path) -> None:
    if runs < 20:
        raise ValueError("По ТЗ требуется не менее 20 независимых запусков")
    if evaluations_limit < 60:
        raise ValueError("Бюджет должен быть не меньше размера популяции (60)")
    output.mkdir(parents=True, exist_ok=True)
    configs = (
        Config("GA_sigma_2pct", mutation_scale=0.02),
        Config("GA_sigma_8pct", mutation_scale=0.08),
    )
    write_csv(
        output / "parameters.csv",
        ["method", "population", "evaluation_budget", "crossover_probability",
         "mutation_probability_per_gene", "mutation_sigma_fraction", "selection",
         "crossover", "boundary_handling", "elitism"],
        [[config.name, config.population_size, evaluations_limit,
          config.crossover_probability, config.mutation_probability,
          config.mutation_scale, f"tournament_{config.tournament_size}",
          f"BLX-alpha_{config.blx_alpha}", "clipping", 1]
         for config in configs]
        + [["random_search", "", evaluations_limit, "", "", "", "", "", "", ""]],
    )
    by_method: dict[str, list[RunResult]] = {}
    for config_index, config in enumerate(configs):
        by_method[config.name] = [
            genetic_algorithm(config, evaluations_limit, 60_000 + config_index * 1_000 + run)
            for run in range(runs)
        ]
    by_method["random_search"] = [random_search(evaluations_limit, 90_000 + run) for run in range(runs)]

    run_rows = []
    for method, results in by_method.items():
        for result in results:
            run_rows.append([method, result.seed, result.value, result.evaluations, result.seconds, " ".join(f"{value:.12g}" for value in result.vector)])
    write_csv(output / "runs.csv", ["method", "seed", "best_value", "evaluations", "seconds", "best_vector"], run_rows)

    summary_rows = [[method, *summarize([result.value for result in results])] for method, results in by_method.items()]
    write_csv(output / "summary.csv", ["method", "best", "mean", "median", "std", "worst"], summary_rows)

    trajectories = by_method[configs[0].name]
    length = min(len(result.trajectory) for result in trajectories)
    columns = [[result.trajectory[index] for result in trajectories] for index in range(length)]
    minimum = [min(column) for column in columns]
    mean = [statistics.fmean(column) for column in columns]
    maximum = [max(column) for column in columns]
    write_csv(output / "convergence.csv", ["generation", "minimum", "mean", "maximum"], [[index, minimum[index], mean[index], maximum[index]] for index in range(length)])
    write_svg(output / "convergence.svg", minimum, mean, maximum)

    best = min((result for method, results in by_method.items() if method != "random_search" for result in results), key=lambda result: result.value)
    means = {method: statistics.fmean(result.value for result in results)
             for method, results in by_method.items()}
    mutation_improvement = 100 * (means[configs[0].name] - means[configs[1].name]) / means[configs[0].name]
    random_factor = means["random_search"] / means[configs[1].name]
    table = "\n".join(f"| {row[0]} | " + " | ".join(f"{float(value):.8g}" for value in row[1:]) + " |" for row in summary_rows)
    report = f"""# Отчёт по лабораторной работе №1

## Вариант и постановка

Вариант 20, группа 517, номер в списке 18. Размерность `d=7`. Минимизируется

`f(x) = √|(Σxᵢ²)² − (Σxᵢ)²| + (0.5Σxᵢ² + Σxᵢ)/d + 0.5` при `−15 ≤ xᵢ ≤ 15`.

Известный глобальный минимум расположен в `xᵢ=-1`, `f(x*)=0`.

## Представление и алгоритм

Пространство решений — `D=[−15,15]⁷`. Генотип и фенотип совпадают: вещественный вектор из семи координат, декодирование не требуется. Равномерная инициализация сразу создаёт допустимые решения.

Турнирная селекция не требует масштабирования значений HGBat. BLX-α-кроссовер работает непосредственно с вещественными координатами и исследует промежуток между родителями и его окрестность. Гауссовская мутация обеспечивает локальное непрерывное изменение, отсечение гарантирует границы, элитизм сохраняет лучшее найденное решение.

| Параметр | Значение |
|---|---|
| Популяция | 60 |
| Селекция | турнир 3 |
| Кроссовер | BLX-α, `α=0.35`, вероятность 0.9 |
| Мутация | гауссовская, вероятность каждой координаты 0.2 |
| Масштаб мутации | 2% или 8% ширины области |
| Обработка границ | отсечение до `[−15,15]` |
| Элитизм | 1 особь |
| Критерий остановки | {evaluations_limit} вычислений функции |

```mermaid
flowchart TD
    A[Инициализация популяции] --> B[Вычисление f]
    B --> C[Сохранение элиты]
    C --> D[Турнирная селекция]
    D --> E[BLX-alpha кроссовер]
    E --> F[Гауссовская мутация]
    F --> G[Ограничение координат]
    G --> H[Новая популяция]
    H --> I{{Бюджет исчерпан?}}
    I -- нет --> B
    I -- да --> J[Лучшее решение]
```

## Эксперимент

Выполнено {runs} независимых запусков с различными фиксированными seed. Конфигурации отличаются только масштабом мутации: 2% и 8% ширины области. Случайный поиск получает тот же бюджет вычислений. Полная таблица параметров сохранена в [parameters.csv](parameters.csv). На графике сходимости для `GA_sigma_2pct` показаны минимум, среднее и максимум лучшего значения по всем {runs} запускам на каждом поколении.

| Метод | Лучшее | Среднее | Медиана | Ст. отклонение | Худшее |
|---|---:|---:|---:|---:|---:|
{table}

Лучший результат ГА: `{best.value:.12g}` при `x={ [round(value, 7) for value in best.vector] }`.

## Вывод

Функция HGBat невыпукла и несепарабельна: изменение одной координаты одновременно влияет на обе агрегированные суммы, поэтому независимый покоординатный локальный поиск и случайная выборка не используют структуру хороших решений.

Масштаб 8% дал среднее значение на `{mutation_improvement:.1f}%` лучше масштаба 2%. Его средний результат лучше случайного поиска примерно в `{random_factor:.1f}` раза. Лучшее значение `{best.value:.6g}` имеет абсолютный разрыв `{best.value:.6g}` до известного оптимума 0: это хороший приближённый результат при заданном бюджете, но не доказательство достижения точного минимума. Вывод подтверждается серией запусков, а не единственной удачной траекторией.

Файлы: [parameters.csv](parameters.csv), [runs.csv](runs.csv), [summary.csv](summary.csv), [convergence.csv](convergence.csv), [convergence.svg](convergence.svg).
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--evaluations", type=int, default=12_000)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.runs, args.evaluations, args.output)
    print(f"Готово: {args.output.resolve()}")
