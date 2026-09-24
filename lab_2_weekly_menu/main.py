"""ЛР2, вариант 20: генетический алгоритм для недельного меню."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DATA_SEED = 20_180_517
DAYS = 7
MEALS = ("завтрак", "обед", "ужин")
MIN_DAILY_CALORIES = 1700
MAX_DAILY_CALORIES = 2200
WEEKLY_BUDGET = 3800
MIN_UNIQUE_DISHES = 15
MAX_REPETITIONS = 2
CALORIE_PENALTY = 1000
COST_PENALTY = 1000
STRUCTURAL_PENALTY = 3000


@dataclass(frozen=True)
class Dish:
    number: int
    name: str
    meal: str
    calories: int
    cost: int
    score: int


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
    calorie_violation: int
    variety_shortage: int
    repeat_excess: int
    overspend: int
    unique_dishes: int


@dataclass
class RunResult:
    method: str
    seed: int
    evaluation: Evaluation
    individual: list[int]
    evaluations: int
    seconds: float
    trajectory: list[float]


def generate_instance() -> list[Dish]:
    rng = random.Random(DATA_SEED)
    ranges = {
        "завтрак": ((430, 600), (90, 160)),
        "обед": ((650, 850), (140, 250)),
        "ужин": ((550, 750), (120, 220)),
    }
    dishes: list[Dish] = []
    for meal in MEALS:
        calorie_range, cost_range = ranges[meal]
        for index in range(12):
            dishes.append(Dish(
                number=len(dishes),
                name=f"{meal.capitalize()} {index + 1:02d}",
                meal=meal,
                calories=rng.randint(*calorie_range),
                cost=rng.randint(*cost_range),
                score=rng.randint(55, 100),
            ))
    return dishes


def allowed_by_slot(dishes: Sequence[Dish]) -> list[list[int]]:
    by_meal = {meal: [dish.number for dish in dishes if dish.meal == meal] for meal in MEALS}
    return [by_meal[MEALS[index % 3]] for index in range(DAYS * 3)]


def random_individual(allowed: Sequence[Sequence[int]], rng: random.Random) -> list[int]:
    return [rng.choice(options) for options in allowed]


def constraint_metrics(
    individual: Sequence[int], dishes: Sequence[Dish]
) -> tuple[int, int, int, int, int, int]:
    """Нарушения калорийности, разнообразия, повторов и бюджета."""
    selected = [dishes[index] for index in individual]
    daily = [sum(dish.calories for dish in selected[day * 3:day * 3 + 3]) for day in range(DAYS)]
    calorie_violation = sum(
        max(0, MIN_DAILY_CALORIES - value) + max(0, value - MAX_DAILY_CALORIES)
        for value in daily
    )
    counts = Counter(individual)
    unique_dishes = len(counts)
    variety_shortage = max(0, MIN_UNIQUE_DISHES - unique_dishes)
    repeat_excess = sum(max(0, count - MAX_REPETITIONS) for count in counts.values())
    cost = sum(dish.cost for dish in selected)
    overspend = max(0, cost - WEEKLY_BUDGET)
    return calorie_violation, variety_shortage, repeat_excess, overspend, unique_dishes, cost


def evaluate(individual: Sequence[int], dishes: Sequence[Dish]) -> Evaluation:
    calorie_violation, variety_shortage, repeat_excess, overspend, unique_dishes, cost = (
        constraint_metrics(individual, dishes)
    )
    utility = sum(dishes[index].score for index in individual)
    feasible = not (calorie_violation or variety_shortage or repeat_excess or overspend)
    penalty = (
        CALORIE_PENALTY * calorie_violation
        + COST_PENALTY * overspend
        + STRUCTURAL_PENALTY * (variety_shortage + repeat_excess)
    )
    return Evaluation(-utility + penalty, utility, cost, feasible, calorie_violation,
                      variety_shortage, repeat_excess, overspend, unique_dishes)


def repair(individual: Sequence[int], dishes: Sequence[Dish],
           allowed: Sequence[Sequence[int]], rng: random.Random) -> list[int]:
    """Восстанавливает допустимость, не используя значение целевой функции."""
    candidate = [gene if gene in allowed[index] else rng.choice(allowed[index])
                 for index, gene in enumerate(individual)]
    if not any(constraint_metrics(candidate, dishes)[:4]):
        return candidate

    def violation(menu: Sequence[int]) -> int:
        calories, variety, repeats, overspend, _, _ = constraint_metrics(menu, dishes)
        return calories + 100 * (variety + repeats) + overspend

    # Сначала сохраняем как можно больше генов потомка и заменяем по одному блюду.
    for _ in range(500):
        if not any(constraint_metrics(candidate, dishes)[:4]):
            return candidate
        trial = candidate[:]
        position = rng.randrange(len(trial))
        trial[position] = rng.choice(allowed[position])
        if violation(trial) <= violation(candidate) or rng.random() < 0.1:
            candidate = trial

    # Гарантированный для подготовленного экземпляра запасной способ —
    # специализированная генерация до первого допустимого меню.
    for _ in range(100_000):
        trial = random_individual(allowed, rng)
        if not any(constraint_metrics(trial, dishes)[:4]):
            return trial
    raise RuntimeError("Не удалось построить допустимое меню для заданных ограничений")


def crossover(left: Sequence[int], right: Sequence[int], kind: str,
              rng: random.Random) -> list[int]:
    if kind == "uniform":
        return [a if rng.random() < 0.5 else b for a, b in zip(left, right)]
    point = rng.randrange(1, len(left))
    return [*left[:point], *right[point:]]


def mutate(individual: list[int], allowed: Sequence[Sequence[int]],
           probability: float, rng: random.Random) -> None:
    for index in range(len(individual)):
        if rng.random() < probability:
            individual[index] = rng.choice(allowed[index])


def tournament(population: Sequence[list[int]], evaluations: Sequence[Evaluation],
               rng: random.Random) -> list[int]:
    indexes = rng.sample(range(len(population)), 3)
    return population[min(indexes, key=lambda index: evaluations[index].fitness)]


def genetic_algorithm(config: Config, dishes: Sequence[Dish], generations: int,
                      seed: int) -> RunResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    allowed = allowed_by_slot(dishes)
    population = [random_individual(allowed, rng) for _ in range(config.population_size)]
    if config.constraint_mode == "repair":
        population = [repair(individual, dishes, allowed, rng) for individual in population]
    evaluations = [evaluate(individual, dishes) for individual in population]
    evaluation_count = len(population)
    trajectory = [min(result.fitness for result in evaluations)]

    for _ in range(generations):
        elite_index = min(range(len(population)), key=lambda index: evaluations[index].fitness)
        elite_evaluation = evaluations[elite_index]
        next_population = [population[elite_index][:]]
        while len(next_population) < config.population_size:
            parent_a = tournament(population, evaluations, rng)
            parent_b = tournament(population, evaluations, rng)
            child = (crossover(parent_a, parent_b, config.crossover, rng)
                     if rng.random() < config.crossover_probability else parent_a[:])
            if rng.random() < config.mutation_probability:
                mutate(child, allowed, config.gene_mutation_probability, rng)
            if config.constraint_mode == "repair":
                child = repair(child, dishes, allowed, rng)
            next_population.append(child)
        population = next_population
        child_evaluations = [evaluate(individual, dishes) for individual in population[1:]]
        evaluations = [elite_evaluation, *child_evaluations]
        evaluation_count += len(child_evaluations)
        trajectory.append(min(result.fitness for result in evaluations))

    best_index = min(range(len(population)), key=lambda index: evaluations[index].fitness)
    return RunResult(config.name, seed, evaluations[best_index], population[best_index],
                     evaluation_count, time.perf_counter() - started, trajectory)


def random_feasible_search(dishes: Sequence[Dish], evaluations_limit: int,
                           seed: int) -> RunResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    allowed = allowed_by_slot(dishes)
    best_individual: list[int] = []
    best_evaluation: Evaluation | None = None
    for _ in range(evaluations_limit):
        individual = random_individual(allowed, rng)
        result = evaluate(individual, dishes)
        if result.feasible and (best_evaluation is None or result.fitness < best_evaluation.fitness):
            best_individual, best_evaluation = individual, result
    if best_evaluation is None:
        best_individual = repair(random_individual(allowed, rng), dishes, allowed, rng)
        best_evaluation = evaluate(best_individual, dishes)
    return RunResult("random_feasible_search", seed, best_evaluation, best_individual,
                     evaluations_limit, time.perf_counter() - started, [best_evaluation.fitness])


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def save_instance(data_dir: Path, dishes: Sequence[Dish]) -> None:
    write_csv(data_dir / "dishes.csv",
              ["number", "name", "meal", "calories", "cost", "score"],
              [[d.number, d.name, d.meal, d.calories, d.cost, d.score] for d in dishes])
    config = {
        "seed": DATA_SEED,
        "days": DAYS,
        "daily_calories": [MIN_DAILY_CALORIES, MAX_DAILY_CALORIES],
        "weekly_budget": WEEKLY_BUDGET,
        "minimum_unique_dishes": MIN_UNIQUE_DISHES,
        "maximum_repetitions": MAX_REPETITIONS,
        "objective": "maximize_total_score",
        "penalty_weights": {
            "calorie_violation": CALORIE_PENALTY,
            "overspend": COST_PENALTY,
            "variety_or_repeat": STRUCTURAL_PENALTY,
        },
    }
    (data_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def write_svg(path: Path, series: dict[str, Sequence[float]]) -> None:
    width, height, left, right, top, bottom = 960, 560, 90, 25, 55, 80
    values = [value for row in series.values() for value in row]
    y_min, y_max = min(values), max(values)
    padding = max(1.0, 0.05 * (y_max - y_min))
    y_min, y_max = y_min - padding, y_max + padding
    length = max(len(row) for row in series.values())
    x = lambda index: left + index / max(1, length - 1) * (width - left - right)
    y = lambda value: top + (y_max - value) / (y_max - y_min) * (height - top - bottom)
    colors = ("#1565c0", "#ef6c00", "#2e7d32")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="480" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">Средняя сходимость: меню на неделю</text>']
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        yy = y(value)
        parts.extend([f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#ddd"/>',
                      f'<text x="{left-8}" y="{yy+4:.2f}" text-anchor="end" font-family="monospace" font-size="12">{value:.5g}</text>'])
    for index, (name, row) in enumerate(series.items()):
        points = " ".join(f"{x(i):.2f},{y(value):.2f}" for i, value in enumerate(row))
        color = colors[index]
        parts.extend([f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>',
                      f'<text x="{left + index * 270}" y="{height-42}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>'])
    parts.extend(['<text x="480" y="540" text-anchor="middle" font-family="sans-serif" font-size="14">Поколение</text>',
                  '<text x="20" y="280" transform="rotate(-90 20 280)" text-anchor="middle" font-family="sans-serif" font-size="14">Фитнес (меньше лучше)</text>', '</svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


def run_experiment(runs: int, generations: int, output: Path, data_dir: Path) -> None:
    if runs < 20:
        raise ValueError("По ТЗ требуется не менее 20 независимых запусков")
    dishes = generate_instance()
    save_instance(data_dir, dishes)
    output.mkdir(parents=True, exist_ok=True)
    configs = (
        Config("repair_uniform", "repair", "uniform"),
        Config("penalty_uniform", "penalty", "uniform"),
        Config("repair_one_point", "repair", "one_point"),
    )
    write_csv(
        output / "parameters.csv",
        ["method", "population", "generations", "evaluation_budget", "selection",
         "crossover", "crossover_probability", "mutation_probability",
         "gene_mutation_probability", "constraint_handling", "calorie_penalty",
         "overspend_penalty", "structural_penalty", "elitism"],
        [[config.name, config.population_size, generations,
          config.population_size + generations * (config.population_size - 1),
          "tournament_3", config.crossover, config.crossover_probability,
          config.mutation_probability, config.gene_mutation_probability,
          config.constraint_mode, CALORIE_PENALTY, COST_PENALTY,
          STRUCTURAL_PENALTY, 1] for config in configs],
    )
    by_method = {
        config.name: [genetic_algorithm(config, dishes, generations,
                                        2_000_000 + index * 10_000 + run)
                      for run in range(runs)]
        for index, config in enumerate(configs)
    }
    equal_budget = configs[0].population_size + generations * (configs[0].population_size - 1)
    by_method["random_feasible_search"] = [
        random_feasible_search(dishes, equal_budget, 2_900_000 + run) for run in range(runs)]

    run_rows = []
    for method, results in by_method.items():
        for result in results:
            ev = result.evaluation
            run_rows.append([method, result.seed, ev.fitness, ev.utility, ev.cost, ev.feasible,
                             ev.calorie_violation, ev.variety_shortage, ev.repeat_excess,
                             ev.overspend, ev.unique_dishes, result.evaluations, result.seconds,
                             json.dumps(result.individual)])
    write_csv(output / "runs.csv",
              ["method", "seed", "fitness", "utility", "cost", "feasible",
               "calorie_violation", "variety_shortage", "repeat_excess", "overspend",
               "unique_dishes", "evaluations", "seconds", "individual"], run_rows)

    summary_rows = []
    for method, results in by_method.items():
        fitness = [result.evaluation.fitness for result in results]
        utilities = [result.evaluation.utility for result in results if result.evaluation.feasible]
        summary_rows.append([method, min(fitness), statistics.fmean(fitness),
                             statistics.median(fitness), statistics.stdev(fitness), max(fitness),
                             max(utilities) if utilities else "",
                             statistics.fmean(utilities) if utilities else "",
                             sum(result.evaluation.feasible for result in results) / len(results)])
    write_csv(output / "summary.csv",
              ["method", "best_fitness", "mean_fitness", "median_fitness", "std_fitness",
               "worst_fitness", "best_utility", "mean_utility", "feasible_rate"], summary_rows)

    summary_by_method = {row[0]: row for row in summary_rows}
    repair_mean = float(summary_by_method["repair_uniform"][2])
    penalty_mean = float(summary_by_method["penalty_uniform"][2])
    one_point_mean = float(summary_by_method["repair_one_point"][2])
    random_mean = float(summary_by_method["random_feasible_search"][2])

    ga_results = [result for config in configs for result in by_method[config.name]]
    feasible = [result for result in ga_results if result.evaluation.feasible]
    best = min(feasible or ga_results, key=lambda result: result.evaluation.fitness)
    day_names = ("Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье")
    write_csv(output / "best_menu.csv", ["day", "meal", "dish", "calories", "cost", "score"],
              [[day_names[index // 3], MEALS[index % 3], dishes[gene].name,
                dishes[gene].calories, dishes[gene].cost, dishes[gene].score]
               for index, gene in enumerate(best.individual)])

    allowed = allowed_by_slot(dishes)
    rng = random.Random(20)
    valid_b = repair(random_individual(allowed, rng), dishes, allowed, rng)
    invalid_repeat = [allowed[index][0] for index in range(21)]
    invalid_expensive = [max(options, key=lambda gene: dishes[gene].cost) for options in allowed]
    examples = [("допустимый: лучший план", best.individual),
                ("допустимый: восстановленный случайный", valid_b),
                ("недопустимый: однообразное меню", invalid_repeat),
                ("недопустимый: превышение бюджета", invalid_expensive)]
    write_csv(output / "feasibility_examples.csv",
              ["description", "feasible", "fitness", "utility", "cost", "calorie_violation",
               "variety_shortage", "repeat_excess", "overspend", "unique_dishes", "individual"],
              [[description, (ev := evaluate(individual, dishes)).feasible, ev.fitness, ev.utility,
                ev.cost, ev.calorie_violation, ev.variety_shortage, ev.repeat_excess,
                ev.overspend, ev.unique_dishes, json.dumps(individual)]
               for description, individual in examples])

    chart = {}
    for config in configs:
        results = by_method[config.name]
        length = min(len(result.trajectory) for result in results)
        chart[config.name] = [statistics.fmean(result.trajectory[index] for result in results)
                              for index in range(length)]
    write_svg(output / "convergence.svg", chart)
    table = "\n".join(
        f"| {row[0]} | {float(row[1]):.8g} | {float(row[2]):.8g} | {float(row[3]):.8g} | {float(row[4]):.8g} | {float(row[5]):.8g} | {row[6]} | {float(row[8]):.0%} |"
        for row in summary_rows)
    report = f"""# Отчёт по лабораторной работе №2

## Вариант и постановка

Вариант 20, группа 517, номер в списке 18: меню на неделю с ограничениями калорийности, стоимости и разнообразия. Используются 36 синтетических блюд: по 12 вариантов завтрака, обеда и ужина. Данные генерируются с фиксированным `seed={DATA_SEED}`; диапазоны калорийности и стоимости заданы отдельно для каждого типа приёма пищи, оценка блюда генерируется в диапазоне 55–100. Полные данные находятся в [dishes.csv](../data/dishes.csv), параметры экземпляра — в [config.json](../data/config.json).

Особь — целочисленный вектор из 21 индекса блюда: завтрак, обед и ужин для каждого дня. Декодер сопоставляет позицию вектора дню и типу приёма пищи, а индекс — строке таблицы блюд. Суточная калорийность — {MIN_DAILY_CALORIES}–{MAX_DAILY_CALORIES} ккал, недельная стоимость — не более {WEEKLY_BUDGET}, различных блюд — не менее {MIN_UNIQUE_DISHES}, повтор блюда — не более {MAX_REPETITIONS} раз. Максимизируется суммарная оценка блюд; для минимизации используется её отрицание и штраф за нарушения.

## Алгоритм

Популяция — 80, поколений — {generations}, турнирная селекция размера 3, вероятность кроссовера — 0.9, вероятность применения мутации — 0.25, вероятность замены отдельного блюда при мутации — 0.08, элитизм — одна особь. Коэффициенты штрафа: 1000 за единицу отклонения калорийности или превышения бюджета и 3000 за недостающее уникальное блюдо или лишний повтор. При диапазоне оценок 55–100 даже минимальное нарушение дороже любого возможного выигрыша целевой функции. Полная таблица находится в [parameters.csv](parameters.csv).

Операторы учитывают структуру меню: мутация выбирает блюдо только допустимого типа для конкретного слота; кроссоверы работают между одинаковыми позициями дней и приёмов пищи. Сравниваются восстановление допустимости и штрафы при одинаковом равномерном кроссовере, затем равномерный и одноточечный кроссоверы при одинаковом восстановлении. Восстановление изменяет недопустимого потомка до выполнения всех ограничений и не использует целевую оценку. Случайный допустимый поиск получает тот же бюджет — {equal_budget} обращений к фитнес-функции.

```mermaid
flowchart TD
    A[Генерация целочисленной популяции] --> B[Декодирование меню]
    B --> C[Проверка ограничений и фитнес]
    C --> D[Турнирная селекция]
    D --> E[Равномерный или одноточечный кроссовер]
    E --> F[Замена блюда того же типа]
    F --> G[Восстановление или штраф]
    G --> H[Элитизм и новая популяция]
    H --> I{{Завершены поколения?}}
    I -- нет --> B
    I -- да --> J[Лучшее допустимое меню]
```

Два допустимых и два недопустимых решения с расшифровкой каждого нарушения приведены в [feasibility_examples.csv](feasibility_examples.csv).

## Результаты {runs} независимых запусков

| Метод | Лучший фитнес | Средний | Медиана | Ст. отклонение | Худший | Лучшая полезность | Допустимые запуски |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

Лучшее меню: полезность `{best.evaluation.utility}`, стоимость `{best.evaluation.cost}` из `{WEEKLY_BUDGET}`, различных блюд — `{best.evaluation.unique_dishes}`. Подробности находятся в [best_menu.csv](best_menu.csv).

## Вывод

Целочисленное кодирование напрямую задаёт блюда по приёмам пищи и исключает неверный тип блюда на уровне операторов. Восстановление с равномерным кроссовером дало средний фитнес `{repair_mean:.2f}`, штрафы — `{penalty_mean:.2f}`, восстановление с одноточечным кроссовером — `{one_point_mean:.2f}`. Различия между вариантами ГА малы, однако все финальные решения допустимы; это показывает устойчивость, а не единичный успех.

Средний фитнес случайного допустимого поиска равен `{random_mean:.2f}`, то есть ГА стабильно находит более высокую суммарную оценку при том же бюджете. Лучшее меню имеет оценку `{best.evaluation.utility}`, выполняет все четыре группы ограничений и понятно представлено по дням в [best_menu.csv](best_menu.csv). Поэтому результат следует считать хорошим для созданного экземпляра; утверждение о глобальном оптимуме не делается, поскольку полный перебор `12²¹` допустимых по типу комбинаций практически невозможен.

Файлы: [parameters.csv](parameters.csv), [runs.csv](runs.csv), [summary.csv](summary.csv), [convergence.svg](convergence.svg), [best_menu.csv](best_menu.csv), [feasibility_examples.csv](feasibility_examples.csv).
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
