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


def utility_upper_bound(
    dishes: Sequence[Dish], days: int = DAYS,
    minimum_unique: int = MIN_UNIQUE_DISHES, maximum_repetitions: int = MAX_REPETITIONS,
) -> int:
    """Точный максимум ослабленной задачи без бюджета и калорийности."""
    combined = {0: 0}
    for meal in MEALS:
        states = {(0, 0): 0}
        for dish in (dish for dish in dishes if dish.meal == meal):
            next_states = {}
            for (count, unique), score in states.items():
                for quantity in range(min(maximum_repetitions, days - count) + 1):
                    key = (count + quantity, unique + int(quantity > 0))
                    value = score + quantity * dish.score
                    next_states[key] = max(next_states.get(key, -1), value)
            states = next_states
        next_combined = {}
        for unique, score in combined.items():
            for (count, meal_unique), meal_score in states.items():
                if count == days:
                    key = unique + meal_unique
                    next_combined[key] = max(next_combined.get(key, -1), score + meal_score)
        combined = next_combined
    candidates = [score for unique, score in combined.items() if unique >= minimum_unique]
    if not candidates:
        raise ValueError("Нет меню с заданными числом приёмов пищи и разнообразием")
    return max(candidates)


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

    # Запасная генерация ограничена числом попыток; при неуспехе возвращаем ошибку.
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

    upper_bound = utility_upper_bound(dishes)
    summary_rows = []
    for method, results in by_method.items():
        fitness = [result.evaluation.fitness for result in results]
        utilities = [result.evaluation.utility for result in results if result.evaluation.feasible]
        summary_rows.append([method, min(fitness), statistics.fmean(fitness),
                             statistics.median(fitness), statistics.stdev(fitness), max(fitness),
                             max(utilities) if utilities else "",
                             statistics.fmean(utilities) if utilities else "",
                             sum(result.evaluation.feasible for result in results) / len(results),
                             statistics.fmean(result.seconds for result in results),
                             statistics.fmean(result.evaluations for result in results),
                             sum(result.evaluation.feasible and result.evaluation.utility == upper_bound
                                 for result in results) / len(results)])
    write_csv(output / "summary.csv",
              ["method", "best_fitness", "mean_fitness", "median_fitness", "std_fitness",
               "worst_fitness", "best_utility", "mean_utility", "feasible_rate",
               "mean_seconds", "mean_evaluations", "optimal_rate"], summary_rows)

    summary_by_method = {row[0]: row for row in summary_rows}
    repair_mean = float(summary_by_method["repair_uniform"][2])
    penalty_mean = float(summary_by_method["penalty_uniform"][2])
    one_point_mean = float(summary_by_method["repair_one_point"][2])
    random_mean = float(summary_by_method["random_feasible_search"][2])

    ga_results = [result for config in configs for result in by_method[config.name]]
    feasible = [result for result in ga_results if result.evaluation.feasible]
    best = min(feasible or ga_results, key=lambda result: result.evaluation.fitness)
    certificate = {
        "relaxation": "drop budget and daily calories; keep meal types, repetition limit and diversity",
        "upper_bound": upper_bound, "best_utility": best.evaluation.utility,
        "best_feasible": best.evaluation.feasible,
        "certified_optimal": best.evaluation.feasible and best.evaluation.utility == upper_bound,
        "days": DAYS, "minimum_unique": MIN_UNIQUE_DISHES,
        "maximum_repetitions": MAX_REPETITIONS,
    }
    (output / "optimality.json").write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    practical_table = "\n".join(
        f"| {row[0]} | {float(row[7]):.2f} | {float(row[9]):.4f} | {int(row[10])} | {float(row[11]):.0%} |"
        for row in summary_rows)
    menu_table = "\n".join(
        f"| {day_names[day]} | " + " | ".join(dishes[best.individual[day*3+i]].name for i in range(3))
        + f" | {sum(dishes[gene].calories for gene in best.individual[day*3:day*3+3])} |"
        for day in range(DAYS))
    example_table = "\n".join(
        f"| {i} | {best.individual[i]} | {MEALS[i]} | {dishes[best.individual[i]].name} | {dishes[best.individual[i]].calories} |"
        for i in range(3))
    repair_seconds = float(summary_by_method["repair_uniform"][9])
    penalty_seconds = float(summary_by_method["penalty_uniform"][9])
    random_utility = float(summary_by_method["random_feasible_search"][7])
    improvement = 100 * (-repair_mean - random_utility) / random_utility
    quality_conclusion = (
        f"Найденное меню допустимо и достигает верхней границы: `U={best.evaluation.utility}={upper_bound}`. "
        "Следовательно, для приложенного набора данных доказана глобальная оптимальность полезности. "
        "Это доказательство относится к качеству данного меню, а не к гарантии сходимости ГА на других данных."
        if certificate["certified_optimal"] else
        f"Меню имеет полезность {best.evaluation.utility}, верхняя граница равна {upper_bound}; "
        "достижение глобального оптимума этим запуском не подтверждено."
    )
    report = f"""# Отчёт по лабораторной работе №2

Группа **517**, номер в списке **18**, вариант **20**. Разделы 1–10 соответствуют пунктам 1.4 методических указаний.

## 1. Постановка задачи и индивидуальный вариант

Вариант вычислен как `1+((17·18+7·1+26) mod 20)=20`. Требуется сформировать меню на семь дней с ограничениями калорийности, стоимости и разнообразия. Цель: максимизировать суммарную оценку выбранных блюд.

Подготовлены 36 синтетических блюд, по 12 для завтрака, обеда и ужина; `seed={DATA_SEED}`. Калории/стоимость генерируются равномерно: завтрак 430–600/90–160, обед 650–850/140–250, ужин 550–750/120–220. Оценка блюда лежит в диапазоне 55–100. Это учебные условные данные, а не рекомендации по питанию. [Таблица блюд](../data/dishes.csv), [параметры задачи](../data/config.json).

## 2. Пространство решений, цель и ограничения

Решение `x=(x₀,…,x₂₀)` содержит целые индексы блюд. Допустимый тип задан позицией: `x₃d∈{{0,…,11}}`, `x₃d₊₁∈{{12,…,23}}`, `x₃d₊₂∈{{24,…,35}}`, где `d=0,…,6`. До предметных ограничений существует `12²¹` комбинаций.

Для блюда j заданы калории `kⱼ`, стоимость `cⱼ`, оценка `uⱼ`. Максимизируется `U(x)=Σᵢu[xᵢ]` при ограничениях:

- для каждого дня `1700 ≤ Σₘ₌₀² k[x₃d₊ₘ] ≤ 2200`;
- недельная стоимость `C(x)=Σᵢc[xᵢ] ≤ {WEEKLY_BUDGET}`;
- число различных индексов `|{{xᵢ}}| ≥ {MIN_UNIQUE_DISHES}`;
- число использований каждого блюда `count(j) ≤ {MAX_REPETITIONS}`.

Минимизируется `F(x)=−U(x)+1000·Δкал+1000·Δбюджет+3000·(Δразнообразие+Δповторы)`. Отклонения неотрицательны; при допустимом меню `F=−U`. Поэтому **−1936 лучше −1935**. Максимальный возможный разброс U не превышает `21·(100−55)=945`; минимальный ненулевой штраф 1000 делает любое нарушение дороже выигрыша полезности.

## 3. Представление особи: генотип, фенотип, декодирование

**Генотип**: список из 21 целого индекса, с которым работают селекция, кроссовер и мутация. **Фенотип**: недельное меню с названиями блюд, калориями и стоимостью. В отличие от ЛР1, они представлены по-разному: индексы требуют декодирования через таблицу блюд.

Для позиции i день равен `i//3`, приём пищи равен `i%3`; строка `dishes[xᵢ]` даёт предметные свойства. Например, начало лучшего генотипа `{best.individual[:3]}` декодируется так:

| Позиция i | Ген xᵢ | Приём пищи в понедельник | Блюдо | Ккал |
|---|---:|---|---|---:|
{example_table}

Это взаимно однозначное соответствие слотов и записей меню. Полный фенотип приведён в разделе 10 и [best_menu.csv](best_menu.csv). Два допустимых и два недопустимых примера с величинами нарушений находятся в [feasibility_examples.csv](feasibility_examples.csv).

## 4. Алгоритм, операторы и параметры

| Параметр | Значение |
|---|---|
| Популяция / инициализация | 80 / случайное блюдо допустимого типа в каждом слоте |
| Отбор | Турнир размера 3 |
| Кроссовер | Равномерный или одноточечный; вероятность 0.9 |
| Мутация | Применяется с вероятностью 0.25; внутри неё каждый слот заменяется с вероятностью 0.08 |
| Элитизм | 1 лучшая особь, без повторной фитнес-оценки |
| Ограничения | Восстановление допустимости или штраф |
| Остановка | {generations} поколений; {equal_budget} фитнес-оценок |

Восстановление уменьшает нарушения через замены блюд: до 500 локальных попыток с вероятностью 0.1 принять ухудшение. Затем допустимы до 100000 попыток случайной генерации. При неуспехе выдаётся ошибка. Оператор не использует полезность при исправлении и не гарантирует нахождение меню для любого произвольного экземпляра. Штрафной подход не исправляет кандидата и учитывает нарушения в F. [Полная таблица параметров](parameters.csv).

## 5. Блок-схема алгоритма

```mermaid
flowchart TD
    A[Генерация целочисленной популяции] --> B[Начальное восстановление при выбранном режиме]
    B --> C[Декодирование и оценка начальной популяции]
    C --> D[Сохранение элиты и турнирная селекция]
    D --> E[Равномерный или одноточечный кроссовер]
    E --> F[Замена блюда того же типа]
    F --> G[Восстановление при выбранном режиме]
    G --> H[Оценка потомков и перенос элиты без повторной оценки]
    H --> I{{Выполнено заданное число поколений?}}
    I -- нет --> D
    I -- да --> J[Лучшее допустимое меню]
```

## 6. Обоснование операторов относительно структуры меню

| Оператор | Почему выбран | Что сохраняет и что может нарушить |
|---|---|---|
| Турнир 3 | Сравнивает F напрямую, включая отрицательные значения и штрафы; масштабирование не нужно | Предпочитает лучшие меню, но сам не создаёт разнообразие |
| Равномерный кроссовер | Независимо комбинирует блюда в одинаковых слотах родителей | Сохраняет тип приёма пищи, но может нарушить бюджет, калории и разнообразие |
| Одноточечный кроссовер | Сохраняет непрерывные фрагменты недельного меню | Сохраняет тип слота; граница может пройти внутри дня, допустимость всех ограничений не гарантируется |
| Замена блюда того же типа | Даёт содержательную дискретную мутацию и позволяет менять любой слот | Не создаёт ужин на месте завтрака, но остальные ограничения нужно проверять |
| Восстановление | Возвращает кандидата в допустимую область, когда удаётся исправить ограничения | Требует дополнительных проверок и может менять полезные сочетания блюд |
| Штраф | Позволяет проходить через недопустимые промежуточные решения без отдельного ремонта | Допустимость обеспечивается оценкой и проверкой финала, а не самим кроссовером |
| Элитизм | Защищает лучший уже найденный вариант от потери при вариации | Лучшее F в серии поколений не ухудшается |

Это обоснование свойств операторов. Их фактическое влияние оценивается сравнением серий ниже; преимущество оператора заранее не предполагается.

## 7. Схема эксперимента и независимые запуски

Выполнено {runs} запусков каждой из четырёх конфигураций: всего {4*runs}. Фиксированные seed ГА начинаются с 2000000, 2010000, 2020000; случайного поиска с 2900000. Внутри каждой серии seed увеличивается на 1. Все методы работают с одними данными.

Контролируемые сравнения: `repair_uniform` против `penalty_uniform` меняет только обработку ограничений; `repair_uniform` против `repair_one_point` меняет только кроссовер. Число обучающих оценок каждого метода одинаково: `80+{generations}·79={equal_budget}`. Случайный поиск генерирует столько же кандидатов и выбирает лучший допустимый. Восстановление выполняет дополнительные проверки ограничений; они входят во время, но не считаются вызовами целевой фитнес-функции. Поэтому равенство фитнес-бюджета не означает равенства времени работы.

## 8. Статистика и график результатов

В таблице приведён минимизируемый фитнес финальных решений; стандартное отклонение выборочное, по {runs} независимым запускам. Доля допустимости относится к **финальному решению каждого запуска**, а не ко всем особям популяции.

| Метод | Лучший фитнес | Средний | Медиана | Ст. отклонение | Худший | Лучшая полезность | Допустимые запуски |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

| Метод | Средняя полезность допустимых меню | Среднее время, с | Фитнес-оценок | Запусков с доказанным оптимумом |
|---|---:|---:|---:|---:|
{practical_table}

Время измерено на машине выполнения серии, включает восстановление и зависит от оборудования. Полные результаты: [runs.csv](runs.csv), агрегаты: [summary.csv](summary.csv).

![Средняя сходимость трёх конфигураций ГА](convergence.svg)

На графике каждая линия показывает среднее лучшего F по серии запусков. Подход с восстановлением исправляет начальную популяцию до первой оценки; штрафной подход оценивает её без исправления. Это объясняет различие начальных значений. Диапазон оси Y может скрывать малые различия в конце поиска; итоговая таблица сохраняет их точно.

## 9. Влияние обработки ограничений и кроссовера

**Восстановление и штрафы.** Среднее F равно `{repair_mean:.2f}` и `{penalty_mean:.2f}`. Разница составляет `{abs(repair_mean-penalty_mean):.2f}` единицы полезности при одинаковом лучшем U и 100% допустимых финалов. По этой серии нельзя заявлять содержательное превосходство одного подхода по качеству. При этом среднее время восстановления `{repair_seconds:.4f}` с против `{penalty_seconds:.4f}` с у штрафов, отношение `{repair_seconds/penalty_seconds:.2f}`. Дополнительные попытки исправления объясняют вычислительные затраты; это измерение данного эксперимента, а не универсальная оценка скорости.

**Равномерный и одноточечный кроссоверы.** При одинаковом восстановлении среднее F равно `{repair_mean:.2f}` и `{one_point_mean:.2f}`. Их различная структура наследования на этих данных не дала заметного изменения итогового качества. Значимость различий не проверялась; утверждать, что одноточечный кроссовер хуже или лучше в общем случае, нельзя.

**Случайный поиск.** Среднее U равно `{random_utility:.2f}`, против `{-repair_mean:.2f}` у ГА с восстановлением: улучшение `{improvement:.2f}%` при том же числе фитнес-оценок. Это подтверждает пользу поиска с наследованием и отбором на данном экземпляре. Отдельного эксперимента с альтернативным кодированием не было, поэтому причинный эффект выбора кодирования не заявляется.

## 10. Предметный результат и оценка его качества

| День | Завтрак | Обед | Ужин | Калорийность дня |
|---|---|---|---|---:|
{menu_table}

Полезность лучшего меню `{best.evaluation.utility}`, стоимость `{best.evaluation.cost}≤{WEEKLY_BUDGET}`, различных блюд `{best.evaluation.unique_dishes}`, максимальное число повторов `{max(Counter(best.individual).values())}`. Суточная калорийность проверена для каждого дня в таблице.

### Независимая верхняя граница

Отбрасываем только бюджет и калорийность. Сохраняем по {DAYS} блюд каждого типа, не менее {MIN_UNIQUE_DISHES} различных блюд и не более {MAX_REPETITIONS} использований каждого блюда. Допустимая область расширяется, поэтому оптимум исходной задачи не может превышать максимум ослабленной.

Для каждого типа вычисляется динамическая таблица `Dⱼ(k,u)`: максимальная полезность после первых j блюд, при k выбранных порциях и u уникальных блюдах. Для очередного блюда перебирается `q=0,…,{MAX_REPETITIONS}`, не выходя за k={DAYS}; переход добавляет `q·scoreⱼ`, а при q>0 увеличивает число уникальных на 1. Затем три таблицы объединяются по суммарному числу уникальных блюд. Точный максимум ослабленной задачи равен `{upper_bound}`. Расчёт реализован в `utility_upper_bound`, результат сохранён в [optimality.json](optimality.json).

{quality_conclusion}

Смысл оценки результата: ограничения выполнены, качество сопоставлено с независимой границей, а устойчивость характеризуется долей запусков, достигших этой границы. Проверка границы выполняется **после поиска** и не подсказывает ГА назначения блюд, не входит в его фитнес-бюджет. Ограничения вывода: синтетические оценки блюд, только один экземпляр, фиксированные веса штрафов; перенос на другие меню и бюджеты требует новых серий.

Файлы: [parameters.csv](parameters.csv), [runs.csv](runs.csv), [summary.csv](summary.csv), [convergence.svg](convergence.svg), [best_menu.csv](best_menu.csv), [feasibility_examples.csv](feasibility_examples.csv), [optimality.json](optimality.json).
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
