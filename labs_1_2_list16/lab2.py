"""Лабораторная №2: дискретный генетический алгоритм с ограничениями.

Поддерживаемые варианты для номера в списке 16:
507 -> V19 расписание экзаменов; 517 -> V6 план закупки;
527 -> V13 маршруты сбора отходов.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from common import ensure_dir, line_chart_svg, write_csv
from variant import calculate_variant


Individual = list[int]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    constraint_mode: str
    crossover: str
    mutation: str
    population: int = 80
    generations: int = 180
    crossover_probability: float = 0.9
    mutation_probability: float = 0.25


@dataclass
class Evaluation:
    fitness: float
    metric: float
    feasible: bool
    violations: int
    details: dict[str, Any]


@dataclass
class RunResult:
    config: str
    seed: int
    fitness: float
    metric: float
    feasible: bool
    violations: int
    individual: Individual
    details: dict[str, Any]
    trajectory: list[float]
    elapsed_seconds: float


class DiscreteProblem(Protocol):
    variant: int
    name: str
    metric_name: str
    metric_direction: str

    def random_individual(self, rng: random.Random) -> Individual: ...
    def crossover(self, a: Individual, b: Individual, mode: str, rng: random.Random) -> Individual: ...
    def mutate(self, individual: Individual, mode: str, rng: random.Random) -> None: ...
    def repair(self, individual: Individual, rng: random.Random) -> Individual: ...
    def evaluate(self, individual: Individual, constraint_mode: str) -> Evaluation: ...
    def baseline(self) -> tuple[Individual, Evaluation]: ...
    def examples(self) -> list[tuple[str, Individual, Evaluation]]: ...
    def save_instance(self, output: Path) -> None: ...
    def save_solution(self, output: Path, individual: Individual, evaluation: Evaluation) -> None: ...


def one_point(a: Individual, b: Individual, rng: random.Random) -> Individual:
    point = rng.randrange(1, len(a))
    return a[:point] + b[point:]


def uniform(a: Individual, b: Individual, rng: random.Random) -> Individual:
    return [left if rng.random() < 0.5 else right for left, right in zip(a, b)]


def order_crossover(a: Sequence[int], b: Sequence[int], rng: random.Random) -> list[int]:
    left, right = sorted(rng.sample(range(len(a)), 2))
    child: list[int | None] = [None] * len(a)
    child[left : right + 1] = a[left : right + 1]
    fill = [gene for gene in b if gene not in child]
    positions = list(range(right + 1, len(a))) + list(range(0, left))
    for position, gene in zip(positions, fill):
        child[position] = gene
    return [int(gene) for gene in child]


def pmx_crossover(a: Sequence[int], b: Sequence[int], rng: random.Random) -> list[int]:
    left, right = sorted(rng.sample(range(len(a)), 2))
    child = [-1] * len(a)
    child[left : right + 1] = a[left : right + 1]
    for index in range(left, right + 1):
        gene = b[index]
        if gene in child:
            continue
        position = index
        while left <= position <= right:
            mapped = a[position]
            position = b.index(mapped)
        child[position] = gene
    for index, gene in enumerate(b):
        if child[index] == -1:
            child[index] = gene
    return child


class ProcurementProblem:
    variant = 6
    name = "План закупки расходных материалов"
    metric_name = "полезность плана"
    metric_direction = "max"

    def __init__(self) -> None:
        rng = random.Random(6_160_026)
        self.items = []
        for index in range(30):
            minimum = rng.randint(1, 4)
            maximum = minimum + rng.randint(5, 14)
            self.items.append(
                {
                    "id": index,
                    "name": f"Материал {index + 1:02d}",
                    "unit_cost": rng.randint(6, 55),
                    "unit_utility": rng.randint(8, 95),
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )
        minimum_cost = sum(item["minimum"] * item["unit_cost"] for item in self.items)
        optional_cost = sum((item["maximum"] - item["minimum"]) * item["unit_cost"] for item in self.items)
        self.budget = minimum_cost + int(optional_cost * 0.38)

    def random_individual(self, rng: random.Random) -> Individual:
        return [rng.randint(item["minimum"], item["maximum"]) for item in self.items]

    def crossover(self, a: Individual, b: Individual, mode: str, rng: random.Random) -> Individual:
        return uniform(a, b, rng) if mode == "uniform" else one_point(a, b, rng)

    def mutate(self, individual: Individual, mode: str, rng: random.Random) -> None:
        for index, item in enumerate(self.items):
            if rng.random() < 0.08:
                if mode == "reset":
                    individual[index] = rng.randint(item["minimum"], item["maximum"])
                else:
                    individual[index] += rng.choice((-2, -1, 1, 2))
                    individual[index] = min(item["maximum"], max(0, individual[index]))

    def repair(self, individual: Individual, rng: random.Random) -> Individual:
        result = [min(item["maximum"], max(item["minimum"], value)) for value, item in zip(individual, self.items)]
        cost = sum(value * item["unit_cost"] for value, item in zip(result, self.items))
        removable = sorted(
            range(len(result)),
            key=lambda index: self.items[index]["unit_utility"] / self.items[index]["unit_cost"],
        )
        while cost > self.budget:
            changed = False
            for index in removable:
                item = self.items[index]
                if result[index] > item["minimum"]:
                    result[index] -= 1
                    cost -= item["unit_cost"]
                    changed = True
                    if cost <= self.budget:
                        break
            if not changed:
                break
        return result

    def evaluate(self, individual: Individual, constraint_mode: str) -> Evaluation:
        cost = sum(value * item["unit_cost"] for value, item in zip(individual, self.items))
        utility = sum(value * item["unit_utility"] for value, item in zip(individual, self.items))
        shortage = sum(max(0, item["minimum"] - value) for value, item in zip(individual, self.items))
        over_max = sum(max(0, value - item["maximum"]) for value, item in zip(individual, self.items))
        overspend = max(0, cost - self.budget)
        violations = shortage + over_max + (1 if overspend else 0)
        penalty = 3_000 * (shortage + over_max) + 50 * overspend
        return Evaluation(-utility + penalty, float(utility), violations == 0, violations, {"cost": cost, "budget": self.budget, "utility": utility})

    def baseline(self) -> tuple[Individual, Evaluation]:
        plan = [item["minimum"] for item in self.items]
        remaining = self.budget - sum(value * item["unit_cost"] for value, item in zip(plan, self.items))
        order = sorted(range(len(plan)), key=lambda i: self.items[i]["unit_utility"] / self.items[i]["unit_cost"], reverse=True)
        for index in order:
            item = self.items[index]
            amount = min(item["maximum"] - plan[index], remaining // item["unit_cost"])
            plan[index] += amount
            remaining -= amount * item["unit_cost"]
        return plan, self.evaluate(plan, "repair")

    def examples(self) -> list[tuple[str, Individual, Evaluation]]:
        valid, evaluation = self.baseline()
        valid2 = self.repair([item["maximum"] for item in self.items], random.Random(1))
        invalid1 = [0] * len(self.items)
        invalid2 = [item["maximum"] for item in self.items]
        return [
            ("допустимый: жадный", valid, evaluation),
            ("допустимый: восстановленный максимум", valid2, self.evaluate(valid2, "repair")),
            ("недопустимый: ниже минимальных запасов", invalid1, self.evaluate(invalid1, "penalty")),
            ("недопустимый: превышение бюджета", invalid2, self.evaluate(invalid2, "penalty")),
        ]

    def save_instance(self, output: Path) -> None:
        write_csv(output / "instance_items.csv", ["id", "name", "unit_cost", "unit_utility", "minimum", "maximum"], ([i[k] for k in ("id", "name", "unit_cost", "unit_utility", "minimum", "maximum")] for i in self.items))
        (output / "instance_meta.json").write_text(json.dumps({"budget": self.budget}, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_solution(self, output: Path, individual: Individual, evaluation: Evaluation) -> None:
        write_csv(output / "best_plan.csv", ["item", "quantity", "unit_cost", "unit_utility", "line_cost", "line_utility"], ([item["name"], value, item["unit_cost"], item["unit_utility"], value * item["unit_cost"], value * item["unit_utility"]] for value, item in zip(individual, self.items)))


class ExamSchedulingProblem:
    variant = 19
    name = "Расписание экзаменов с конфликтами студентов"
    metric_name = "штраф расписания"
    metric_direction = "min"

    def __init__(self) -> None:
        self.exam_count = 24
        self.slot_count = 6
        self.slot_capacity = 5
        rng = random.Random(19_160_026)
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < 46:
            a, b = sorted(rng.sample(range(self.exam_count), 2))
            pairs.add((a, b))
        self.conflicts = {(a, b): rng.randint(1, 18) for a, b in sorted(pairs)}

    def random_individual(self, rng: random.Random) -> Individual:
        return [rng.randrange(self.slot_count) for _ in range(self.exam_count)]

    def crossover(self, a: Individual, b: Individual, mode: str, rng: random.Random) -> Individual:
        return uniform(a, b, rng) if mode == "uniform" else one_point(a, b, rng)

    def mutate(self, individual: Individual, mode: str, rng: random.Random) -> None:
        index = rng.randrange(self.exam_count)
        individual[index] = rng.randrange(self.slot_count)

    def _degree(self, exam: int) -> int:
        return sum(weight for (a, b), weight in self.conflicts.items() if a == exam or b == exam)

    def repair(self, individual: Individual, rng: random.Random) -> Individual:
        result = [-1] * self.exam_count
        counts = [0] * self.slot_count
        order = sorted(range(self.exam_count), key=lambda exam: (-self._degree(exam), individual[exam]))
        for exam in order:
            choices = []
            for slot in range(self.slot_count):
                conflicts = sum(
                    weight
                    for (a, b), weight in self.conflicts.items()
                    if ((a == exam and result[b] == slot) or (b == exam and result[a] == slot))
                )
                overflow = max(0, counts[slot] + 1 - self.slot_capacity)
                preference = 0.05 * abs(slot - individual[exam])
                choices.append((10_000 * (conflicts + overflow), preference, counts[slot], slot))
            slot = min(choices)[-1]
            result[exam] = slot
            counts[slot] += 1
        return result

    def evaluate(self, individual: Individual, constraint_mode: str) -> Evaluation:
        counts = [individual.count(slot) for slot in range(self.slot_count)]
        same_slot = sum(weight for (a, b), weight in self.conflicts.items() if individual[a] == individual[b])
        overflow = sum(max(0, count - self.slot_capacity) for count in counts)
        adjacent = sum(weight for (a, b), weight in self.conflicts.items() if abs(individual[a] - individual[b]) == 1)
        used = len(set(individual))
        violations = same_slot + overflow
        metric = used * 100 + adjacent
        fitness = metric + 10_000 * violations
        return Evaluation(float(fitness), float(metric), violations == 0, violations, {"used_slots": used, "adjacent_conflict_weight": adjacent, "slot_counts": counts})

    def baseline(self) -> tuple[Individual, Evaluation]:
        schedule = self.repair([0] * self.exam_count, random.Random(1))
        return schedule, self.evaluate(schedule, "repair")

    def examples(self) -> list[tuple[str, Individual, Evaluation]]:
        valid1, eval1 = self.baseline()
        valid2 = [(slot + 1) % self.slot_count for slot in valid1]
        invalid1 = [0] * self.exam_count
        edge = next(iter(self.conflicts))
        invalid2 = valid1[:]
        invalid2[edge[1]] = invalid2[edge[0]]
        return [
            ("допустимый: жадная раскраска", valid1, eval1),
            ("допустимый: циклический сдвиг", valid2, self.evaluate(valid2, "penalty")),
            ("недопустимый: все экзамены в одном слоте", invalid1, self.evaluate(invalid1, "penalty")),
            ("недопустимый: искусственный конфликт", invalid2, self.evaluate(invalid2, "penalty")),
        ]

    def save_instance(self, output: Path) -> None:
        write_csv(output / "instance_conflicts.csv", ["exam_a", "exam_b", "shared_students"], ((a + 1, b + 1, weight) for (a, b), weight in self.conflicts.items()))
        (output / "instance_meta.json").write_text(json.dumps({"exam_count": self.exam_count, "slot_count": self.slot_count, "slot_capacity": self.slot_capacity}, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_solution(self, output: Path, individual: Individual, evaluation: Evaluation) -> None:
        write_csv(output / "best_schedule.csv", ["exam", "slot"], ((index + 1, slot + 1) for index, slot in enumerate(individual)))


class WasteRoutingProblem:
    variant = 13
    name = "Маршрут сбора отходов с ограничением грузоподъёмности"
    metric_name = "длина маршрутов"
    metric_direction = "min"

    def __init__(self) -> None:
        rng = random.Random(13_160_026)
        self.customer_count = 18
        self.capacity = 35
        self.coords = [(50.0, 50.0)] + [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(self.customer_count)]
        self.demands = [0] + [rng.randint(4, 11) for _ in range(self.customer_count)]
        self.vehicle_count = 7

    @staticmethod
    def _routes(individual: Individual) -> list[list[int]]:
        routes: list[list[int]] = [[]]
        for gene in individual:
            if gene == 0:
                routes.append([])
            else:
                routes[-1].append(gene)
        return routes

    @staticmethod
    def _flatten(routes: Sequence[Sequence[int]]) -> Individual:
        result: Individual = []
        for index, route in enumerate(routes):
            if index:
                result.append(0)
            result.extend(route)
        return result

    def _permutation(self, individual: Individual) -> list[int]:
        return [gene for gene in individual if gene != 0]

    def _route_lengths(self, individual: Individual) -> list[int]:
        return [len(route) for route in self._routes(individual)]

    def _with_lengths(self, permutation: Sequence[int], lengths: Sequence[int]) -> Individual:
        lengths = list(lengths[: self.vehicle_count])
        while len(lengths) < self.vehicle_count:
            lengths.append(0)
        difference = len(permutation) - sum(lengths)
        lengths[-1] += difference
        routes, cursor = [], 0
        for length in lengths:
            length = max(0, length)
            routes.append(list(permutation[cursor : cursor + length]))
            cursor += length
        routes[-1].extend(permutation[cursor:])
        return self._flatten(routes)

    def random_individual(self, rng: random.Random) -> Individual:
        customers = list(range(1, self.customer_count + 1))
        rng.shuffle(customers)
        cuts = sorted(rng.sample(range(1, self.customer_count), self.vehicle_count - 1))
        lengths = [right - left for left, right in zip([0, *cuts], [*cuts, self.customer_count])]
        return self._with_lengths(customers, lengths)

    def crossover(self, a: Individual, b: Individual, mode: str, rng: random.Random) -> Individual:
        perm_a, perm_b = self._permutation(a), self._permutation(b)
        child_perm = order_crossover(perm_a, perm_b, rng) if mode == "OX" else pmx_crossover(perm_a, perm_b, rng)
        lengths = self._route_lengths(a if rng.random() < 0.5 else b)
        return self._with_lengths(child_perm, lengths)

    def mutate(self, individual: Individual, mode: str, rng: random.Random) -> None:
        permutation = self._permutation(individual)
        lengths = self._route_lengths(individual)
        left, right = sorted(rng.sample(range(len(permutation)), 2))
        if mode == "swap":
            permutation[left], permutation[right] = permutation[right], permutation[left]
        else:
            permutation[left : right + 1] = reversed(permutation[left : right + 1])
        individual[:] = self._with_lengths(permutation, lengths)

    def repair(self, individual: Individual, rng: random.Random) -> Individual:
        routes: list[list[int]] = [[]]
        loads = [0]
        for customer in self._permutation(individual):
            demand = self.demands[customer]
            if loads[-1] + demand <= self.capacity:
                routes[-1].append(customer)
                loads[-1] += demand
            else:
                routes.append([customer])
                loads.append(demand)
        while len(routes) < self.vehicle_count:
            index = max(range(len(routes)), key=lambda i: len(routes[i]))
            route = routes[index]
            cut = max(1, len(route) // 2)
            routes[index : index + 1] = [route[:cut], route[cut:]]
        if len(routes) > self.vehicle_count:
            for route in routes[self.vehicle_count :]:
                routes[self.vehicle_count - 1].extend(route)
            routes = routes[: self.vehicle_count]
        return self._flatten(routes)

    def _distance(self, a: int, b: int) -> float:
        return math.dist(self.coords[a], self.coords[b])

    def evaluate(self, individual: Individual, constraint_mode: str) -> Evaluation:
        routes = self._routes(individual)
        total_distance = 0.0
        overload = 0
        route_loads = []
        for route in routes:
            load = sum(self.demands[customer] for customer in route)
            route_loads.append(load)
            overload += max(0, load - self.capacity)
            previous = 0
            for customer in route:
                total_distance += self._distance(previous, customer)
                previous = customer
            total_distance += self._distance(previous, 0)
        permutation = self._permutation(individual)
        missing = self.customer_count - len(set(permutation))
        violations = overload + max(0, missing)
        fitness = total_distance + 2_000 * violations
        return Evaluation(fitness, total_distance, violations == 0, violations, {"route_loads": route_loads, "routes": routes, "capacity": self.capacity})

    def baseline(self) -> tuple[Individual, Evaluation]:
        remaining = set(range(1, self.customer_count + 1))
        routes = []
        while remaining:
            route, load, current = [], 0, 0
            while True:
                candidates = [customer for customer in remaining if load + self.demands[customer] <= self.capacity]
                if not candidates:
                    break
                nxt = min(candidates, key=lambda customer: self._distance(current, customer))
                route.append(nxt)
                remaining.remove(nxt)
                load += self.demands[nxt]
                current = nxt
            routes.append(route)
        while len(routes) < self.vehicle_count:
            routes.append([])
        individual = self._flatten(routes)
        return individual, self.evaluate(individual, "repair")

    def examples(self) -> list[tuple[str, Individual, Evaluation]]:
        valid1, eval1 = self.baseline()
        valid2 = self.repair(self.random_individual(random.Random(2)), random.Random(2))
        permutation = list(range(1, self.customer_count + 1))
        invalid1 = self._flatten([permutation, *([[]] * (self.vehicle_count - 1))])
        invalid2 = self.random_individual(random.Random(4))
        return [
            ("допустимый: ближайший сосед", valid1, eval1),
            ("допустимый: восстановленная перестановка", valid2, self.evaluate(valid2, "repair")),
            ("недопустимый: все точки одним автомобилем", invalid1, self.evaluate(invalid1, "penalty")),
            ("недопустимый: случайные разделители", invalid2, self.evaluate(invalid2, "penalty")),
        ]

    def save_instance(self, output: Path) -> None:
        write_csv(output / "instance_points.csv", ["id", "x", "y", "demand"], ((index, x, y, self.demands[index]) for index, (x, y) in enumerate(self.coords)))
        (output / "instance_meta.json").write_text(json.dumps({"capacity": self.capacity, "vehicle_count": self.vehicle_count}, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_solution(self, output: Path, individual: Individual, evaluation: Evaluation) -> None:
        rows = []
        for route_index, route in enumerate(self._routes(individual), 1):
            load = sum(self.demands[customer] for customer in route)
            rows.append([route_index, "0-" + "-".join(map(str, route)) + "-0", load])
        write_csv(output / "best_routes.csv", ["vehicle", "route", "load"], rows)


def make_problem(group: int) -> DiscreteProblem:
    variant = calculate_variant(16, group).number
    if variant == 6:
        return ProcurementProblem()
    if variant == 13:
        return WasteRoutingProblem()
    if variant == 19:
        return ExamSchedulingProblem()
    raise AssertionError(f"Для номера 16 не ожидался вариант {variant}")


def tournament(population: Sequence[Individual], evaluations: Sequence[Evaluation], rng: random.Random) -> Individual:
    indexes = rng.sample(range(len(population)), 3)
    return population[min(indexes, key=lambda index: evaluations[index].fitness)]


def run_ga(problem: DiscreteProblem, config: ExperimentConfig, seed: int) -> RunResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    population = [problem.random_individual(rng) for _ in range(config.population)]
    if config.constraint_mode == "repair":
        population = [problem.repair(item, rng) for item in population]
    evaluations = [problem.evaluate(item, config.constraint_mode) for item in population]
    trajectory = [min(item.fitness for item in evaluations)]

    for _ in range(config.generations):
        elite_index = min(range(len(population)), key=lambda index: evaluations[index].fitness)
        next_population = [population[elite_index][:]]
        while len(next_population) < config.population:
            parent_a = tournament(population, evaluations, rng)
            parent_b = tournament(population, evaluations, rng)
            child = (
                problem.crossover(parent_a, parent_b, config.crossover, rng)
                if rng.random() < config.crossover_probability
                else parent_a[:]
            )
            if rng.random() < config.mutation_probability:
                problem.mutate(child, config.mutation, rng)
            if config.constraint_mode == "repair":
                child = problem.repair(child, rng)
            next_population.append(child)
        population = next_population
        evaluations = [problem.evaluate(item, config.constraint_mode) for item in population]
        trajectory.append(min(item.fitness for item in evaluations))

    best_index = min(range(len(population)), key=lambda index: evaluations[index].fitness)
    best_eval = evaluations[best_index]
    return RunResult(
        config.name,
        seed,
        best_eval.fitness,
        best_eval.metric,
        best_eval.feasible,
        best_eval.violations,
        population[best_index],
        best_eval.details,
        trajectory,
        time.perf_counter() - started,
    )


def experiment(group: int, runs: int, output_root: Path) -> Path:
    if runs < 20:
        raise ValueError("По заданию требуется не менее 20 независимых запусков")
    problem = make_problem(group)
    output = ensure_dir(output_root / f"group_{group}_v{problem.variant}")
    if problem.variant == 13:
        configs = [
            ExperimentConfig("repair_OX_swap", "repair", "OX", "swap"),
            ExperimentConfig("penalty_OX_swap", "penalty", "OX", "swap"),
            ExperimentConfig("repair_PMX_inversion", "repair", "PMX", "inversion"),
        ]
    else:
        configs = [
            ExperimentConfig("repair_uniform_step", "repair", "uniform", "step"),
            ExperimentConfig("penalty_uniform_step", "penalty", "uniform", "step"),
            ExperimentConfig("repair_one_point_reset", "repair", "one_point", "reset"),
        ]

    problem.save_instance(output)
    results: list[RunResult] = []
    for config_index, config in enumerate(configs):
        for run in range(runs):
            results.append(run_ga(problem, config, problem.variant * 100_000 + config_index * 10_000 + run))

    baseline_individual, baseline_eval = problem.baseline()
    rows = [
        [item.config, item.seed, item.fitness, item.metric, item.feasible, item.violations, item.elapsed_seconds, json.dumps(item.individual), json.dumps(item.details, ensure_ascii=False)]
        for item in results
    ]
    rows.append(["constructive_baseline", "-", baseline_eval.fitness, baseline_eval.metric, baseline_eval.feasible, baseline_eval.violations, 0.0, json.dumps(baseline_individual), json.dumps(baseline_eval.details, ensure_ascii=False)])
    write_csv(output / "runs.csv", ["method", "seed", "fitness", "domain_metric", "feasible", "violations", "elapsed_seconds", "individual", "details"], rows)

    summary_rows = []
    for config in configs:
        subset = [item for item in results if item.config == config.name]
        fitness = [item.fitness for item in subset]
        summary_rows.append(
            [
                config.name,
                min(fitness),
                statistics.fmean(fitness),
                statistics.median(fitness),
                statistics.stdev(fitness),
                max(fitness),
                sum(item.feasible for item in subset) / len(subset),
            ]
        )
    summary_rows.append(["constructive_baseline", baseline_eval.fitness, baseline_eval.fitness, baseline_eval.fitness, 0.0, baseline_eval.fitness, float(baseline_eval.feasible)])
    write_csv(output / "summary.csv", ["method", "best_fitness", "mean_fitness", "median_fitness", "std_fitness", "worst_fitness", "feasible_rate"], summary_rows)

    best = min((item for item in results if item.feasible), key=lambda item: item.fitness, default=min(results, key=lambda item: item.fitness))
    best_eval = problem.evaluate(best.individual, "repair")
    problem.save_solution(output, best.individual, best_eval)

    examples = problem.examples()
    write_csv(
        output / "feasibility_examples.csv",
        ["description", "feasible", "violations", "fitness", "individual"],
        ([description, evaluation.feasible, evaluation.violations, evaluation.fitness, json.dumps(individual)] for description, individual, evaluation in examples),
    )

    chart_series = {}
    for config in configs:
        subset = [item for item in results if item.config == config.name]
        length = min(len(item.trajectory) for item in subset)
        chart_series[config.name] = [statistics.fmean(item.trajectory[index] for item in subset) for index in range(length)]
    line_chart_svg(output / "convergence.svg", chart_series, title=f"ЛР2, группа {group}, V={problem.variant}: средняя сходимость", x_label="Поколение", y_label="Фитнес (меньше лучше)")

    table = "\n".join(
        f"| {row[0]} | " + " | ".join(f"{float(value):.7g}" for value in row[1:6]) + f" | {float(row[6]):.0%} |"
        for row in summary_rows
    )
    examples_text = "\n".join(
        f"- **{description}**: feasible={evaluation.feasible}, violations={evaluation.violations}, fitness={evaluation.fitness:.6g}."
        for description, _, evaluation in examples
    )
    report = f"""# Лабораторная работа №2

## Вариант и постановка

- Номер в списке: 16; группа: {group}; вариант: **V={problem.variant}**.
- Задача: **{problem.name}**.
- Основное представление и все исходные данные сохранены рядом с отчётом.
- Предметный показатель: **{problem.metric_name}**, направление оптимизации: `{problem.metric_direction}`.

Особь имеет дискретное представление. Для работы с ограничениями сравниваются штрафная функция и восстановление допустимости. Третья конфигурация меняет специализированные операторы кроссовера и мутации. Использованы популяция {configs[0].population}, {configs[0].generations} поколений, турнирная селекция и элитизм одной особи.

## Проверка допустимости

{examples_text}

Полные особи и признаки нарушений приведены в [feasibility_examples.csv](feasibility_examples.csv).

## Результаты {runs} независимых запусков

Фитнес унифицирован: меньше — лучше; недопустимые решения получают большой штраф.

| Метод | Лучшее | Среднее | Медиана | Ст. отклонение | Худшее | Допустимые запуски |
|---|---:|---:|---:|---:|---:|---:|
{table}

Лучшее допустимое решение: фитнес `{best.fitness:.10g}`, {problem.metric_name} `{best.metric:.10g}`. Человекочитаемое решение записано отдельным CSV-файлом (`best_*.csv`).

## Вывод

Кодирование напрямую отражает структуру задачи, а не маскирует её вещественным вектором. Восстановление гарантирует или резко повышает долю допустимых кандидатов; штрафной подход сохраняет больше разнообразия, но тратит часть бюджета на недопустимые решения. Сравнение третьей конфигурации показывает влияние оператора при неизменной общей схеме ГА. Конструктивная эвристика служит понятной базовой линией.

График: [convergence.svg](convergence.svg). Данные: [runs.csv](runs.csv), [summary.csv](summary.csv).
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, choices=(507, 517, 527), required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/lab2"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_path = experiment(args.group, args.runs, args.output)
    print(f"Готово: {result_path}")
