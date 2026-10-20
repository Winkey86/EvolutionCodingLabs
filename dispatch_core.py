"""Симулятор и собственное генетическое программирование для ЛР3–ЛР4.

Приоритеты используют только прогноз: фактические сценарные длительности
применяются после назначения и никогда не передаются правилу.
"""

from __future__ import annotations

import csv
import html
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

TERMINALS = ("finish", "duration", "late", "cost", "load", "slack")
OPERATORS = ("add", "sub", "mul", "div", "min", "max")
CONSTANTS = (0.25, 0.5, 1.0, 2.0)
COLORS = ("#2563eb", "#ea580c", "#059669", "#9333ea")


@dataclass(frozen=True)
class Task:
    release: float
    work: float
    deadline: float
    skill: int


@dataclass(frozen=True)
class Agent:
    speed: float
    cost: float
    skills: tuple[int, ...]


@dataclass(frozen=True)
class Scenario:
    seed: int
    tasks: tuple[Task, ...]
    agents: tuple[Agent, ...]
    multipliers: tuple[tuple[float, ...], ...]


def scenario(seed: int, task_count: int, uncertainty: float) -> Scenario:
    rng = random.Random(seed)
    agents = tuple(Agent(rng.uniform(0.7, 1.8), rng.uniform(0.6, 2.0),
                         (0, 1, 2) if i == 0 else tuple(sorted(rng.sample(range(3), 2))))
                   for i in range(5))
    tasks = []
    release = 0.0
    for _ in range(task_count):
        release += rng.uniform(0.1, 1.0)
        work = rng.uniform(1.0, 8.0)
        tasks.append(Task(release, work, release + rng.uniform(6.0, 16.0), rng.randrange(3)))
    multipliers = tuple(tuple(rng.lognormvariate(-uncertainty**2 / 2, uncertainty)
                             if uncertainty else 1.0 for _ in agents) for _ in tasks)
    return Scenario(seed, tuple(tasks), agents, multipliers)


def make_dataset(task_count: int, uncertainty: float, base_seed: int) -> dict:
    return {split: [scenario(base_seed + offset + i, task_count, uncertainty)
                    for i in range(count)]
            for split, offset, count in (("train", 0, 4), ("validation", 1000, 4), ("test", 2000, 10))}


def save_dataset(path: Path, dataset: dict) -> None:
    def encode(s):
        return {"seed": s.seed,
                "tasks": [[t.release, t.work, t.deadline, t.skill] for t in s.tasks],
                "agents": [[a.speed, a.cost, list(a.skills)] for a in s.agents],
                "multipliers": s.multipliers}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: [encode(s) for s in v] for k, v in dataset.items()},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dataset(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {split: [Scenario(s["seed"], tuple(Task(*t) for t in s["tasks"]),
                            tuple(Agent(a[0], a[1], tuple(a[2])) for a in s["agents"]),
                            tuple(tuple(row) for row in s["multipliers"])) for s in items]
            for split, items in data.items()}


def freeze(tree):
    return tuple(freeze(item) for item in tree) if isinstance(tree, (list, tuple)) else tree


def size(tree) -> int:
    return 1 + size(tree[1]) + size(tree[2]) if isinstance(tree, tuple) else 1


def depth(tree) -> int:
    return 1 + max(depth(tree[1]), depth(tree[2])) if isinstance(tree, tuple) else 0


def validate(tree, max_depth: int = 4) -> None:
    if isinstance(tree, tuple):
        if len(tree) != 3 or tree[0] not in OPERATORS:
            raise ValueError("Недопустимый оператор дерева")
        validate(tree[1], max_depth)
        validate(tree[2], max_depth)
    elif not (tree in TERMINALS if isinstance(tree, str)
              else type(tree) in (int, float) and tree in CONSTANTS):
        raise ValueError("Недопустимый терминал дерева")
    if depth(tree) > max_depth:
        raise ValueError("Превышена допустимая глубина дерева")


def expression(tree) -> str:
    if not isinstance(tree, tuple):
        return str(tree)
    op, left, right = tree
    a, b = expression(left), expression(right)
    return f"{op}({a}, {b})"


def compile_rule(tree):
    if isinstance(tree, str):
        index = TERMINALS.index(tree)
        return lambda features: features[index]
    if not isinstance(tree, tuple):
        return lambda features: tree
    op, left, right = tree
    a, b = compile_rule(left), compile_rule(right)
    def rule(features):
        x, y = a(features), b(features)
        if op == "add": value = x + y
        elif op == "sub": value = x - y
        elif op == "mul": value = x * y
        elif op == "div": value = x / y if abs(y) >= 0.05 else x
        elif op == "min": value = min(x, y)
        else: value = max(x, y)
        return max(-1000.0, min(1000.0, value))
    return rule


def simulate(tree, instance: Scenario, trace: bool = False) -> dict:
    rule = compile_rule(tree)
    forecast = [0.0] * len(instance.agents)
    actual = [0.0] * len(instance.agents)
    tardiness = cost = 0.0
    late_count = 0
    schedule = []
    for number, task in enumerate(instance.tasks):
        choices = []
        for index, agent in enumerate(instance.agents):
            if task.skill not in agent.skills:
                continue
            duration = task.work / agent.speed
            finish = max(task.release, forecast[index]) + duration
            features = (finish / 50, duration / 10, max(0.0, finish-task.deadline) / 10,
                        duration * agent.cost / 10,
                        max(0.0, forecast[index]-task.release) / 20,
                        (task.deadline-finish) / 20)
            choices.append((rule(features), index, finish, duration))
        if not choices:
            raise ValueError("Нет исполнителя с нужным навыком")
        _, index, forecast_finish, predicted_duration = min(choices)
        forecast[index] = forecast_finish
        start = max(task.release, actual[index])
        duration = predicted_duration * instance.multipliers[number][index]
        end = start + duration
        actual[index] = end
        tardiness += max(0.0, end-task.deadline)
        late_count += end > task.deadline
        cost += duration * instance.agents[index].cost
        if trace:
            schedule.append({"task": number, "agent": index, "release": task.release,
                             "deadline": task.deadline, "skill": task.skill,
                             "start": start, "end": end, "duration": duration})
    makespan = max(actual)
    mean_tardiness = tardiness / len(instance.tasks)
    return {"loss": makespan + 4*mean_tardiness + 0.05*cost,
            "makespan": makespan, "mean_tardiness": mean_tardiness, "cost": cost,
            "late_fraction": late_count / len(instance.tasks), "schedule": schedule}


def assess(tree, scenarios: list[Scenario], risk: float = 0.0) -> dict:
    metrics = [simulate(tree, s) for s in scenarios]
    losses = [item["loss"] for item in metrics]
    return {"quality": statistics.fmean(losses) + risk * statistics.pstdev(losses),
            **{key: statistics.fmean(item[key] for item in metrics)
               for key in ("loss", "makespan", "mean_tardiness", "cost", "late_fraction")}}


def random_tree(rng: random.Random, max_depth: int):
    if max_depth == 0 or rng.random() < 0.3:
        return rng.choice((*TERMINALS, *CONSTANTS))
    return (rng.choice(OPERATORS), random_tree(rng, max_depth-1), random_tree(rng, max_depth-1))


def paths(tree, prefix=()):
    yield prefix
    if isinstance(tree, tuple):
        yield from paths(tree[1], prefix+(1,))
        yield from paths(tree[2], prefix+(2,))


def subtree(tree, path):
    for branch in path:
        tree = tree[branch]
    return tree


def replace(tree, path, value):
    if not path:
        return value
    items = list(tree)
    items[path[0]] = replace(items[path[0]], path[1:], value)
    return tuple(items)


def vary(left, right, rng: random.Random, max_depth: int):
    child = left
    if rng.random() < 0.85:
        p = rng.choice(list(paths(left)))
        donor = subtree(right, rng.choice(list(paths(right))))
        candidate = replace(left, p, donor)
        if depth(candidate) <= max_depth:
            child = candidate
    if rng.random() < 0.35:
        child = mutate(child, rng, max_depth)
    return child


def mutate(tree, rng: random.Random, max_depth: int):
    path = rng.choice(list(paths(tree)))
    return replace(tree, path, random_tree(rng, min(2, max_depth-len(path))))


def evolve(dataset: dict, seed: int, budget: int, max_depth: int, penalty: float,
           risk: float = 0.0, memetic: bool = False, population_size: int = 32) -> dict:
    if budget < population_size or population_size < 3 or max_depth not in (1, 2, 3, 4):
        raise ValueError("Бюджет >= популяции >= 3; глубина от 1 до 4")
    rng = random.Random(seed)
    count = 0
    archive = {}
    history = []
    best_fitness = math.inf

    def evaluate(tree):
        nonlocal count, best_fitness
        quality = assess(tree, dataset["train"], risk)["quality"]
        fitness = quality + penalty*size(tree)
        count += 1
        record = (fitness, tree, quality)
        nodes = size(tree)
        if nodes not in archive or quality < archive[nodes][2]:
            archive[nodes] = record
        best_fitness = min(best_fitness, fitness)
        history.append((count, best_fitness))
        return record

    population = [evaluate(random_tree(rng, max_depth)) for _ in range(population_size)]
    while count < budget:
        elite = min(population, key=lambda item: item[0])
        next_population = [elite]
        while len(next_population) < population_size and count < budget:
            a = min(rng.sample(population, 3), key=lambda item: item[0])
            b = min(rng.sample(population, 3), key=lambda item: item[0])
            child = evaluate(vary(a[1], b[1], rng, max_depth))
            if memetic and count < budget and rng.random() < 0.4:
                neighbor = evaluate(mutate(child[1], rng, max_depth))
                if neighbor[0] < child[0]:
                    child = neighbor
            next_population.append(child)
        population = next_population

    # Выбор только по validation; test не участвует в поиске или выборе.
    candidates = [{"tree": item[1], "size": size(item[1]), "train_quality": item[2],
                   "validation_quality": assess(item[1], dataset["validation"], risk)["quality"]}
                  for item in archive.values()]
    chosen = min(candidates, key=lambda c: (c["validation_quality"]+penalty*c["size"], c["size"]))
    return {**chosen, "evaluations": count, "validation_evaluations": len(candidates),
            "history": history, "archive": candidates}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Нельзя сохранить пустую таблицу")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def summary(rows: list[dict]) -> list[dict]:
    output = []
    for method in dict.fromkeys(row["method"] for row in rows):
        group = [row for row in rows if row["method"] == method]
        values = [row["test_quality"] for row in group]
        output.append({"method": method, "runs": len(group), "best": min(values),
                       "mean": statistics.fmean(values), "median": statistics.median(values),
                       "std": statistics.stdev(values), "worst": max(values),
                       "mean_size": statistics.fmean(row["size"] for row in group),
                       "mean_seconds": statistics.fmean(row["seconds"] for row in group)})
    return output


def chart(path: Path, title: str, xlabel: str, ylabel: str, series: dict,
          scatter: bool = False) -> None:
    width, height = 1000, 640
    left, top, plot_w, plot_h = 105, 75, 850, 405
    points = [point for values in series.values() for point in values]
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    if xmax == xmin:
        xmin, xmax = xmin-0.5, xmax+0.5
    if abs(ymax-ymin) < 1e-8:
        padding = max(1.0, abs(ymin)*0.05)
        ymin, ymax = ymin-padding, ymax+padding
    dx, dy = max(1, xmax-xmin), max(1e-6, ymax-ymin)
    ymin, ymax = ymin-0.08*dy, ymax+0.08*dy
    def xy(a, b):
        return left+(a-xmin)/dx*plot_w, top+(ymax-b)/(ymax-ymin)*plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="500" y="35" text-anchor="middle" font-family="sans-serif" font-size="21">{html.escape(title)}</text>']
    for i in range(6):
        yval = ymin+i*(ymax-ymin)/5
        _, yy = xy(xmin, yval)
        parts += [f'<path d="M{left},{yy:.2f}h{plot_w}" stroke="#e2e8f0"/>',
                  f'<text x="95" y="{yy+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{yval:.2f}</text>']
        xval = xmin+i*dx/5
        xx, _ = xy(xval, ymin)
        parts += [f'<text x="{xx:.2f}" y="505" text-anchor="middle" font-family="sans-serif" font-size="13">{xval:.0f}</text>']
    for i, (name, values) in enumerate(series.items()):
        color = COLORS[i % len(COLORS)]
        if scatter:
            for a, b in values:
                xx, yy = xy(a, b)
                parts.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="4" fill="{color}" fill-opacity="0.75"/>')
        else:
            coordinates = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in (xy(a,b) for a,b in values))
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        columns = 3 if len(series) <= 3 else 2
        legend_x = 110+(i % columns)*(270 if columns == 3 else 430)
        legend_y = 575+(i // columns)*28
        parts.append(f'<text x="{legend_x}" y="{legend_y}" fill="{color}" font-family="sans-serif" font-size="14">{html.escape(name)}</text>')
    parts += [f'<text x="520" y="540" text-anchor="middle" font-family="sans-serif" font-size="16">{html.escape(xlabel)}</text>',
              f'<text x="25" y="280" text-anchor="middle" transform="rotate(-90 25 280)" font-family="sans-serif" font-size="15">{html.escape(ylabel)}</text>', '</svg>']
    path.write_text("\n".join(parts), encoding="utf-8")


def gantt(path: Path, schedule: list[dict], title: str) -> None:
    end = max(row["end"] for row in schedule)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="430" viewBox="0 0 1000 430">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="500" y="30" text-anchor="middle" font-family="sans-serif" font-size="21">{html.escape(title)}</text>']
    for agent in range(5):
        y = 65+agent*55
        parts.append(f'<text x="15" y="{y+20}" font-family="sans-serif" font-size="14">Агент {agent}</text>')
    for row in schedule:
        x, w = 100+row["start"]/end*870, row["duration"]/end*870
        y = 65+row["agent"]*55
        parts.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="35" fill="{COLORS[row["skill"]]}" stroke="white"><title>Задача {row["task"]}: {row["start"]:.2f}–{row["end"]:.2f}; срок {row["deadline"]:.2f}</title></rect>')
    for i in range(6):
        parts.append(f'<text x="{100+i*174}" y="365" text-anchor="middle" font-family="sans-serif" font-size="14">{end*i/5:.1f}</text>')
    parts += ['<text x="500" y="405" text-anchor="middle" font-family="sans-serif" font-size="16">Время, условные единицы; цвет — требуемый навык</text>', '</svg>']
    path.write_text("\n".join(parts), encoding="utf-8")
