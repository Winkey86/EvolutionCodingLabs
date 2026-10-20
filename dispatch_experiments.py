"""Воспроизводимые серии и отчёты для двух исследований распределения задач."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import dispatch_core as core

SOURCE = "https://docs.google.com/document/d/1yAlTbZGFGEu4ww2wHcwsvXlZOYQTkUbYLDp3l8Rfnko/edit"


def configuration(lab: int) -> dict:
    if lab == 3:
        return {"lab": 3, "runs": 20, "budget": 480, "population": 32,
                "task_count": 24, "uncertainty": 0.0, "data_seed": 3_201_800,
                "risk": 0.0, "penalty": 0.2,
                "methods": [{"name": f"GP_depth_{d}", "depth": d, "memetic": False}
                            for d in (1, 2, 4)]}
    return {"lab": 4, "runs": 30, "budget": 640, "population": 32,
            "task_count": 48, "uncertainty": 0.35, "data_seed": 4_201_800,
            "risk": 0.5, "penalty": 0.2,
            "methods": [{"name": "GP", "depth": 4, "memetic": False},
                        {"name": "Memetic_GP", "depth": 4, "memetic": True}]}


def run(lab: int, directory: Path, runs: int | None = None,
        budget: int | None = None, output: Path | None = None) -> None:
    config = configuration(lab)
    config["runs"] = config["runs"] if runs is None else runs
    config["budget"] = config["budget"] if budget is None else budget
    if config["runs"] < (20 if lab == 3 else 30):
        raise ValueError("Требуется минимум 20 запусков для ЛР3 и 30 для ЛР4")
    if config["budget"] < config["population"]:
        raise ValueError("Бюджет меньше популяции")
    output = output or directory / "results"
    output.mkdir(parents=True, exist_ok=True)
    dataset = core.make_dataset(config["task_count"], config["uncertainty"], config["data_seed"])
    core.save_dataset(directory / "data/scenarios.json", dataset)
    core.save_json(output / "config.json", {**config, "group": 517, "list_number": 18,
                                           "variant": 20, "crossover_probability": 0.85,
                                           "mutation_probability": 0.35,
                                           "local_search_probability": 0.4,
                                           "split_sizes": {k: len(v) for k,v in dataset.items()}})
    baseline_rules = {"earliest_finish": "finish", "least_cost": "cost", "shortest_duration": "duration"}
    baseline = []
    for name, tree in baseline_rules.items():
        baseline.append({"method": name, "tree": tree, "size": core.size(tree),
                         **core.assess(tree, dataset["test"], config["risk"])})
    core.write_csv(output / "baselines.csv", baseline)
    rows, models, histories, archives = [], [], {}, []
    for method_index, method in enumerate(config["methods"]):
        name = method["name"]
        histories[name] = []
        for run_index in range(config["runs"]):
            # Одинаковый seed в сравниваемых методах делает серии парными.
            seed = lab * 1_000_000 + 180_000 + run_index
            start = time.perf_counter()
            result = core.evolve(dataset, seed, config["budget"], method["depth"],
                                 config["penalty"], config["risk"], method["memetic"],
                                 config["population"])
            elapsed = time.perf_counter()-start
            metrics = core.assess(result["tree"], dataset["test"], config["risk"])
            row = {"method": name, "seed": seed, "evaluations": result["evaluations"],
                   "training_simulations": result["evaluations"]*len(dataset["train"]),
                   "validation_evaluations": result["validation_evaluations"],
                   "test_simulations": len(dataset["test"]), "size": result["size"],
                   "train_quality": result["train_quality"],
                   "validation_quality": result["validation_quality"],
                   "test_quality": metrics["quality"], "test_loss": metrics["loss"],
                   "test_makespan": metrics["makespan"],
                   "test_tardiness": metrics["mean_tardiness"],
                   "test_cost": metrics["cost"], "test_late_fraction": metrics["late_fraction"],
                   "seconds": elapsed, "expression": core.expression(result["tree"])}
            rows.append(row)
            models.append({"method": name, "seed": seed, "tree": result["tree"],
                           "validation_quality": result["validation_quality"], "size": result["size"]})
            histories[name].append(result["history"])
            archives.extend({**candidate, "method": name, "seed": seed}
                            for candidate in result["archive"])
            if (run_index+1) % 5 == 0:
                print(f"ЛР{lab} {name}: {run_index+1}/{config['runs']}", flush=True)
    core.write_csv(output / "runs.csv", rows)
    summaries = core.summary(rows)
    core.write_csv(output / "summary.csv", summaries)
    core.save_json(output / "models.json", models)
    history_rows, means = [], {}
    for name, runs_history in histories.items():
        low, mean, high = [], [], []
        for index in range(config["budget"]):
            values = [history[index][1] for history in runs_history]
            x = index+1
            low.append((x, min(values)))
            mean.append((x, statistics.fmean(values)))
            high.append((x, max(values)))
            history_rows.append({"method": name, "evaluation": x,
                                 "minimum": min(values), "mean": statistics.fmean(values),
                                 "maximum": max(values)})
        means[name] = mean
        core.chart(output / f"convergence_{name}.svg", f"ЛР{lab}: {name}, серия запусков",
                   "Фитнес-оценки", "Обучающий фитнес (меньше лучше)",
                   {"Минимум": low, "Среднее": mean, "Максимум": high})
    core.write_csv(output / "convergence.csv", history_rows)
    core.chart(output / "convergence.svg", f"ЛР{lab}: средняя сходимость",
               "Фитнес-оценки", "Обучающий фитнес (меньше лучше)", means)
    # Три эволюционировавшие модели выбираются по validation, до просмотра test.
    by_size = {}
    for candidate in archives:
        n = candidate["size"]
        if n not in by_size or candidate["validation_quality"] < by_size[n]["validation_quality"]:
            by_size[n] = candidate
    sizes = sorted(by_size)
    if len(sizes) < 3:
        raise RuntimeError("В архиве недостаточно различных сложностей для ЛР3")
    selected_sizes = (sizes[0], sizes[len(sizes)//2], sizes[-1])
    representatives = []
    for label, n in zip(("compact", "medium", "complex"), selected_sizes):
        candidate = by_size[n]
        representatives.append({"label": label, **candidate,
                                "expression": core.expression(candidate["tree"]),
                                **{"test_"+key: value for key,value in
                                   core.assess(candidate["tree"], dataset["test"], config["risk"]).items()}})
    core.save_json(output / "representatives.json", representatives)
    representative_rows = [{k: v for k,v in item.items() if k != "tree"} for item in representatives]
    core.write_csv(output / "representatives.csv", representative_rows)
    core.chart(output / "quality_complexity.svg", f"ЛР{lab}: качество и сложность",
               "Узлы дерева", "Тестовое качество (меньше лучше)",
               {**{method["name"]: [(row["size"], row["test_quality"]) for row in rows
                                    if row["method"] == method["name"]]
                   for method in config["methods"]},
                "Три представителя": [(item["size"], item["test_quality"]) for item in representatives]},
               scatter=True)
    # Демонстрационная модель выбирается только по validation и штрафу сложности.
    chosen = min(models, key=lambda m: m["validation_quality"]+config["penalty"]*m["size"])
    core.save_json(output / "selected_model.json", chosen)
    schedule = core.simulate(chosen["tree"], dataset["test"][0], trace=True)["schedule"]
    core.write_csv(output / "schedule.csv", schedule)
    core.gantt(output / "schedule.svg", schedule, f"ЛР{lab}: расписание на первом тестовом сценарии")
    report(directory, output, config, summaries, baseline, representatives, chosen)
    print(f"ЛР{lab}: готово, {output}", flush=True)


def report(directory, output, config, summaries, baseline, representatives, chosen):
    lab = config["lab"]
    table = "\n".join(f"| {r['method']} | {r['runs']} | {r['best']:.3f} | {r['mean']:.3f} | {r['median']:.3f} | {r['std']:.3f} | {r['worst']:.3f} | {r['mean_size']:.2f} | {r['mean_seconds']:.3f} |" for r in summaries)
    baselines = "\n".join(f"| {r['method']} | {r['quality']:.3f} | {r['makespan']:.3f} | {r['mean_tardiness']:.3f} | {r['cost']:.3f} | {100*r['late_fraction']:.1f}% |" for r in baseline)
    reps = "\n".join(f"| {r['label']} | {r['size']} | {r['validation_quality']:.3f} | {r['test_quality']:.3f} | `{r['expression']}` |" for r in representatives)
    best_method = min(summaries, key=lambda r: r["mean"])
    reference = min(baseline, key=lambda r: r["quality"])
    difference = 100*(reference["quality"]-best_method["mean"])/reference["quality"]
    if abs(difference) < 1e-9:
        comparison = (f"Среднее качество {best_method['method']} совпадает с лучшим ручным правилом. "
                      "Превосходство эволюции на этом экземпляре не обнаружено.")
    elif difference > 0:
        comparison = f"Среднее качество {best_method['method']} лучше лучшего ручного правила на {difference:.2f}%."
    else:
        comparison = (f"Среднее качество {best_method['method']} хуже лучшего ручного правила на {-difference:.2f}%; "
                      "превосходство эволюции над этой эвристикой не подтверждено.")
    if lab == 3:
        specific = """## Соответствие треку B

Вариант 20: синтез стратегии распределения задач между агентами. Эволюционирует структура программы, а не только числовые параметры. Механизмы, отсутствовавшие в ЛР1–ЛР2: симуляционный фитнес и рекомбинация/мутация поддеревьев.

Сравниваются три предельные глубины дерева: 1, 2 и 4. Остальные параметры одинаковы. Ниже показаны три эволюционировавшие программы различной сложности: для каждой сложности выбрана лучшая на validation, test при выборе не используется. Сложная модель может оказаться хуже простой — это результат исследования, а не ошибка отбора.
"""
        if all(r["mean_size"] == 1 for r in summaries):
            specific += "\nВо всех конфигурациях валидационный выбор с текущим штрафом предпочёл одноузловую программу `finish`. Увеличение разрешённой глубины не улучшило выбранную политику. Это приемлемый результат автоматического упрощения, но на данном экземпляре польза сложных программ не подтверждена. Архивные программы ниже показывают, почему требуется сравнивать не только выразительность, но и фактическое отложенное качество.\n"
        project = ""
    else:
        specific = """## Исследовательское расширение и сравнение

Вариант 20: собственная прикладная задача. До основной реализации подготовлена [заявка](../PROPOSAL.md). Подтверждение утверждения преподавателем не предоставлено; программная готовность не заменяет этого условия.

По сравнению с ЛР3 число задач увеличено с 24 до 48, фактические длительности имеют независимые логнормальные множители со стандартным отклонением логарифма 0.35 и средним множителем 1. Политика видит только прогнозируемые признаки. Фактическая длительность применяется в симуляторе после назначения, недоступна при выборе агента и не просачивается в прогноз.

Сравниваются GP и Memetic_GP с одинаковыми начальными seed, популяцией, глубиной, штрафом и общим бюджетом. Меметический вариант с вероятностью 0.4 проверяет одну локальную мутацию потомка и принимает улучшение; каждая такая проверка расходует один общий фитнес-вызов. Поэтому локальный поиск получает часть бюджета вместо части обычных потомков. Превосходство меметического метода не предполагается заранее.
"""
        project = "\nСценарий демонстрации на 6 минут и ответы на вопросы: [DEMO.md](../DEMO.md). Устная защита остаётся действием студента.\n"
    text = f"""# Отчёт по лабораторной работе №{lab}

Группа **517**, номер в списке **18**, вариант **20**. [Методические указания]({SOURCE}).

## Постановка и данные

Нужно построить компактное правило назначения задач неоднородным исполнителям. Каждый сценарий содержит {config['task_count']} задач и 5 агентов. Для задачи заданы время поступления `r`, объём `w`, срок `d`, навык `s`; агент имеет скорость `v`, стоимость единицы времени `c` и набор навыков. Задачи назначаются в порядке поступления, каждый агент исполняет очередь без прерываний. Допустим только агент с нужным навыком; агент 0 владеет всеми навыками, поэтому допустимое назначение существует. Начало задачи — максимум её поступления и окончания предыдущей задачи агента. Работы не делятся и не исполняются параллельно одним агентом.

Синтетические данные: скорость равномерна на [0.7,1.8], цена на [0.6,2], работа на [1,8], интервалы поступления на [0.1,1], резерв до срока на [6,16]. Каждый агент, кроме универсального, имеет два из трёх навыков. Seed данных `{config['data_seed']}`; разделение: 4 train, 4 validation, 10 test, непересекающиеся seed с отступами 0, 1000, 2000. Полные входные данные: [scenarios.json](../data/scenarios.json). Это учебная модель, а не измерения реальной системы.

## Представление и целевая функция

Генотип — дерево выражения глубиной до 4 (не более 31 узла); фенотип — функция приоритета агента. Для каждого допустимого агента вычисляется выражение, выбирается минимальное значение, при равенстве — меньший индекс агента.

Терминалы: `finish` — прогноз окончания /50; `duration` — прогноз длительности /10; `late` — прогноз опоздания /10; `cost` — прогноз стоимости /10; `load` — прогноз ожидания /20; `slack` — резерв до срока /20. Константы: 0.25, 0.5, 1, 2. Бинарные функции: `add`, `sub`, `mul`, `div`, `min`, `max`. В `div(a,b)` при |b|<0.05 возвращается a. Результаты функций ограничиваются диапазоном [−1000,1000].

Стоимость одного сценария `L = makespan + 4·mean_tardiness + 0.05·total_cost`. Качество — `Q = mean(L) + {config['risk']}·std_population(L)` по набору сценариев. Минимизируется обучающий фитнес `F = Q + {config['penalty']}·nodes`. Это явно выбранная свёртка времени, опозданий и расходов; конфликт качества со сложностью показан отдельно. Выигрыш качества может требовать более длинной программы, штраф ограничивает разрастание.

## Эволюционный алгоритм

Популяция {config['population']}, турнир 3, элитизм 1. Инициализация — grow: терминал с вероятностью 0.3, иначе случайный оператор с двумя потомками; на пределе глубины только терминал. Кроссовер поддеревьев с вероятностью 0.85, мутация поддерева с вероятностью 0.35. Недопустимо глубокий кроссовер отклоняется, глубина мутации ограничена оставшимся уровнем. Деревья синтаксически корректны по построению. Параметры и критерий остановки: [config.json](config.json).

```mermaid
flowchart TD
    A[Сгенерировать деревья] --> B[Симуляция train и штраф сложности]
    B --> C[Турнир и кроссовер поддеревьев]
    C --> D[Мутация и проверка глубины]
    D --> E[Оценка потомков в общем бюджете]
    E --> F[Элитизм и новая популяция]
    F --> G{{Бюджет исчерпан?}}
    G -- нет --> C
    G -- да --> H[Выбор из архива по validation]
    H --> I[Однократный тест и отчёт]
```

Турнир не требует масштабирования фитнеса; поддеревья сохраняют структуру языка, в отличие от вещественного кроссовера. Разделение train/validation/test снижает переобучение, архив по числу узлов сохраняет различные уровни сложности. Для выбора модели используется только validation + штраф; результаты test не возвращаются в эволюцию.

{specific}
## Эксперимент и честность бюджета

{config['runs']} независимых запусков каждого метода, seed `{lab*1_000_000+180_000}`…`{lab*1_000_000+180_000+config['runs']-1}`. Все методы получают одинаковые сценарии. Остановка после ровно {config['budget']} обучающих оценок, каждая включает 4 симуляции; кеширования нет, оценка элиты повторно не выполняется. Валидация выполняется отдельно для не более 31 размера архива; её число записано в runs.csv и не скрыто в обучающем бюджете. Тест выбранной модели — 10 симуляций; тест не используется для поиска. Ручные правила не обучаются и оценены на тех же 10 тестовых сценариях.

| Метод | Запусков | Лучшее Q | Среднее Q | Медиана | Ст. откл. | Худшее Q | Средние узлы | Время поиска, с |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

Меньше Q — лучше. [Все запуски](runs.csv), [сводка](summary.csv), [траектории](convergence.csv).

![Средние траектории](convergence.svg)

Для каждого метода отдельный файл `convergence_<method>.svg` показывает минимум, среднее и максимум по серии, поэтому разброс не скрыт за лучшей траекторией.

## Сравнение с простыми правилами

| Правило | Q | Время завершения | Среднее опоздание | Стоимость | Опоздавшие |
|---|---:|---:|---:|---:|---:|
{baselines}

{comparison} Сравнение относится к фиксированным отложенным сценариям; статистическая значимость и превосходство на всех возможных нагрузках не заявляются. [Данные baseline](baselines.csv).

## Три эволюционировавших решения

| Тип | Узлы | Q validation | Q test | Правило |
|---|---:|---:|---:|---|
{reps}

Компактное правило легче интерпретировать; среднее добавляет условия/взаимодействия признаков; сложное имеет большую выразительность, но риск избыточности и переобучения. В [representatives.json](representatives.json) сохранены сами деревья и происхождение каждого решения. Это представители различных сложностей, а не обещание монотонного роста качества.

![Качество и сложность](quality_complexity.svg)

Модель для демонстрации: `{chosen['method']}`, seed `{chosen['seed']}`, `{core.expression(chosen['tree'])}`. Выбрана по validation, а не по тестовому рекорду. [Модель](selected_model.json), [расписание](schedule.csv).

![Предметная визуализация](schedule.svg)

## Вывод и ограничения

Лучший по среднему тестовому качеству метод — `{best_method['method']}`, среднее Q `{best_method['mean']:.3f}`, стандартное отклонение `{best_method['std']:.3f}`. Качество и размер следует рассматривать совместно: штраф 0.2 делает дополнительный узел оправданным только при сопоставимом уменьшении потерь. При равном качестве меньшая программа предпочтительна. Получен работающий и воспроизводимый способ синтеза допустимых назначений с измеренным сравнением ручных правил и разных эволюционных вариантов. Достижение глобального оптимума не доказано.

Ограничения: пять агентов, синтетическое распределение нагрузок, фиксированный порядок очередей, нет перепланирования и зависимостей задач, ограниченный язык, общие для всех запусков тестовые сценарии. Независимость seed поиска характеризует устойчивость алгоритма на этом наборе, а не неопределённость генеральной совокупности. При росте системы потребуется больше сценариев, реальные измерения, проверка чувствительности весов и адаптивное перепланирование. Возможные улучшения: островная модель, сохранение разнообразия семантики и анализ OOD-нагрузок.
{project}
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")


def cli(lab: int, directory: Path) -> None:
    parser = argparse.ArgumentParser(description=f"ЛР{lab}, вариант 20, группа 517, номер 18")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(lab, directory, args.runs, args.budget, args.output)
