"""Независимые проверки верхней границы и соответствия отчёта данным."""

import csv
import itertools
import json
import unittest
from collections import Counter
from pathlib import Path

from lab_2_weekly_menu import main as lab


class MenuReportTests(unittest.TestCase):
    def test_upper_bound_matches_small_exhaustive_search(self):
        dishes = [lab.Dish(i, f"dish{i}", meal, 500, 100, score)
                  for i, (meal, score) in enumerate(
                      (meal, score) for meal in lab.MEALS for score in (2, 7))]
        choices = [[d.number for d in dishes if d.meal == meal] for meal in lab.MEALS]
        slots = [choices[i % 3] for i in range(6)]
        for minimum_unique in (3, 4, 5, 6):
            brute_force = max(
                sum(dishes[i].score for i in menu)
                for menu in itertools.product(*slots)
                if len(set(menu)) >= minimum_unique and max(Counter(menu).values()) <= 2
            )
            self.assertEqual(lab.utility_upper_bound(dishes, 2, minimum_unique, 2), brute_force)
        with self.assertRaises(ValueError):
            lab.utility_upper_bound(dishes, 2, 7, 2)

    def test_saved_optimality_certificate(self):
        directory = Path(lab.__file__).parent
        certificate = json.loads((directory / "results/optimality.json").read_text(encoding="utf-8"))
        dishes = lab.generate_instance()
        upper_bound = lab.utility_upper_bound(dishes)
        with (directory / "results/best_menu.csv").open(encoding="utf-8") as stream:
            menu = list(csv.DictReader(stream))
        by_name = {dish.name: dish.number for dish in dishes}
        evaluation = lab.evaluate([by_name[row["dish"]] for row in menu], dishes)
        self.assertTrue(evaluation.feasible)
        self.assertEqual(upper_bound, 1936)
        self.assertEqual(evaluation.utility, upper_bound)
        self.assertTrue(certificate["certified_optimal"])
        self.assertEqual(certificate["upper_bound"], upper_bound)

    def test_report_exposes_all_ten_requirements(self):
        path = Path(lab.__file__).parent / "results/REPORT.md"
        report = path.read_text(encoding="utf-8")
        for number in range(1, 11):
            self.assertIn(f"## {number}. ", report)
        for text in ("**Генотип**", "**Фенотип**", "Среднее время, с",
                     "Фитнес-оценок", "![Средняя сходимость", "optimality.json"):
            self.assertIn(text, report)


if __name__ == "__main__":
    unittest.main()
