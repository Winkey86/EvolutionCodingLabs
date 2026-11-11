"""Проверки сравнительного графика и равного бюджета методов ЛР1."""

import csv
import random
import statistics
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from lab_1_hgbat import main as lab


class ComparisonTests(unittest.TestCase):
    def test_random_trajectory_matches_cumulative_minimum(self):
        checkpoints = [60, 119, 135]
        result = lab.random_search(135, 18, checkpoints)
        rng = random.Random(18)
        values = [lab.hgbat([rng.uniform(lab.LOWER_BOUND, lab.UPPER_BOUND)
                            for _ in range(lab.DIMENSION)]) for _ in range(135)]
        self.assertEqual(result.trajectory, [min(values[:n]) for n in checkpoints])
        default = lab.random_search(135, 18)
        self.assertEqual((result.value, result.vector), (default.value, default.vector))
        self.assertEqual(default.trajectory, [result.value])

    def test_ga_and_random_use_same_checkpoints(self):
        for budget in (60, 61, 119, 135):
            checkpoints = list(range(60, budget, 59)) + [budget]
            ga = lab.genetic_algorithm(lab.Config("test", 0.02), budget, 18)
            baseline = lab.random_search(budget, 18, checkpoints)
            self.assertEqual(len(ga.trajectory), len(checkpoints))
            self.assertEqual(len(baseline.trajectory), len(checkpoints))
            self.assertEqual(ga.evaluations, baseline.evaluations)
            self.assertEqual(ga.trajectory[-1], ga.value)

    def test_exported_comparison_matches_runs(self):
        directory = Path(lab.__file__).parent / "results"
        with (directory / "comparison.csv").open(encoding="utf-8") as stream:
            comparison = list(csv.DictReader(stream))
        with (directory / "runs.csv").open(encoding="utf-8") as stream:
            runs = list(csv.DictReader(stream))
        for method in {row["method"] for row in runs}:
            points = [row for row in comparison if row["method"] == method]
            self.assertEqual([int(p["evaluations"]) for p in points],
                             list(range(60, 12000, 59)) + [12000])
            self.assertTrue(all(float(b["mean"]) <= float(a["mean"])
                                for a, b in zip(points, points[1:])))
            self.assertTrue(all(float(p["minimum"]) <= float(p["mean"]) <= float(p["maximum"])
                                for p in points))
            values = [float(row["best_value"]) for row in runs if row["method"] == method]
            self.assertAlmostEqual(float(points[-1]["mean"]), statistics.fmean(values))
        root = ET.parse(directory / "comparison.svg").getroot()
        self.assertEqual(len(root.findall("{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}polyline")), 3)


if __name__ == "__main__":
    unittest.main()
