"""Проверки численного декодирования и воспроизводимой сборки отчёта."""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import dispatch_core as core
import dispatch_experiments as experiment

DIRECTORY = Path(__file__).resolve().parent


def numeric_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [{key: value if key in ("method", "tree") else int(value) if key == "runs" else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


class DispatchReportTests(unittest.TestCase):
    def test_decoded_priority_and_schedule(self):
        instance = core.load_dataset(DIRECTORY / "data/scenarios.json")["test"][0]
        with (DIRECTORY / "results/decoding_example.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        expected_agents = [i for i,a in enumerate(instance.agents) if instance.tasks[0].skill in a.skills]
        self.assertEqual([int(row["agent"]) for row in rows], expected_agents)
        model = json.loads((DIRECTORY / "results/selected_model.json").read_text(encoding="utf-8"))
        self.assertEqual(model["tree"], "finish")
        for row in rows:
            agent = instance.agents[int(row["agent"])]
            finish = instance.tasks[0].release + instance.tasks[0].work / agent.speed
            self.assertAlmostEqual(float(row["predicted_finish"]), finish)
            self.assertAlmostEqual(float(row["priority"]), finish / 50)
        selected = [row for row in rows if row["selected"] == "True"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected[0]["priority"]), min(float(row["priority"]) for row in rows))

    def test_report_rebuild_keeps_all_explanations(self):
        results = DIRECTORY / "results"
        config = json.loads((results / "config.json").read_text(encoding="utf-8"))
        representatives = json.loads((results / "representatives.json").read_text(encoding="utf-8"))
        chosen = json.loads((results / "selected_model.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            shutil.copyfile(results / "runs.csv", output / "runs.csv")
            experiment.report(DIRECTORY, output, config, numeric_rows(results / "summary.csv"),
                              numeric_rows(results / "baselines.csv"), representatives, chosen)
            rebuilt = (output / "REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(rebuilt, (results / "REPORT.md").read_text(encoding="utf-8"))
        for heading in ("Пошаговый пример декодирования", "Обоснование операторов",
                        "Как читать статистику", "Почему результат приемлем"):
            self.assertIn(heading, rebuilt)

    def test_shared_generator_preserves_lab4_content(self):
        directory = DIRECTORY.parent / "lab_4_robust_dispatch"
        results = directory / "results"
        config = json.loads((results / "config.json").read_text(encoding="utf-8"))
        representatives = json.loads((results / "representatives.json").read_text(encoding="utf-8"))
        chosen = json.loads((results / "selected_model.json").read_text(encoding="utf-8"))
        chosen["tree"] = core.freeze(chosen["tree"])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            experiment.report(directory, output, config, numeric_rows(results / "summary.csv"),
                              numeric_rows(results / "baselines.csv"), representatives, chosen)
            rebuilt = (output / "REPORT.md").read_text(encoding="utf-8")
        original = (results / "REPORT.md").read_text(encoding="utf-8")
        self.assertEqual([line for line in rebuilt.splitlines() if line.strip()],
                         [line for line in original.splitlines() if line.strip()])


if __name__ == "__main__":
    unittest.main()
