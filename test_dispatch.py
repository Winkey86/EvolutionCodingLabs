"""Проверки математики, языка деревьев, симулятора и честности бюджета."""

import math
import random
import unittest
from dataclasses import replace
from unittest.mock import patch

import dispatch_core as core


class DispatchTests(unittest.TestCase):
    def test_hand_calculated_schedule(self):
        instance = core.Scenario(1, (core.Task(0, 2, 1, 0), core.Task(0, 2, 3, 0)),
                                 (core.Agent(1, 1, (0,)),), ((1,), (1,)))
        result = core.simulate("finish", instance, trace=True)
        self.assertEqual(result["makespan"], 4)
        self.assertEqual(result["mean_tardiness"], 1)
        self.assertEqual(result["cost"], 4)
        self.assertAlmostEqual(result["loss"], 8.2)
        self.assertEqual([r["start"] for r in result["schedule"]], [0, 2])

    def test_no_actual_duration_leaks_into_decision(self):
        instance = core.scenario(44, 24, 0)
        changed = replace(instance, multipliers=tuple(tuple(10.0 if j == 0 else 0.1
                          for j in range(5)) for _ in instance.tasks))
        original = core.simulate("finish", instance, trace=True)
        perturbed = core.simulate("finish", changed, trace=True)
        self.assertEqual([r["agent"] for r in original["schedule"]],
                         [r["agent"] for r in perturbed["schedule"]])
        self.assertNotEqual(original["loss"], perturbed["loss"])

    def test_variation_preserves_language_and_depth(self):
        rng = random.Random(18)
        for max_depth in (1, 2, 4):
            for _ in range(120):
                left = core.random_tree(rng, max_depth)
                right = core.random_tree(rng, max_depth)
                for child in (core.vary(left, right, rng, max_depth),
                              core.mutate(left, rng, max_depth)):
                    core.validate(child, max_depth)
                    self.assertTrue(math.isfinite(core.compile_rule(child)((0.2,)*6)))
        with self.assertRaises(ValueError):
            core.validate(("exec", "finish", "cost"))

    def test_protected_arithmetic(self):
        tree = ("div", "finish", ("sub", "cost", "cost"))
        self.assertAlmostEqual(core.compile_rule(tree)((0.2,)*6), 0.2)
        self.assertEqual(core.compile_rule(("mul", 2.0, 2.0))((0,)*6), 4)

    def test_feasible_schedule(self):
        instance = core.scenario(20, 48, 0.35)
        schedule = core.simulate("finish", instance, trace=True)["schedule"]
        self.assertEqual(len(schedule), 48)
        for row in schedule:
            self.assertIn(row["skill"], instance.agents[row["agent"]].skills)
            self.assertGreaterEqual(row["start"], row["release"])
            self.assertGreater(row["duration"], 0)
        for agent in range(5):
            rows = [r for r in schedule if r["agent"] == agent]
            self.assertTrue(all(b["start"] >= a["end"] for a,b in zip(rows, rows[1:])))

    def test_budget_reproducibility_and_test_isolation(self):
        dataset = core.make_dataset(8, 0.35, 123)
        for memetic in (False, True):
            calls = {k: 0 for k in dataset}
            original = core.assess
            def counted(tree, scenarios, risk=0):
                for name, items in dataset.items():
                    if scenarios is items:
                        calls[name] += 1
                return original(tree, scenarios, risk)
            with patch.object(core, "assess", counted):
                result = core.evolve(dataset, 18, 45, 2, 0.2, 0.5, memetic, 8)
            self.assertEqual(calls["train"], 45)
            self.assertEqual(calls["validation"], result["validation_evaluations"])
            self.assertEqual(calls["test"], 0)
            self.assertEqual(len(result["history"]), 45)
            again = core.evolve(dataset, 18, 45, 2, 0.2, 0.5, memetic, 8)
            self.assertEqual(result, again)
        splits = [{s.seed for s in dataset[name]} for name in dataset]
        self.assertTrue(all(not a & b for i,a in enumerate(splits) for b in splits[i+1:]))


if __name__ == "__main__":
    unittest.main()
