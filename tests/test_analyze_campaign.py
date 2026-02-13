import json
from pathlib import Path

import pytest

from scripts.analyze_campaign import analyze_campaign


class TestAnalyzeCampaign:
    def _make_summary(self, tmp_path: Path) -> Path:
        """Create a fixture summary.json with known data."""
        summary = [
            # Config A: single_agent, no semantic — 2 solved, 1 unsolved per task
            *[
                {
                    "config_id": "A",
                    "task": task,
                    "seed": seed,
                    "solved": solved,
                    "best_fitness": fitness,
                    "best_train_pass_ratio": 0.95 if solved else 0.5,
                    "best_hidden_pass_ratio": 0.9 if solved else 0.4,
                    "total_steps": 100,
                    "final_energy": 0.3,
                    "error": None,
                }
                for task in ["two_sum_sorted", "run_length_encode", "slugify"]
                for seed, solved, fitness in [
                    (0, True, 0.85),
                    (7, True, 0.78),
                    (13, False, 0.45),
                ]
            ],
            # Config B: single_agent, semantic — all solved
            *[
                {
                    "config_id": "B",
                    "task": task,
                    "seed": seed,
                    "solved": True,
                    "best_fitness": 0.9,
                    "best_train_pass_ratio": 0.98,
                    "best_hidden_pass_ratio": 0.95,
                    "total_steps": 80,
                    "final_energy": 0.5,
                    "error": None,
                }
                for task in ["two_sum_sorted", "run_length_encode", "slugify"]
                for seed in [0, 7, 13]
            ],
            # Config C: population, no semantic — 1 solved per task
            *[
                {
                    "config_id": "C",
                    "task": task,
                    "seed": seed,
                    "solved": solved,
                    "best_fitness": fitness,
                    "best_train_pass_ratio": 0.92 if solved else 0.3,
                    "best_hidden_pass_ratio": 0.88 if solved else 0.2,
                    "total_generations": 30,
                    "convergence_reason": None if solved else "low_diversity",
                    "diversity_trajectory": [
                        {"shannon_entropy": 0.8, "structural_diversity_ratio": 0.7}
                    ],
                    "error": None,
                }
                for task in ["two_sum_sorted", "run_length_encode", "slugify"]
                for seed, solved, fitness in [
                    (0, True, 0.82),
                    (7, False, 0.4),
                    (13, False, 0.35),
                ]
            ],
            # Config D: population, semantic — 2 solved per task
            *[
                {
                    "config_id": "D",
                    "task": task,
                    "seed": seed,
                    "solved": solved,
                    "best_fitness": fitness,
                    "best_train_pass_ratio": 0.94 if solved else 0.5,
                    "best_hidden_pass_ratio": 0.91 if solved else 0.4,
                    "total_generations": 25,
                    "convergence_reason": None if solved else "stalled_low_entropy",
                    "diversity_trajectory": [
                        {"shannon_entropy": 0.85, "structural_diversity_ratio": 0.75}
                    ],
                    "error": None,
                }
                for task in ["two_sum_sorted", "run_length_encode", "slugify"]
                for seed, solved, fitness in [
                    (0, True, 0.88),
                    (7, True, 0.83),
                    (13, False, 0.5),
                ]
            ],
        ]
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary_path

    def test_returns_analysis_dict(self, tmp_path: Path) -> None:
        self._make_summary(tmp_path)
        report_path = tmp_path / "report.md"
        analysis = analyze_campaign(results_dir=tmp_path, output_path=report_path)

        assert "success_rates" in analysis
        assert "fitness_stats" in analysis

    def test_success_rates_by_config(self, tmp_path: Path) -> None:
        self._make_summary(tmp_path)
        report_path = tmp_path / "report.md"
        analysis = analyze_campaign(results_dir=tmp_path, output_path=report_path)

        rates = analysis["success_rates"]
        # Config A: 2/3 solved per task, 6/9 overall = 66.7%
        assert rates["A"]["overall"] == pytest.approx(6 / 9, abs=0.01)
        # Config B: all solved = 100%
        assert rates["B"]["overall"] == pytest.approx(1.0)
        # Config C: 1/3 per task, 3/9 = 33.3%
        assert rates["C"]["overall"] == pytest.approx(3 / 9, abs=0.01)
        # Config D: 2/3 per task, 6/9 = 66.7%
        assert rates["D"]["overall"] == pytest.approx(6 / 9, abs=0.01)

    def test_success_rates_by_task(self, tmp_path: Path) -> None:
        self._make_summary(tmp_path)
        report_path = tmp_path / "report.md"
        analysis = analyze_campaign(results_dir=tmp_path, output_path=report_path)

        rates = analysis["success_rates"]
        # Each config has same pattern across all tasks in fixture
        for config_id in ["A", "B", "C", "D"]:
            for task in ["two_sum_sorted", "run_length_encode", "slugify"]:
                assert task in rates[config_id]["by_task"]

    def test_fitness_stats_computed(self, tmp_path: Path) -> None:
        self._make_summary(tmp_path)
        report_path = tmp_path / "report.md"
        analysis = analyze_campaign(results_dir=tmp_path, output_path=report_path)

        stats = analysis["fitness_stats"]
        for config_id in ["A", "B", "C", "D"]:
            assert "mean_fitness" in stats[config_id]
            assert "median_fitness" in stats[config_id]

    def test_generates_report_file(self, tmp_path: Path) -> None:
        self._make_summary(tmp_path)
        report_path = tmp_path / "report.md"
        analyze_campaign(results_dir=tmp_path, output_path=report_path)

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "# Experiment Campaign Report" in content
        assert "Success Rate" in content
        assert "Config A" in content

    def test_report_contains_population_diversity_analysis(self, tmp_path: Path) -> None:
        self._make_summary(tmp_path)
        report_path = tmp_path / "report.md"
        analyze_campaign(results_dir=tmp_path, output_path=report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "Diversity" in content or "diversity" in content

    def test_report_contains_recommendation(self, tmp_path: Path) -> None:
        self._make_summary(tmp_path)
        report_path = tmp_path / "report.md"
        analyze_campaign(results_dir=tmp_path, output_path=report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "Recommendation" in content or "recommendation" in content
