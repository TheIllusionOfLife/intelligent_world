#!/usr/bin/env python3
"""Analyze experiment campaign results and generate a markdown report."""

import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze_campaign(results_dir: Path, output_path: Path) -> dict:
    """Read summary.json, compute statistics, and generate a markdown report.

    Returns a dict with success_rates and fitness_stats for programmatic use.
    """
    summary_path = results_dir / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    # Group by config_id
    by_config: dict[str, list[dict]] = defaultdict(list)
    for entry in data:
        by_config[entry["config_id"]].append(entry)

    # Compute success rates
    success_rates: dict[str, dict] = {}
    for config_id in sorted(by_config):
        entries = by_config[config_id]
        valid = [e for e in entries if not e.get("error")]
        total_valid = len(valid)
        total_solved = sum(1 for e in valid if e["solved"])
        overall_rate = total_solved / total_valid if total_valid > 0 else 0.0

        by_task: dict[str, dict[str, float | int]] = {}
        task_groups: dict[str, list[dict]] = defaultdict(list)
        for e in valid:
            task_groups[e["task"]].append(e)
        for task_name in sorted(task_groups):
            task_entries = task_groups[task_name]
            task_solved = sum(1 for e in task_entries if e["solved"])
            by_task[task_name] = {
                "solved": task_solved,
                "total": len(task_entries),
                "rate": task_solved / len(task_entries) if task_entries else 0.0,
            }

        success_rates[config_id] = {
            "overall": overall_rate,
            "solved": total_solved,
            "total": total_valid,
            "by_task": by_task,
        }

    # Compute fitness stats
    fitness_stats: dict[str, dict] = {}
    for config_id in sorted(by_config):
        entries = by_config[config_id]
        valid = [e for e in entries if not e.get("error")]
        fitnesses = [e["best_fitness"] for e in valid]
        if fitnesses:
            fitness_stats[config_id] = {
                "mean_fitness": statistics.fmean(fitnesses),
                "median_fitness": statistics.median(fitnesses),
                "min_fitness": min(fitnesses),
                "max_fitness": max(fitnesses),
                "stdev_fitness": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            }
        else:
            fitness_stats[config_id] = {
                "mean_fitness": 0.0,
                "median_fitness": 0.0,
                "min_fitness": 0.0,
                "max_fitness": 0.0,
                "stdev_fitness": 0.0,
            }

    # Population diversity analysis (configs C, D)
    diversity_analysis: dict[str, dict] = {}
    for config_id in ["C", "D"]:
        if config_id not in by_config:
            continue
        entries = by_config[config_id]
        valid = [e for e in entries if not e.get("error")]
        convergence_reasons: dict[str, int] = defaultdict(int)
        all_entropy: list[float] = []
        for e in valid:
            reason = e.get("convergence_reason")
            if reason:
                convergence_reasons[reason] += 1
            trajectory = e.get("diversity_trajectory", [])
            for point in trajectory:
                if isinstance(point, dict) and "shannon_entropy" in point:
                    all_entropy.append(point["shannon_entropy"])

        diversity_analysis[config_id] = {
            "convergence_reasons": dict(convergence_reasons),
            "mean_entropy": statistics.fmean(all_entropy) if all_entropy else 0.0,
        }

    # Determine recommendation
    best_config = max(success_rates, key=lambda c: success_rates[c]["overall"])
    best_rate = success_rates[best_config]["overall"]

    # Generate report
    report = _render_report(
        success_rates=success_rates,
        fitness_stats=fitness_stats,
        diversity_analysis=diversity_analysis,
        best_config=best_config,
        best_rate=best_rate,
        total_runs=len(data),
        failed_runs=sum(1 for e in data if e.get("error")),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return {
        "success_rates": success_rates,
        "fitness_stats": fitness_stats,
        "diversity_analysis": diversity_analysis,
    }


_CONFIG_LABELS: dict[str, str] = {
    "A": "Single-agent, no semantic mutations (baseline)",
    "B": "Single-agent, with semantic mutations",
    "C": "Population, no semantic mutations",
    "D": "Population, with semantic mutations",
}


def _render_report(
    *,
    success_rates: dict,
    fitness_stats: dict,
    diversity_analysis: dict,
    best_config: str,
    best_rate: float,
    total_runs: int,
    failed_runs: int,
) -> str:
    lines: list[str] = []
    lines.append("# Experiment Campaign Report\n")

    # Executive summary
    lines.append("## Executive Summary\n")
    lines.append(f"- **Total runs**: {total_runs} ({failed_runs} failed)")
    lines.append(
        f"- **Best configuration**: Config {best_config} "
        f"({_CONFIG_LABELS.get(best_config, 'unknown')})"
    )
    lines.append(f"- **Best overall success rate**: {best_rate:.1%}\n")

    # Recommendation
    lines.append("## Recommendation\n")
    if best_rate >= 0.6:
        lines.append(
            f"**GO** — Config {best_config} achieves {best_rate:.1%} success rate "
            f"across all tasks. The system demonstrates viable evolutionary "
            f"optimization. Recommended next steps: scale to harder tasks, "
            f"tune hyperparameters for underperforming configurations."
        )
    elif best_rate >= 0.3:
        lines.append(
            f"**CONDITIONAL GO** — Config {best_config} achieves {best_rate:.1%} "
            f"success rate. Results are promising but inconsistent. "
            f"Recommended: investigate failure modes and improve mutation "
            f"operators before scaling."
        )
    else:
        lines.append(
            f"**NO-GO** — Best success rate is only {best_rate:.1%}. "
            f"The current evolutionary approach does not reliably solve "
            f"even simple tasks. Fundamental algorithmic changes are needed."
        )
    lines.append("")

    # Success rate table
    lines.append("## Success Rates\n")
    tasks = sorted({t for rates in success_rates.values() for t in rates["by_task"]})
    header = "| Config | Description | Overall |"
    separator = "|--------|-------------|---------|"
    for t in tasks:
        header += f" {t} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)
    for config_id in sorted(success_rates):
        rates = success_rates[config_id]
        label = _CONFIG_LABELS.get(config_id, "")
        row = f"| {config_id} | {label} | {rates['overall']:.1%} |"
        for t in tasks:
            task_info = rates["by_task"].get(t, {})
            task_rate = task_info.get("rate", 0.0)
            solved = task_info.get("solved", 0)
            total = task_info.get("total", 0)
            row += f" {task_rate:.0%} ({solved}/{total}) |"
        lines.append(row)
    lines.append("")

    # Fitness distribution
    lines.append("## Fitness Distribution\n")
    lines.append("| Config | Mean | Median | Min | Max | Stdev |")
    lines.append("|--------|------|--------|-----|-----|-------|")
    for config_id in sorted(fitness_stats):
        stats = fitness_stats[config_id]
        lines.append(
            f"| {config_id} | {stats['mean_fitness']:.3f} "
            f"| {stats['median_fitness']:.3f} "
            f"| {stats['min_fitness']:.3f} "
            f"| {stats['max_fitness']:.3f} "
            f"| {stats['stdev_fitness']:.3f} |"
        )
    lines.append("")

    # Population diversity analysis
    if diversity_analysis:
        lines.append("## Population Diversity Analysis\n")
        for config_id in sorted(diversity_analysis):
            info = diversity_analysis[config_id]
            lines.append(f"### Config {config_id}\n")
            lines.append(f"- **Mean Shannon entropy**: {info['mean_entropy']:.3f}")
            reasons = info.get("convergence_reasons", {})
            if reasons:
                lines.append("- **Convergence reasons**:")
                for reason, count in sorted(reasons.items()):
                    lines.append(f"  - {reason}: {count} runs")
            else:
                lines.append("- No premature convergence observed")
            lines.append("")

    # Key findings
    lines.append("## Key Findings\n")
    sorted_configs = sorted(success_rates, key=lambda c: success_rates[c]["overall"], reverse=True)
    for i, config_id in enumerate(sorted_configs, 1):
        rate = success_rates[config_id]["overall"]
        mean_f = fitness_stats[config_id]["mean_fitness"]
        lines.append(
            f"{i}. **Config {config_id}** ({_CONFIG_LABELS.get(config_id, '')}): "
            f"{rate:.1%} success rate, mean fitness {mean_f:.3f}"
        )

    # Semantic mutation effect
    lines.append("")
    a_rate = success_rates.get("A", {}).get("overall", 0.0)
    b_rate = success_rates.get("B", {}).get("overall", 0.0)
    c_rate = success_rates.get("C", {}).get("overall", 0.0)
    d_rate = success_rates.get("D", {}).get("overall", 0.0)
    lines.append(
        f"- **Semantic mutation effect (single-agent)**: "
        f"{a_rate:.1%} → {b_rate:.1%} ({b_rate - a_rate:+.1%})"
    )
    lines.append(
        f"- **Semantic mutation effect (population)**: "
        f"{c_rate:.1%} → {d_rate:.1%} ({d_rate - c_rate:+.1%})"
    )
    lines.append(
        f"- **Population vs single-agent (no semantic)**: "
        f"{a_rate:.1%} → {c_rate:.1%} ({c_rate - a_rate:+.1%})"
    )
    lines.append(
        f"- **Population vs single-agent (with semantic)**: "
        f"{b_rate:.1%} → {d_rate:.1%} ({d_rate - b_rate:+.1%})"
    )
    lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze campaign results")
    parser.add_argument(
        "--results-dir", default="campaign_results", help="Campaign results directory"
    )
    parser.add_argument(
        "--output",
        default="docs/experiment_campaign_report.md",
        help="Output report path",
    )
    args = parser.parse_args()

    analysis = analyze_campaign(
        results_dir=Path(args.results_dir),
        output_path=Path(args.output),
    )
    print(json.dumps({"success_rates": analysis["success_rates"]}, indent=2))
