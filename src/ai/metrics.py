"""Persistent training metrics."""

import csv
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Iterable, Mapping

from torch.utils.tensorboard import SummaryWriter


METRIC_COLUMNS = (
    "episode",
    "global_step",
    "phase",
    "epsilon",
    "win_rate",
    "win_rate_ci_low",
    "win_rate_ci_high",
    "non_loss_rate",
    "non_loss_rate_ci_low",
    "non_loss_rate_ci_high",
    "loss",
    "q_mean",
    "q_std",
    "target_mean",
    "target_std",
    "td_abs_mean",
    "td_std",
    "grad_norm",
    "gradients_finite",
)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson confidence interval for a binary rate."""
    if total <= 0:
        return 0.0, 0.0

    z = 1.959963984540054
    rate = successes / total
    z_squared = z**2
    denominator = 1 + z_squared / total
    center = (rate + z_squared / (2 * total)) / denominator
    margin = (
        z
        * sqrt((rate * (1 - rate) + z_squared / (4 * total)) / total)
        / denominator
    )
    return center - margin, center + margin


def result_rate_metrics(results: Mapping[str, int]) -> dict[str, float]:
    """Summarize match outcomes as win and non-loss rates with 95% CIs."""
    total = sum(results.values())
    wins = results["wins"]
    non_losses = wins + results["draws"]
    win_rate = wins / total if total else 0.0
    non_loss_rate = non_losses / total if total else 0.0
    win_low, win_high = wilson_interval(wins, total)
    non_loss_low, non_loss_high = wilson_interval(non_losses, total)
    return {
        "win_rate": win_rate,
        "win_rate_ci_low": win_low,
        "win_rate_ci_high": win_high,
        "non_loss_rate": non_loss_rate,
        "non_loss_rate_ci_low": non_loss_low,
        "non_loss_rate_ci_high": non_loss_high,
    }


def format_result_rates(results: Mapping[str, int]) -> str:
    """Format win and non-loss results for concise training output."""
    rates = result_rate_metrics(results)
    return (
        f"win rate: {rates['win_rate']:.1%} "
        f"(95% CI {rates['win_rate_ci_low']:.1%}–{rates['win_rate_ci_high']:.1%}) | "
        f"non-loss rate: {rates['non_loss_rate']:.1%} "
        f"(95% CI {rates['non_loss_rate_ci_low']:.1%}–"
        f"{rates['non_loss_rate_ci_high']:.1%})"
    )


def _margin_chart(prefix: str, rate_name: str) -> list[object]:
    """Describe a TensorBoard margin chart for one rate and its 95% CI."""
    return [
        "Margin",
        [
            f"{prefix}/{rate_name}",
            f"{prefix}/{rate_name}_ci_low",
            f"{prefix}/{rate_name}_ci_high",
        ],
    ]


def _custom_scalar_layout(anchor_episodes: Iterable[int]) -> dict[str, dict[str, list[object]]]:
    """Build TensorBoard Custom Scalars charts for all known rate benchmarks."""
    layout: dict[str, dict[str, list[object]]] = {
        "Training": {
            "Win rate (95% CI)": _margin_chart("results", "win_rate"),
            "Non-loss rate (95% CI)": _margin_chart("results", "non_loss_rate"),
        },
        "Evaluation": {},
    }
    opponents = ["random", "dummy", "frozen"]
    opponents.extend(f"anchor_{episode // 1000}k" for episode in anchor_episodes)
    for opponent in opponents:
        prefix = f"evaluation/{opponent}"
        layout["Evaluation"][f"{opponent}: win rate (95% CI)"] = _margin_chart(
            prefix,
            "win_rate",
        )
        layout["Evaluation"][f"{opponent}: non-loss rate (95% CI)"] = _margin_chart(
            prefix,
            "non_loss_rate",
        )
    return layout


class TrainingMetricsLogger:
    """Write checkpoint metrics to CSV and TensorBoard."""

    def __init__(
        self,
        directory: str | Path = "metrics",
        tensorboard_directory: str | Path = "runs",
        anchor_episodes: Iterable[int] = (),
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.directory / "training_metrics.csv"
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=METRIC_COLUMNS)
        self._writer.writeheader()
        self._csv_file.flush()
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_path = Path(tensorboard_directory) / run_id
        self.tensorboard = SummaryWriter(log_dir=self.tensorboard_path)
        self.tensorboard.add_custom_scalars(_custom_scalar_layout(anchor_episodes))
        self._last_phase: str | None = None

    def write_checkpoint(
        self,
        *,
        episode: int,
        global_step: int,
        phase: str,
        epsilon: float,
        results: Mapping[str, int],
        averages: Mapping[str, float],
    ) -> None:
        rates = result_rate_metrics(results)
        record: dict[str, float | int | str] = {
            "episode": episode,
            "global_step": global_step,
            "phase": phase,
            "epsilon": epsilon,
            **rates,
        }
        for metric in (
            "loss",
            "q_mean",
            "q_std",
            "target_mean",
            "target_std",
            "td_abs_mean",
            "td_std",
            "grad_norm",
            "gradients_finite",
        ):
            record[metric] = averages.get(metric, 0.0)
        self._writer.writerow(record)
        self._csv_file.flush()

        self.tensorboard.add_scalar("policy/epsilon", epsilon, episode)
        self._log_rates(prefix="results", episode=episode, rates=rates)
        for name, value in averages.items():
            self.tensorboard.add_scalar(f"train/{name}", value, episode)
        if phase != self._last_phase:
            self.tensorboard.add_text("training/phase", phase, episode)
            self._last_phase = phase
        self.tensorboard.flush()

    def close(self) -> None:
        self._csv_file.close()
        self.tensorboard.close()

    def write_evaluation(
        self,
        *,
        episode: int,
        opponent: str,
        results: Mapping[str, int],
    ) -> None:
        """Record greedy evaluation results without affecting training metrics."""
        self._log_rates(
            prefix=f"evaluation/{opponent}",
            episode=episode,
            rates=result_rate_metrics(results),
        )
        self.tensorboard.flush()

    def _log_rates(
        self,
        *,
        prefix: str,
        episode: int,
        rates: dict[str, float],
    ) -> None:
        """Log rates and their confidence-interval bounds as scalar lines."""
        for name, value in rates.items():
            self.tensorboard.add_scalar(f"{prefix}/{name}", value, episode)
