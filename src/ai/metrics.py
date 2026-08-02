"""Persistent training metrics."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping

from torch.utils.tensorboard import SummaryWriter


METRIC_COLUMNS = (
    "episode",
    "global_step",
    "phase",
    "epsilon",
    "wins",
    "losses",
    "draws",
    "win_rate",
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


class TrainingMetricsLogger:
    """Write checkpoint metrics to CSV and TensorBoard."""

    def __init__(
        self,
        directory: str | Path = "metrics",
        tensorboard_directory: str | Path = "runs",
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
        total = sum(results.values())
        record: dict[str, float | int | str] = {
            "episode": episode,
            "global_step": global_step,
            "phase": phase,
            "epsilon": epsilon,
            "wins": results["wins"],
            "losses": results["losses"],
            "draws": results["draws"],
            "win_rate": results["wins"] / total if total else 0.0,
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
        self.tensorboard.add_scalar(
            "results/win_rate",
            results["wins"] / total if total else 0.0,
            episode,
        )
        for result, count in results.items():
            self.tensorboard.add_scalar(f"results/{result}", count, episode)
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
        total = sum(results.values())
        self.tensorboard.add_scalar(
            f"evaluation/{opponent}/win_rate",
            results["wins"] / total if total else 0.0,
            episode,
        )
        for result, count in results.items():
            self.tensorboard.add_scalar(
                f"evaluation/{opponent}/{result}",
                count,
                episode,
            )
        self.tensorboard.flush()
