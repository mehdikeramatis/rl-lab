"""Plot stochastic training returns and deterministic evaluation returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pendulum_comparison.json"
DISPLAY_NAMES = {
    "reinforce": "REINFORCE",
    "reinforce_baseline": "REINFORCE + baseline",
    "actor_critic": "Actor--Critic",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as config_file:
        return json.load(config_file)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float32) / window
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, values)


def plot_summary(axis, x: np.ndarray, values: np.ndarray, label: str, **style: object) -> None:
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    line, = axis.plot(x, mean, label=label, **style)
    axis.fill_between(x, mean - standard_deviation, mean + standard_deviation, color=line.get_color(), alpha=0.18)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, help="Override results_path in the configuration.")
    parser.add_argument("--output", type=Path, help="Override plot_path in the configuration.")
    args = parser.parse_args()
    config = load_config(args.config)
    input_path = args.input or PROJECT_ROOT / config["results_path"]
    output_path = args.output or PROJECT_ROOT / config["plot_path"]
    window = config["training_moving_average_window"]

    with np.load(input_path) as results:
        names = [str(name) for name in results["algorithm_names"]]
        episodes = int(results["training_episodes"])
        if not 1 <= window <= episodes:
            parser.error("training_moving_average_window must be within the training episode range")
        training_x = np.arange(window - 1, episodes)
        evaluation_x = results["evaluation_checkpoints"]
        figure, (training_axis, evaluation_axis) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(9, 8))

        for name in names:
            label = DISPLAY_NAMES.get(name, name.replace("_", " ").title())
            plot_summary(
                training_axis,
                training_x,
                moving_average(results[f"training_{name}"], window),
                label,
                linestyle="-",
            )
            evaluation = results[f"evaluation_{name}"]
            samples = evaluation.transpose(0, 2, 1).reshape(-1, evaluation.shape[1])
            plot_summary(evaluation_axis, evaluation_x, samples, label, linestyle="--", marker="o")

    training_axis.set_title("Stochastic training returns")
    training_axis.set_ylabel(f"Return ({window}-episode moving average)")
    training_axis.legend()
    evaluation_axis.set_title("Deterministic evaluation returns")
    evaluation_axis.set_xlabel("Training episode")
    evaluation_axis.set_ylabel("Return")
    evaluation_axis.legend()
    figure.suptitle("Pendulum-v1: stochastic training vs deterministic evaluation")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    main()
