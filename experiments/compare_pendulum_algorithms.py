"""Compare stochastic training returns with periodic deterministic evaluation."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rl_lab.evaluation.evaluate import evaluate
from rl_lab.networks.policy import GaussianPolicy
from rl_lab.networks.value import ValueNetwork
from rl_lab.training import actor_critic, reinforce, reinforce_baseline


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pendulum_comparison.json"


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as config_file:
        return json.load(config_file)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluation_seeds(checkpoint: int, episodes: int, seed_base: int) -> list[int]:
    """Identical initial states for every algorithm at a checkpoint."""
    start = seed_base + checkpoint * episodes
    return list(range(start, start + episodes))


def make_optimizer(parameters, config: dict[str, Any]) -> torch.optim.Adam:
    return torch.optim.Adam(
        parameters,
        lr=config["learning_rate"],
        betas=tuple(config["adam_betas"]),
        eps=config["adam_epsilon"],
    )


def train_reinforce(
    env: gym.Env, policy: GaussianPolicy, config: dict[str, Any], callback: Callable[[int], None]
) -> list[float]:
    optimizer = make_optimizer(policy.parameters(), config)
    _, returns = reinforce.train(
        env,
        policy,
        optimizer,
        config["training_episodes"],
        config["gamma"],
        batch_size=config["batch_size"],
        on_episode_end=callback,
    )
    return returns


def train_reinforce_baseline(
    env: gym.Env, policy: GaussianPolicy, config: dict[str, Any], callback: Callable[[int], None]
) -> list[float]:
    optimizer = make_optimizer(policy.parameters(), config)
    _, returns = reinforce_baseline.train(
        env,
        policy,
        optimizer,
        config["training_episodes"],
        config["gamma"],
        batch_size=config["batch_size"],
        on_episode_end=callback,
    )
    return returns


def train_actor_critic(
    env: gym.Env, policy: GaussianPolicy, config: dict[str, Any], callback: Callable[[int], None]
) -> list[float]:
    policy_random_state = torch.get_rng_state()
    value_network = ValueNetwork(int(np.prod(env.observation_space.shape)), hidden_dim=config["hidden_dim"])
    torch.set_rng_state(policy_random_state)
    policy_optimizer = make_optimizer(policy.parameters(), config)
    value_optimizer = make_optimizer(value_network.parameters(), config)
    _, _, returns = actor_critic.train(
        env,
        policy,
        value_network,
        policy_optimizer,
        value_optimizer,
        config["training_episodes"],
        config["gamma"],
        batch_size=config["batch_size"],
        on_episode_end=callback,
    )
    return returns


TRAINERS = {
    "reinforce": train_reinforce,
    "reinforce_baseline": train_reinforce_baseline,
    "actor_critic": train_actor_critic,
}


def run_algorithm(name: str, training_seed: int, config: dict[str, Any]) -> tuple[list[float], list[list[float]]]:
    seed_everything(training_seed)
    training_env = gym.make(config["environment_id"])
    evaluation_env = gym.make(config["environment_id"])
    training_env.reset(seed=training_seed)
    policy = GaussianPolicy(
        observation_dim=int(np.prod(training_env.observation_space.shape)),
        action_dim=int(np.prod(training_env.action_space.shape)),
        hidden_dim=config["hidden_dim"],
        action_scale=float(training_env.action_space.high[0]),
    )
    deterministic_returns: list[list[float]] = []

    def callback(episode: int) -> None:
        if episode % config["evaluation_interval"] == 0:
            checkpoint = episode // config["evaluation_interval"] - 1
            deterministic_returns.append(
                evaluate(
                    evaluation_env,
                    policy,
                    episodes=config["evaluation_episodes"],
                    seeds=evaluation_seeds(
                        checkpoint,
                        config["evaluation_episodes"],
                        config["evaluation_seed_base"],
                    ),
                )
            )

    try:
        training_returns = TRAINERS[name](training_env, policy, config, callback)
    finally:
        training_env.close()
        evaluation_env.close()
    return training_returns, deterministic_returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Override results_path in the configuration.")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["training_episodes"] < 1:
        parser.error("training_episodes must be positive")
    if config["training_episodes"] % config["evaluation_interval"]:
        parser.error("training_episodes must be divisible by evaluation_interval")

    results: dict[str, np.ndarray] = {}
    for name in TRAINERS:
        training_runs, evaluation_runs = [], []
        for seed in config["seeds"]:
            print(f"Training {name} with seed {seed} ({config['training_episodes']} episodes)")
            training, evaluation = run_algorithm(name, seed, config)
            training_runs.append(training)
            evaluation_runs.append(evaluation)
        results[f"training_{name}"] = np.asarray(training_runs, dtype=np.float32)
        results[f"evaluation_{name}"] = np.asarray(evaluation_runs, dtype=np.float32)

    output = args.output or PROJECT_ROOT / config["results_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = np.arange(
        config["evaluation_interval"], config["training_episodes"] + 1, config["evaluation_interval"]
    )
    np.savez_compressed(
        output,
        algorithm_names=np.asarray(list(TRAINERS)),
        seeds=np.asarray(config["seeds"]),
        training_episodes=np.asarray(config["training_episodes"]),
        evaluation_checkpoints=checkpoints,
        **results,
    )
    print(f"Saved stochastic training and deterministic evaluation returns to {output}")


if __name__ == "__main__":
    main()
