"""Efficient JAX REINFORCE training for Gymnasium's Pendulum-v1.

Install the runtime dependencies first, for example:
    python -m pip install "jax" gymnasium matplotlib

On Apple Silicon, JAX uses the CPU by default.  If the Apple Metal plug-in is
installed, JAX will select its available Metal device automatically.
"""

from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


ArrayTree = dict[str, Any]


def init_params(key: jax.Array, observation_dim: int, action_dim: int, hidden_dim: int) -> ArrayTree:
    """Initialise the same two-layer tanh Gaussian policy as the Torch version."""
    keys = jax.random.split(key, 3)

    def glorot(k: jax.Array, fan_in: int, fan_out: int) -> jax.Array:
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(k, (fan_in, fan_out), minval=-limit, maxval=limit)

    return {
        "w1": glorot(keys[0], observation_dim, hidden_dim),
        "b1": jnp.zeros(hidden_dim),
        "w2": glorot(keys[1], hidden_dim, hidden_dim),
        "b2": jnp.zeros(hidden_dim),
        "w_mean": glorot(keys[2], hidden_dim, action_dim),
        "b_mean": jnp.zeros(action_dim),
        "log_std": jnp.zeros(action_dim),
    }


def policy_mean(params: ArrayTree, observation: jax.Array) -> jax.Array:
    hidden = jnp.tanh(observation @ params["w1"] + params["b1"])
    hidden = jnp.tanh(hidden @ params["w2"] + params["b2"])
    return hidden @ params["w_mean"] + params["b_mean"]


@jax.jit
def sample_action(params: ArrayTree, observation: jax.Array, key: jax.Array, action_scale: float) -> tuple[jax.Array, jax.Array]:
    raw_action = policy_mean(params, observation) + jnp.exp(params["log_std"]) * jax.random.normal(
        key, params["log_std"].shape
    )
    return jnp.tanh(raw_action) * action_scale, raw_action


@jax.jit
def discounted_returns(rewards: jax.Array, gamma: float) -> jax.Array:
    """Compute reward-to-go with a fused reverse scan."""
    _, returns = jax.lax.scan(
        lambda running, reward: (reward + gamma * running, reward + gamma * running),
        jnp.array(0.0, dtype=rewards.dtype),
        rewards,
        reverse=True,
    )
    return returns


def log_prob_of_raw_action(params: ArrayTree, observations: jax.Array, raw_actions: jax.Array, action_scale: float) -> jax.Array:
    mean = jax.vmap(lambda obs: policy_mean(params, obs))(observations)
    log_std = params["log_std"]
    std = jnp.exp(log_std)
    normal_log_prob = -0.5 * (((raw_actions - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    # Change-of-variables correction for action = action_scale * tanh(raw_action).
    tanh_correction = 2.0 * (jnp.log(2.0) - raw_actions - jax.nn.softplus(-2.0 * raw_actions))
    return jnp.sum(normal_log_prob - tanh_correction - jnp.log(action_scale), axis=-1)


@jax.jit
def adam_reinforce_update(
    params: ArrayTree,
    first_moment: ArrayTree,
    second_moment: ArrayTree,
    step: jax.Array,
    observations: jax.Array,
    raw_actions: jax.Array,
    rewards: jax.Array,
    gamma: float,
    learning_rate: float,
    action_scale: float,
) -> tuple[ArrayTree, ArrayTree, ArrayTree, jax.Array, jax.Array]:
    returns = discounted_returns(rewards, gamma)
    advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

    def loss_fn(p: ArrayTree) -> jax.Array:
        return -jnp.mean(log_prob_of_raw_action(p, observations, raw_actions, action_scale) * advantages)

    loss, gradients = jax.value_and_grad(loss_fn)(params)
    step = step + 1
    beta1, beta2 = 0.9, 0.999
    first_moment = jax.tree.map(lambda m, g: beta1 * m + (1.0 - beta1) * g, first_moment, gradients)
    second_moment = jax.tree.map(lambda v, g: beta2 * v + (1.0 - beta2) * g * g, second_moment, gradients)
    bias1 = 1.0 - beta1**step
    bias2 = 1.0 - beta2**step
    params = jax.tree.map(
        lambda p, m, v: p - learning_rate * (m / bias1) / (jnp.sqrt(v / bias2) + 1e-8),
        params,
        first_moment,
        second_moment,
    )
    return params, first_moment, second_moment, step, loss


def train(episodes: int, gamma: float, learning_rate: float, hidden_dim: int, seed: int) -> tuple[list[float], list[float]]:
    env = gym.make("Pendulum-v1")
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    action_scale = float(env.action_space.high[0])
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, observation_dim, action_dim, hidden_dim)
    first_moment = jax.tree.map(jnp.zeros_like, params)
    second_moment = jax.tree.map(jnp.zeros_like, params)
    step = jnp.array(0, dtype=jnp.int32)
    losses, episode_returns = [], []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            observations, raw_actions, rewards = [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                observations.append(observation)
                key, action_key = jax.random.split(key)
                action, raw_action = sample_action(params, jnp.asarray(observation, dtype=jnp.float32), action_key, action_scale)
                observation, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
                raw_actions.append(np.asarray(raw_action))
                rewards.append(reward)

            rewards_array = np.asarray(rewards, dtype=np.float32)
            params, first_moment, second_moment, step, loss = adam_reinforce_update(
                params,
                first_moment,
                second_moment,
                step,
                jnp.asarray(observations, dtype=jnp.float32),
                jnp.asarray(raw_actions, dtype=jnp.float32),
                jnp.asarray(rewards_array),
                gamma,
                learning_rate,
                action_scale,
            )
            losses.append(float(loss))
            episode_returns.append(float(rewards_array.sum()))
            print(f"Episode {episode + 1:3d}/{episodes}: return={episode_returns[-1]:8.2f}, loss={losses[-1]:.4f}")
    finally:
        env.close()
    return losses, episode_returns


def plot_returns(returns: list[float], window: int, save_path: str | None) -> None:
    values = np.asarray(returns)
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    plt.plot(values, label="Return")
    plt.plot(np.arange(window - 1, len(values)), moving_average, label=f"{window}-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("JAX REINFORCE Training Return — Pendulum-v1")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save", help="Save the plot instead of displaying it.")
    args = parser.parse_args()
    if args.episodes < args.window:
        parser.error("--episodes must be at least --window")
    _, returns = train(args.episodes, args.gamma, args.learning_rate, args.hidden_dim, args.seed)
    plot_returns(returns, args.window, args.save)


if __name__ == "__main__":
    main()
