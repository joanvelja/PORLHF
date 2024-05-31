import numpy as np
import tqdm
from scipy import sparse

from deterministic_mdp import DeterministicMDP


def run_value_iteration(
    sparse_transitions: sparse.csr_matrix,
    rewards_vector: np.ndarray,
    horizon: int,
    gamma: float,
):
    """
    Runs value iteration on the given MDP.

    Args:
        sparse_transitions: A sparse matrix of shape (num_state_actions, num_states) where
            num_state_actions = num_states * num_actions. The matrix is in CSR format.
        rewards_vector: A vector of shape (num_state_actions,) containing the rewards for each state-action pair.
        horizon: The number of steps to run value iteration for.
        gamma: The discount factor.
    """
    num_state_actions, num_states = sparse_transitions.shape
    num_actions = num_state_actions // num_states

    done_q = np.zeros((num_states, num_actions), dtype=rewards_vector.dtype)
    done_v = np.zeros(num_states, dtype=rewards_vector.dtype)

    optimal_qs = [done_q]
    optimal_values = [done_v]

    for t in tqdm.tqdm(list(reversed(list(range(horizon)))), desc="Value iteration"):
        optimal_qs.insert(
            0,
            (rewards_vector + gamma * sparse_transitions @ optimal_values[0]).reshape((num_states, num_actions)),
        )
        optimal_values.insert(0, optimal_qs[0].max(axis=1))

    return optimal_qs[0], optimal_values[0]


def get_optimal_policy_vector_from_qs(optimal_qs):
    """
    Returns the optimal policy vector given the optimal Q values. The policy vector is numerically indexed, rather than
    using an informative state representation.
    """
    return np.argmax(optimal_qs, axis=1)


def get_optimal_policy(env, gamma=0.99, horizon=None, alt_reward_fn=None):
    """
    Returns the optimal policy for the given environment.

    Args:
        env: The environment to compute the optimal policy for.
        gamma: The discount factor.
        horizon: The number of steps to run value iteration for. If None, defaults to the horizon of the environment.
        alt_reward_fn: An optional function that takes in a state and action and returns the reward for that
            state-action pair. If None, defaults to the reward function of the environment. For reward learning
            experiments.

    Returns:
        A TabularPolicy object, the optimal policy for the given environment (and reward function, if specified).
    """
    optimal_policy, _ = get_optimal_policy_and_values(env, gamma, horizon, alt_reward_fn)
    return optimal_policy


def get_optimal_policy_and_values(env, gamma=0.99, horizon=None, alt_reward_fn=None):
    """
    Returns the optimal policy and values for the given environment.

    Args:
        env: The environment to compute the optimal policy for.
        gamma: The discount factor.
        horizon: The number of steps to run value iteration for. If None, defaults to the horizon of the environment.
        alt_reward_fn: An optional function that takes in a state and action and returns the reward for that
            state-action pair. If None, defaults to the reward function of the environment. For reward learning
            experiments.

    Returns:
        A TabularPolicy object, the optimal policy for the given environment (and reward function, if specified).
    """
    if horizon is None:
        if hasattr(env, "horizon"):
            horizon = env.horizon
        else:
            raise ValueError("Must specify horizon if environment does not have max_steps.")

    transition_matrix, reward_vector = env.get_sparse_transition_matrix_and_reward_vector(alt_reward_fn=alt_reward_fn)
    optimal_qs, optimal_values = run_value_iteration(transition_matrix, reward_vector, horizon=horizon, gamma=gamma)
    optimal_policy_vector = get_optimal_policy_vector_from_qs(optimal_qs)
    return TabularPolicy(env, optimal_policy_vector), optimal_values


class TabularPolicy:
    def __init__(self, env: DeterministicMDP, policy_vector: np.ndarray):
        self.env = env
        self.policy_vector = policy_vector

    def predict(self, state):
        state_index = self.env.get_state_index(state)
        return self.policy_vector[state_index]


class RandomPolicy(TabularPolicy):
    def __init__(self, env, seed=None):
        if seed is not None:
            np.random.seed(seed)
        super().__init__(env, np.random.randint(len(env.actions), size=len(env.states)))
