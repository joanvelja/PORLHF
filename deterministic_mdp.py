import abc
import os
from typing import Union

import numpy as np
import tqdm
from imitation.data.types import TrajectoryWithRew
from imitation.rewards import reward_function, reward_nets
from scipy import sparse


class DeterministicMDP(abc.ABC):
    """
    A deterministic MDP.
    """

    @abc.abstractmethod
    def successor(self, state, action):
        """
        Given a state and action, return the successor state and reward.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self, state, action):
        """
        Given a state and action, return the reward.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def enumerate_states(self):
        """
        Enumerate all states in some consistent order.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def enumerate_actions(self):
        """
        Enumerate all actions in some consistent order.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def encode_state(self, state):
        """
        Encode a state as a string.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def encode_action(self, action):
        """
        Encode an action as a string.
        """
        raise NotImplementedError

    def encode_mdp_params(self):
        """
        Encode the MDP parameters as a string. Used for saving/loading the transition matrix and reward vector for the
        MDP, greatly speeding up the process of computing optimal policies.

        By default, this just returns the name of the class. Override this if you want to save multiple MDPs of the same
        class with different parameters.

        Returns:
            A string encoding the MDP parameters.
        """
        return self.__class__.__name__

    @property
    def states(self):
        if not hasattr(self, "_states"):
            self._states = self.enumerate_states()
        return self._states

    @property
    def actions(self):
        if not hasattr(self, "_actions"):
            self._actions = self.enumerate_actions()
        return self._actions

    def get_state_index(self, state):
        if not hasattr(self, "_state_index"):
            self._state_index = {
                self.encode_state(state): i
                for i, state in tqdm.tqdm(
                    enumerate(self.states), desc="Constructing state index", total=len(self.states)
                )
            }
        return self._state_index[self.encode_state(state)]

    def get_action_index(self, action):
        if not hasattr(self, "_action_index"):
            self._action_index = {self.encode_action(action): i for i, action in enumerate(self.actions)}
        return self._action_index[self.encode_action(action)]

    @property
    def reward_fn_vectorized(self):
        """
        Return a vectorized version of the reward function. Useful for setting reward_fn in a TrajectoryGenerator to
        the true reward function.
        """

        def env_reward_fn(state, action, *args):
            return np.array([self.reward(state, action) for state, action in zip(state, action)])

        return env_reward_fn

    def get_sparse_transition_matrix_and_reward_vector(
        self,
        alt_reward_fn: Union[reward_function.RewardFn, reward_nets.RewardNet] = None,
    ):
        """
        Produce the data structures needed to run value iteration. Specifically, the sparse transition matrix and the
        reward vector. The transition matrix is a sparse matrix of shape (num_states * num_actions, num_states), and the
        reward vector is a vector of length num_states * num_actions.
        Args:
            alt_reward_fn (reward_function.RewardFn or reward_nets.RewardNet): If not None, this reward function will
                be used instead of the one specified in the MDP.
        Returns:
            A tuple of (transition_matrix, reward_vector).
        """
        if not (hasattr(self, "_sparse_transition_matrix") and hasattr(self, "_reward_vector")):
            if not self._load_sparse_transition_matrix_and_reward_vector_from_file():
                self._compute_sparse_transition_matrix_and_reward_vector()
                self._save_sparse_transition_matrix_and_reward_vector_to_file()

        if alt_reward_fn is not None:
            # TODO: might have a batch size issue here; trying to predict for |S| * |A| inputs.
            state_inputs = np.repeat(self.states, len(self.actions), axis=0)
            action_inputs = np.tile(self.actions, (len(self.states)))
            if isinstance(alt_reward_fn, reward_nets.RewardNet):
                rewards = alt_reward_fn.predict(
                    state=state_inputs,
                    action=action_inputs,
                    next_state=state_inputs,
                    done=np.zeros_like(state_inputs, dtype=bool),
                )
            else:
                # Use the reward_function.RewardFn protocol
                rewards = np.array(
                    alt_reward_fn(
                        state_inputs,
                        action_inputs,
                        state_inputs,
                        np.zeros_like(state_inputs, dtype=np.bool_),
                    )
                )
            return self._sparse_transition_matrix, rewards

        return self._sparse_transition_matrix, self._reward_vector

    def _compute_sparse_transition_matrix_and_reward_vector(self):
        """
        Compute the sparse transition matrix and reward vector for this MDP. This is a helper function for
        get_sparse_transition_matrix_and_reward_vector, which caches the results.
        """
        num_states = len(self.states)
        num_actions = len(self.actions)

        transitions = []
        rewards = []

        for state in tqdm.tqdm(self.states, desc="Constructing transition matrix"):
            for action in self.actions:
                successor_state, reward = self.successor(state, action)

                transitions.append(self.get_state_index(successor_state))
                rewards.append(reward)

        transitions = np.array(transitions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        self._reward_vector = rewards

        data = np.ones_like(transitions, dtype=np.float32)
        row_indices = np.arange(num_states * num_actions, dtype=np.int32)
        col_indices = transitions

        transition_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)), shape=(num_states * num_actions, num_states)
        )
        self._sparse_transition_matrix = transition_matrix

    def _save_sparse_transition_matrix_and_reward_vector_to_file(self, env_matrix_dir="env_matrices"):
        """
        Save the sparse transition matrix and reward vector to a file.
        Args:
            env_matrix_dir (str): Path to the directory containing the matrices. Defaults to "env_matrices".
        """
        os.makedirs(env_matrix_dir, exist_ok=True)
        sparse.save_npz(
            os.path.join(env_matrix_dir, f"{self.encode_mdp_params()}_transition.npz"),
            self._sparse_transition_matrix,
        )
        np.save(os.path.join(env_matrix_dir, f"{self.encode_mdp_params()}_reward.npy"), self._reward_vector)

    def _load_sparse_transition_matrix_and_reward_vector_from_file(self, env_matrix_dir="env_matrices"):
        """
        Load the sparse transition matrix and reward vector from a file.
        Args:
            env_matrix_dir (str): Path to the directory containing the matrices. Defaults to "env_matrices".

        Returns:
            Whether the file exists.
        """
        try:
            self._sparse_transition_matrix = sparse.load_npz(
                os.path.join(env_matrix_dir, f"{self.encode_mdp_params()}_transition.npz")
            )
            self._reward_vector = np.load(os.path.join(env_matrix_dir, f"{self.encode_mdp_params()}_reward.npy"))
            return True
        except FileNotFoundError:
            return False

    def rollout_with_policy(
        self,
        policy,
        fixed_horizon=None,
        epsilon=None,
        seed=None,
        render=False,
        logging_callback=None,
    ) -> TrajectoryWithRew:
        """
        Runs a rollout of the environment using the given tabular policy.
        Args:
            policy (value_iteration.TabularPolicy): TabularPolicy for this environment.
            fixed_horizon (int): If not None, the rollout will end after this many steps, regardless of whether the
                agent gets stuck.
            epsilon (float): If not None, the policy will be epsilon-greedy.
            seed (int): If not None, the environment will be seeded with this value.
            render (bool): If True, the environment will be rendered after each step.
            logging_callback (callable): If not None, this function will be called after each step with the current
                state, action, and reward.
        Returns:
            The trajectory generated by the policy as a TrajectoryWithRew object.
        """
        state = self.reset(seed=seed)
        # TODO: Might need to store state as a numerical array instead of human-readable dicts
        states = [state]
        state_indices = [self.get_state_index(state)]
        actions = []
        rewards = []
        done = False

        rng = np.random.default_rng(seed)

        if render:
            self.render()

        while not done or (fixed_horizon is not None and len(actions) < fixed_horizon):
            if epsilon is not None and np.random.random() < epsilon:
                action = np.random.choice(self.actions, rng=rng)
            else:
                action = policy.predict(state)
            next_state, reward, done, _ = self.step(self.get_action_index(action))
            if logging_callback is not None:
                logging_callback(state, action, reward)
            states.append(next_state)
            state_indices.append(self.get_state_index(next_state))
            actions.append(action)
            rewards.append(reward)
            if self.get_state_index(next_state) in state_indices[:-1] and fixed_horizon is None:
                if render:
                    print("Repeated state, ending rollout early.")
                # Policy is deterministic, so if we've been here before, we're in a loop.
                # StealingGridworld only has one-off rewards, so we can just terminate as long as it's the only
                # environment in use.
                break
            state = next_state
            if render:
                self.render()

        if render:
            print(f"Total reward: {sum(rewards)}")

        return TrajectoryWithRew(
            obs=np.array(states, dtype=np.int16),
            acts=np.array(actions, dtype=np.int16),
            rews=np.array(rewards, dtype=float),
            terminal=done,
            infos=None,
        )
