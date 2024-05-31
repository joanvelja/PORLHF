import abc
import math
import itertools
import pickle
import re
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch as th
from torch import nn
from torch.utils import data as data_th
from gymnasium import spaces
from imitation.algorithms import base, preference_comparisons
from imitation.algorithms.preference_comparisons import (
    Fragmenter, PreferenceGatherer, PreferenceDataset,
    preference_collate_fn, TrajectoryGenerator, _make_reward_trainer,)
from imitation.data import rollout, types
from imitation.data.types import (
    TrajectoryPair,
    TrajectoryWithRew,
    TrajectoryWithRewPair,
    Transitions,
)
from imitation.regularization import regularizers
from imitation.rewards.reward_nets import RewardNet, RewardEnsemble, AddSTDRewardWrapper
from imitation.util import logger as imit_logger
from imitation.util import networks, util
from scipy import special
from tqdm.auto import tqdm
from stable_baselines3.common import base_class, type_aliases, utils, vec_env
import value_iteration





class DeterministicMDPTrajGenerator(preference_comparisons.TrajectoryGenerator):
    """
    A trajectory generator for a deterministic MDP that can be solved exactly using value iteration.
    """

    def __init__(self, reward_fn, env, rng, vi_gamma=0.99, max_vi_steps=None, epsilon=None, custom_logger=None):
        super().__init__(custom_logger=custom_logger)

        self.reward_fn = reward_fn
        self.env = env
        self.rng = rng
        self.vi_gamma = vi_gamma
        self.epsilon = epsilon

        if max_vi_steps is None:
            if hasattr(self.env, "horizon"):
                max_vi_steps = self.env.horizon
            else:
                raise ValueError("max_vi_steps must be specified if env does not have a horizon attribute")
        self.max_vi_steps = max_vi_steps

        # TODO: Can I just pass `rng` to np.random.seed like this?
        self.policy = value_iteration.RandomPolicy(self.env, self.rng)

    def sample(self, steps):
        """
        Generate trajectories with total number of steps equal to `steps`.
        """
        trajectories = []
        total_steps = 0
        while total_steps < steps:
            trajectory = self.env.rollout_with_policy(
                self.policy,
                fixed_horizon=self.max_vi_steps,
                epsilon=self.epsilon,
            )
            trajectories.append(trajectory)
            total_steps += len(trajectory)
        return trajectories

    def train(self, steps):
        """
        Find the optimal policy using value iteration under the given reward function.
        Overrides the train method as required for imitation.preference_comparisons.
        """
        vi_steps = min(steps, self.max_vi_steps)
        self.policy = value_iteration.get_optimal_policy(
            self.env, gamma=self.vi_gamma, horizon=vi_steps, alt_reward_fn=self.reward_fn
        )


class NonImageCnnRewardNet(RewardNet):
    """
    A CNN reward network that does not make assumptions about the input being an image. In particular, it does not
    apply standard image preprocessing (e.g. normalization) to the input.

    Because the code that requires the input to be an image occurs in the __init__ method of CnnRewardNet (which is a
    more natural choice for superclass), we actually need to only subclass RewardNet and reimplement some functionality
    from CnnRewardNet.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, normalize_images=False)

        input_size = observation_space.shape[0]
        output_size = action_space.n

        full_build_cnn_kwargs: Dict[str, Any] = {
            "hid_channels": (32, 32),
            **kwargs,
            # we do not want the values below to be overridden
            "in_channels": input_size,
            "out_size": output_size,
            "squeeze_output": output_size == 1,
        }

        self.cnn = networks.build_cnn(**full_build_cnn_kwargs)

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Override standard input preprocess to bypass image preprocessing. Only lifts inputs to tensors.
        """
        state_th = util.safe_to_tensor(state).to(self.device).float()
        action_th = util.safe_to_tensor(action).to(self.device)
        next_state_th = util.safe_to_tensor(next_state).to(self.device)
        done_th = util.safe_to_tensor(done).to(self.device)

        return state_th, action_th, next_state_th, done_th

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Computes rewardNet value on input state and action. Ignores next_state, and done flag.

        Args:
            state: current state.
            action: current action.
            next_state: next state.
            done: flag for whether the episode is over.

        Returns:
            th.Tensor: reward of the transition.
        """
        outputs = self.cnn(state)
        # for discrete action spaces, action should be passed to forward as a one-hot vector.
        # If action is not 1-hot vector, then we need to convert it to one-hot vector
        # TODO: Chase down where this should actually be happening upstream of here
        if action.ndim == 1:
            rewards = outputs[th.arange(action.shape[0]), action.int()]
        else:
            rewards = th.sum(outputs * action, dim=1)

        return rewards


class SyntheticValueGatherer(preference_comparisons.SyntheticGatherer):
    """
    Computes synthetic preferences by a weighted combination of ground-truth environment rewards (present in the
    trajectory fragment) and ground-truth optimal value at the end of the trajectory fragment (computed using value
    iteration).
    """

    def __init__(
        self,
        env,
        temperature=1.0,
        rlhf_gamma=1.0,
        sample=True,
        rng=None,
        threshold=50,
        vi_horizon=None,
        vi_gamma=0.99,
        value_coeff=0.1,  # weight of value in synthetic reward
        custom_logger=None,
    ):
        super().__init__(temperature, rlhf_gamma, sample, rng, threshold, custom_logger)

        self.env = env
        self.vi_horizon = vi_horizon
        self.vi_gamma = vi_gamma

        self.value_coeff = value_coeff

        _, self.values = value_iteration.get_optimal_policy_and_values(
            self.env, gamma=self.vi_gamma, horizon=self.vi_horizon
        )

    def _get_value(self, state):
        return self.values[self.env.get_state_index(state)]

    def _augment_fragment_pair_with_value(self, fragment_pair):
        new_fragments = []
        for fragment in fragment_pair:
            final_state = fragment.obs[-1]
            value = self._get_value(final_state)
            new_rews = np.copy(fragment.rews)
            new_rews[-1] += self.value_coeff * value
            new_fragments.append(
                TrajectoryWithRew(fragment.obs, fragment.acts, fragment.infos, fragment.terminal, new_rews)
            )
        return tuple(new_fragments)

    def __call__(self, fragment_pairs):
        fragment_pairs = [self._augment_fragment_pair_with_value(fp) for fp in fragment_pairs]
        return super().__call__(fragment_pairs)


class ScalarFeedbackDataset(data_th.Dataset):
    """A PyTorch Dataset for scalar reward feedback.

    Each item is a tuple consisting of a trajectory fragment and a scalar reward (given by a FeedbackGatherer; not
    necessarily the ground truth environment rewards).

    This dataset is meant to be generated piece by piece during the training process, which is why data can be added
    via the .push() method.
    """

    def __init__(self, max_size=None):
        self.fragments = []
        self.max_size = max_size
        self.reward_labels = np.array([])

    def push(self, fragments, reward_labels):
        self.fragments.extend(fragments)
        self.reward_labels = np.concatenate((self.reward_labels, reward_labels))

        # Evict old samples if the dataset is at max capacity
        if self.max_size is not None:
            extra = len(self.reward_labels) - self.max_size
            if extra > 0:
                self.fragments = self.fragments[extra:]
                self.reward_labels = self.reward_labels[extra:]

    def __getitem__(self, index):
        return self.fragments[index], self.reward_labels[index]

    def __len__(self):
        assert len(self.fragments) == len(self.reward_labels)
        return len(self.reward_labels)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


class RandomSingleFragmenter(preference_comparisons.RandomFragmenter):
    """Fragmenter that samples single fragments rather than fragment pairs.

    Intended to be used for non-comparison-based feedback, such as scalar reward feedback.
    """

    def __call__(self, trajectories, fragment_length, num_fragments):
        fragment_pairs = super().__call__(trajectories, fragment_length, int(np.ceil(num_fragments // 2)))
        # fragment_pairs is a list of (fragment, fragment) tuples. We want to flatten this into a list of fragments.
        return list(itertools.chain.from_iterable(fragment_pairs))


class ScalarFeedbackModel(nn.Module):
    """Class to convert a fragment's reward into a scalar feedback label."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fragments):
        """Computes scalar feedback labels for the given fragments."""
        reward_predictions = []
        for fragment in fragments:
            transitions = rollout.flatten_trajectories([fragment])
            preprocessed = self.model.preprocess(
                transitions.obs,
                transitions.acts,
                transitions.next_obs,
                transitions.dones,
            )
            reward_prediction_per_step = self.model(*preprocessed)
            assert reward_prediction_per_step.shape == (len(transitions.obs),)
            reward_prediction = th.sum(reward_prediction_per_step, dim=0)
            reward_predictions.append(reward_prediction)
        return th.stack(reward_predictions)


class ScalarFeedbackGatherer(abc.ABC):
    """Base class for gathering scalar feedback for a trajectory fragment."""

    def __init__(self, rng=None, custom_logger=None):
        del rng  # unused
        self.logger = custom_logger or imit_logger.configure()

    @abc.abstractmethod
    def __call__(self, fragments):
        """Gathers the scalar feedback for the given fragments.

        See preference_comparisons.PreferenceGatherer for more details.
        """


class SyntheticScalarFeedbackGatherer(ScalarFeedbackGatherer):
    """Computes synthetic scalar feedback using ground-truth environment rewards."""

    # TODO: This is a placeholder for a more sophisticated synthetic feedback gatherer.

    def __call__(self, fragments, limits=None):
        return [np.sum(fragment.rews) for fragment in fragments]


class NoisyObservationGathererWrapper(ScalarFeedbackGatherer):
    """Wraps a scalar feedback gatherer to handle the feedback giver seeing a noisy observation of state, rather than
    the true environment state.

    Current implementation only supports deterministic observation noise (such as occlusion). Later implementations
    will pass a random seed to the observation function to support stochastic observation noise. For now, a stochastic
    observation function will not fail, but will not be seed-able, so results will not be reproducible.
    """

    def __init__(self, gatherer: ScalarFeedbackGatherer, observe_fn):
        self.wrapped_gatherer = gatherer
        self.observe_fn = observe_fn

    def __getattr__(self, name):
        return getattr(self.wrapped_gatherer, name)

    def __call__(self, fragments):
        noisy_fragments = [self.observe_fn(fragment) for fragment in fragments]
        return self.wrapped_gatherer(noisy_fragments)


class PreferenceComparisonNoisyObservationGathererWrapper(preference_comparisons.PreferenceGatherer):
    """
    Wraps a preference comparison gatherer to handle the feedback giver seeing noisy observations of state pairs,
    rather than the true environment state pairs. It processes pairs of fragments through a provided observation
    function before passing them to the wrapped preference gatherer.

    Args:
        gatherer (PreferenceGatherer): The preference comparison gatherer to wrap.
        observe_fn (Callable[[Trajectory], Trajectory]): A function that takes a trajectory and returns a modified
            trajectory representing a noisy observation of the original.
    """

    def __init__(self, gatherer, observe_fn, limits=None):
        self.wrapped_gatherer = gatherer
        self.observe_fn = observe_fn

    def __getattr__(self, name, limits=None):
        """
        Delegate attribute access to the wrapped gatherer, unless the attribute is overridden in this wrapper.
        """
        return getattr(self.wrapped_gatherer, name)

    def __call__(self, fragment_pairs, limits=None):
        """
        Apply the observation function to each fragment in the pairs, then pass the noisy pairs to the wrapped gatherer.

        Args:
            fragment_pairs (list of tuple(Trajectory, Trajectory)): A list of pairs of trajectories.

        Returns:
            np.ndarray: The preference comparisons results (e.g., probabilities or binary decisions) for the noisy pairs.
        """
        #noisy_fragment_pairs = [self.observe_fn((frag1, frag2)) for frag1, frag2 in fragment_pairs]
        # TODO - This is a hack to get around the fact that the limits are not being passed in correctly
        pairs = [(frag1, frag2) for frag1, frag2 in fragment_pairs]
        if limits is not None:
            noisy_fragment_pairs = [self.observe_fn(pair, l) for pair, l in zip(pairs, limits)]
        else:
            noisy_fragment_pairs = [self.observe_fn(pair) for pair in pairs]

        return self.wrapped_gatherer(noisy_fragment_pairs)

class ObservationFunction(abc.ABC):
    """Abstract class for functions that take an observation and return a new observation."""

    @abc.abstractmethod
    def __call__(self, fragment):
        """Returns a new fragment with observations, actions, and rewards filtered through an observation function.

        Args:
            fragment: a TrajectoryWithRew object.

        Returns:
            A new TrajectoryWithRew object with the same infos and terminal flag, but with the observations, actions,
            and rewards filtered through the observation function.
        """


class ScalarFeedbackRewardTrainer(abc.ABC):
    """Base class for training a reward model using scalar feedback."""

    def __init__(self, feedback_model, custom_logger=None):
        self._feedback_model = feedback_model
        self._logger = custom_logger or imit_logger.configure()

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, custom_logger):
        self._logger = custom_logger

    def train(self, dataset, epoch_multiplier=1.0):
        """Trains the reward model using the given dataset (a batch of fragments and feedback).

        Args:
            dataset: a Dataset object containing the feedback data.
            epoch_multiplier: a multiplier for the number of epochs to train for.
        """
        with networks.training(self._feedback_model.model):
            self._train(dataset, epoch_multiplier)

    @abc.abstractmethod
    def _train(self, dataset, epoch_multiplier):
        """Train the reward model; see ``train`` for details."""


class MSERewardLoss(preference_comparisons.RewardLoss):
    """Compute the MSE between the given rewards and the feedback labels."""

    def forward(self, fragments, feedback_labels, feedback_model):
        """Computes the MSE between the given rewards and the feedback labels."""
        reward_predictions = feedback_model(fragments)
        feedback_th = th.as_tensor(feedback_labels, dtype=th.float32, device=reward_predictions.device)
        return th.mean((reward_predictions - feedback_th) ** 2)


class BasicScalarFeedbackRewardTrainer(ScalarFeedbackRewardTrainer):
    """Train a basic reward model from scalar feedback."""

    def __init__(
        self,
        feedback_model,
        loss,
        rng,
        batch_size=32,
        minibatch_size=None,
        epochs=1,
        lr=1e-3,
        custom_logger=None,
    ):
        super().__init__(feedback_model, custom_logger=custom_logger)
        self.loss = loss
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("batch_size must be divisible by minibatch_size")
        self.epochs = epochs
        self.optim = th.optim.AdamW(self._feedback_model.parameters(), lr=lr)
        self.rng = rng
        self.lr = lr

    def _make_data_loader(self, dataset):
        return data_th.DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

    def _train(self, dataset, epoch_multiplier=1.0):
        dataloader = self._make_data_loader(dataset)
        epochs = np.round(self.epochs * epoch_multiplier).astype(int)
        assert epochs > 0, "Must train for at least one epoch."
        with self.logger.accumulate_means("reward"):
            for epoch_num in tqdm(range(epochs), desc="Training reward model"):
                with self.logger.add_key_prefix(f"epoch-{epoch_num}"):
                    train_loss = 0.0
                    accumulated_size = 0
                    self.optim.zero_grad()
                    for fragments, feedback in dataloader:
                        with self.logger.add_key_prefix("train"):
                            loss = self._training_inner_loop(fragments, np.array(feedback))
                            loss *= len(fragments) / self.batch_size  # rescale loss to account for minibatching
                        train_loss += loss.item()
                        loss.backward()
                        accumulated_size += len(fragments)
                        if accumulated_size >= self.batch_size:
                            self.optim.step()
                            self.optim.zero_grad()
                            accumulated_size = 0
                    if accumulated_size > 0:
                        self.optim.step()  # if there remains an incomplete batch

        # after training all the epochs,
        # record also the final value in a separate key for easy access.
        keys = list(self.logger.name_to_value.keys())
        outer_prefix = self.logger.get_accumulate_prefixes()
        for key in keys:
            base_path = f"{outer_prefix}reward/"  # existing prefix + accum_means ctx
            epoch_path = f"mean/{base_path}epoch-{epoch_num}/"  # mean for last epoch
            final_path = f"{base_path}final/"  # path to record last epoch
            pattern = rf"{epoch_path}(.+)"
            if regex_match := re.match(pattern, key):
                (key_name,) = regex_match.groups()
                val = self.logger.name_to_value[key]
                new_key = f"{final_path}{key_name}"
                self.logger.record(new_key, val)

    def _training_inner_loop(self, fragments, feedback):
        """Inner loop of training, for a single minibatch."""
        # The imitation implementation returns a NamedTuple where `loss` has to be unpacked. This is to pass accuracy
        # through in addition to loss for logging. I've decided to skip all that for now.
        loss = self.loss.forward(fragments, feedback, self._feedback_model)
        self.logger.record("loss", loss)
        return loss


class ScalarRewardLearner(base.BaseImitationAlgorithm):
    """Main interface for reward learning using scalar reward feedback.

    Largely mimicking PreferenceComparisons class from imitation.algorithms.preference_comparisons. If this code ever
    sees the light of day, this will first have been refactored to avoid code duplication.
    """

    def __init__(
        self,
        trajectory_generator,
        reward_model,
        num_iterations,
        fragmenter,
        feedback_gatherer,
        reward_trainer,
        feedback_queue_size=None,
        fragment_length=100,
        transition_oversampling=1,
        initial_feedback_frac=0.1,
        initial_epoch_multiplier=200.0,
        custom_logger=None,
        query_schedule="hyperbolic",
        policy_evaluator=None,
    ):
        super().__init__(custom_logger=custom_logger, allow_variable_horizon=False)

        # For keeping track of the global iteration, in case train() is called multiple times
        self._iteration = 0

        self.num_iterations = num_iterations

        self.model = reward_model

        self.trajectory_generator = trajectory_generator
        self.trajectory_generator.logger = self.logger

        self.fragmenter = fragmenter
        self.fragmenter.logger = self.logger

        self.feedback_gatherer = feedback_gatherer
        self.feedback_gatherer.logger = self.logger

        self.reward_trainer = reward_trainer
        self.reward_trainer.logger = self.logger

        self.feedback_queue_size = feedback_queue_size
        self.fragment_length = fragment_length
        self.transition_oversampling = transition_oversampling
        self.initial_feedback_frac = initial_feedback_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier

        if query_schedule not in preference_comparisons.QUERY_SCHEDULES:
            raise NotImplementedError(f"Callable query schedules not implemented.")
        self.query_schedule = preference_comparisons.QUERY_SCHEDULES[query_schedule]

        self.dataset = ScalarFeedbackDataset(max_size=feedback_queue_size)

        self.policy_evaluator = policy_evaluator

    def train(self, total_timesteps, total_queries):
        initial_queries = int(self.initial_feedback_frac * total_queries)
        total_queries -= initial_queries

        # Compute the number of feedback queries to request at each iteration in advance.
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(probs * total_queries)
        schedule = [initial_queries] + shares.tolist()
        print(f"Query schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(total_timesteps, self.num_iterations)
        reward_loss = None

        for i, num_queries in enumerate(schedule):
            iter_log_str = f"Beginning iteration {i} of {self.num_iterations}"
            if self._iteration != i:
                iter_log_str += f" (global iteration {self._iteration})"
            self.logger.log(iter_log_str)

            #######################
            # Gather new feedback #
            #######################
            num_steps = np.ceil(self.transition_oversampling * num_queries * self.fragment_length).astype(int)
            self.logger.log(f"Collecting {num_queries} feedback queries ({num_steps} transitions)")
            trajectories = self.trajectory_generator.sample(num_steps)
            #  This assumes there are no fragments missing initial timesteps
            # (but allows for fragments missing terminal timesteps).
            horizons = (len(traj) for traj in trajectories if traj.terminal)
            self._check_fixed_horizon(horizons)

            self.logger.log("Fragmenting trajectories")
            fragments = self.fragmenter(trajectories, self.fragment_length, num_queries)
            self.logger.log("Gathering feedback")
            feedback = self.feedback_gatherer(fragments)
            self.dataset.push(fragments, feedback)
            self.logger.log(f"Dataset now contains {len(self.dataset.reward_labels)} feedback queries")
            self.logger.record(f"dataset_size", len(self.dataset.reward_labels))

            ######################
            # Train reward model #
            ######################

            # On the first iteration, we train the reward model for longer, as specified by initial_epoch_multiplier.
            epoch_multiplier = self.initial_epoch_multiplier if i == 0 else 1.0

            start_time = time.time()
            self.reward_trainer.train(self.dataset, epoch_multiplier=epoch_multiplier)
            self.logger.record("reward_train_time", time.time() - start_time)

            base_key = self.logger.get_accumulate_prefixes() + "reward/final/train"
            assert f"{base_key}/loss" in self.logger.name_to_value
            reward_loss = self.logger.name_to_value[f"{base_key}/loss"]
            self.logger.record("reward_loss", reward_loss)

            ###################
            # Train the agent #
            ###################

            num_steps = timesteps_per_iteration
            # If the number of timesteps per iteration doesn't exactly divide the desired total number of timesteps,
            # we train the agent a bit longer at the end of training (where the reward model is presumably best).
            if i == self.num_iterations - 1:
                num_steps += extra_timesteps

            self.logger.log(f"Training agent for {num_steps} timesteps")
            self.trajectory_generator.train(steps=num_steps)

            ###################
            # Log information #
            ###################

            if self.policy_evaluator is not None:
                with networks.evaluating(self.model):
                    prop_bad, prop_bad_per_condition = self.policy_evaluator.evaluate(
                        policy=self.trajectory_generator.policy,
                        env=self.trajectory_generator.env,
                        num_trajs=1000,
                    )
                    self.logger.record("policy_behavior/prop_bad_rollouts", prop_bad)
                    for condition, prop in prop_bad_per_condition.items():
                        self.logger.record(f"policy_behavior/prop_bad_rollouts_{condition}", prop)

            self.logger.dump(self._iteration)

            if self.callback is not None:
                self.callback(self)

            self._iteration += 1

        return {"reward_loss": reward_loss}


#######################################################################################################################
#                           Preference comparisons modules                                                            #                                                                                                                     
#######################################################################################################################

"""Learning reward models using preference comparisons.

Trains a reward model and optionally a policy based on preferences
between trajectory fragments.
"""

QUERY_SCHEDULES: Dict[str, type_aliases.Schedule] = {
    "constant": lambda t: 1.0,
    "hyperbolic": lambda t: 1.0 / (1.0 + t),
    "inverse_quadratic": lambda t: 1.0 / (1.0 + t**2),
}


class RandomFragmenter(Fragmenter):
    """Sample fragments of trajectories uniformly at random with replacement.

    Note that each fragment is part of a single episode and has a fixed
    length. This leads to a bias: transitions at the beginning and at the
    end of episodes are less likely to occur as part of fragments (this affects
    the first and last fragment_length transitions).

    An additional bias is that trajectories shorter than the desired fragment
    length are never used.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        warning_threshold: int = 10,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        get_limits: bool = False,
    ) -> None:
        """Initialize the fragmenter.

        Args:
            rng: the random number generator
            warning_threshold: give a warning if the number of available
                transitions is less than this many times the number of
                required samples. Set to 0 to disable this warning.
            custom_logger: Where to log to; if None (default), creates a new logger.
            get_limits: if True, the fragmenter will return the start and end indices
                of the fragments in the original trajectories.
        """
        super().__init__(custom_logger)
        self.rng = rng
        self.warning_threshold = warning_threshold
        self.get_limits = get_limits

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        fragments: List[TrajectoryWithRew] = []
        limits: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

        prev_num_trajectories = len(trajectories)
        # filter out all trajectories that are too short
        trajectories = [traj for traj in trajectories if len(traj) >= fragment_length]
        if len(trajectories) == 0:
            raise ValueError(
                "No trajectories are long enough for the desired fragment length "
                f"of {fragment_length}.",
            )
        num_discarded = prev_num_trajectories - len(trajectories)
        if num_discarded:
            self.logger.log(
                f"Discarded {num_discarded} out of {prev_num_trajectories} "
                "trajectories because they are shorter than the desired length "
                f"of {fragment_length}.",
            )

        weights = [len(traj) for traj in trajectories]

        # number of transitions that will be contained in the fragments
        num_transitions = 2 * num_pairs * fragment_length
        if sum(weights) < num_transitions:
            self.logger.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times.",
            )
        elif (
            self.warning_threshold
            and sum(weights) < self.warning_threshold * num_transitions
        ):
            # If the number of available transitions is not much larger
            # than the number of requires ones, we already give a warning.
            # But only if self.warning_threshold is non-zero.
            self.logger.warn(
                f"Samples will contain {num_transitions} transitions in total "
                f"and only {sum(weights)} are available. "
                f"Because we sample with replacement, a significant number "
                "of transitions are likely to appear multiple times.",
            )

        # we need two fragments for each comparison
        for _ in range(2 * num_pairs):
            # NumPy's annotation here is overly-conservative, but this works at runtime
            traj = self.rng.choice(
                trajectories,  # type: ignore[arg-type]
                p=np.array(weights) / sum(weights),
            )
            n = len(traj)
            start = self.rng.integers(0, n - fragment_length, endpoint=True)
            end = start + fragment_length
            terminal = (end == n) and traj.terminal
            fragment = TrajectoryWithRew(
                obs=traj.obs[start : end + 1],
                acts=traj.acts[start:end],
                infos=traj.infos[start:end] if traj.infos is not None else None,
                rews=traj.rews[start:end],
                terminal=terminal,
            )
            fragments.append(fragment)
            limits.append((start, end))
            # Here start and end are indices into the original trajectory.
            # TODO (joan): they can be useful in the camera observation case
            # to know where the fragment was taken from + the camera position.
            # fragments.append((fragment, start, end)) 

        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        iterator = iter(fragments)
        limit_pairs = list(zip(limits[0::2], limits[1::2]))  # Pairing start/end indices

        if self.get_limits:
            return list(zip(iterator, iterator)), limit_pairs       
        
        return list(zip(iterator, iterator))

    
class SyntheticGatherer(PreferenceGatherer):
    """Computes synthetic preferences using ground-truth environment rewards."""

    def __init__(
        self,
        temperature: float = 1,
        discount_factor: float = 0.9,
        sample: bool = True,
        rng: Optional[np.random.Generator] = None,
        threshold: float = 50,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        get_limits: bool = False,
    ) -> None:
        """Initialize the synthetic preference gatherer.

        Args:
            temperature: the preferences are sampled from a softmax, this is
                the temperature used for sampling. temperature=0 leads to deterministic
                results (for equal rewards, 0.5 will be returned).
            discount_factor: discount factor that is used to compute
                how good a fragment is. Default is to use undiscounted
                sums of rewards (as in the DRLHP paper).
            sample: if True (default), the preferences are 0 or 1, sampled from
                a Bernoulli distribution (or 0.5 in the case of ties with zero
                temperature). If False, then the underlying Bernoulli probabilities
                are returned instead.
            rng: random number generator, only used if
                ``temperature > 0`` and ``sample=True``
            threshold: preferences are sampled from a softmax of returns.
                To avoid overflows, we clip differences in returns that are
                above this threshold (after multiplying with temperature).
                This threshold is therefore in logspace. The default value
                of 50 means that probabilities below 2e-22 are rounded up to 2e-22.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: if `sample` is true and no random state is provided.
        """
        super().__init__(custom_logger=custom_logger)
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.sample = sample
        self.rng = rng
        self.threshold = threshold

        if self.sample and self.rng is None:
            raise ValueError("If `sample` is True, then `rng` must be provided.")

    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Computes probability fragment 1 is preferred over fragment 2."""
        returns1, returns2 = self._reward_sums(fragment_pairs)
        if self.temperature == 0:
            return (np.sign(returns1 - returns2) + 1) / 2

        returns1 /= self.temperature
        returns2 /= self.temperature

        # clip the returns to avoid overflows in the softmax below
        returns_diff = np.clip(returns2 - returns1, -self.threshold, self.threshold)
        # Instead of computing exp(rews1) / (exp(rews1) + exp(rews2)) directly,
        # we divide enumerator and denominator by exp(rews1) to prevent overflows:
        model_probs = 1 / (1 + np.exp(returns_diff))
        # Compute the mean binary entropy. This metric helps estimate
        # how good we can expect the performance of the learned reward
        # model to be at predicting preferences.
        entropy = -(
            special.xlogy(model_probs, model_probs)
            + special.xlogy(1 - model_probs, 1 - model_probs)
        ).mean()
        self.logger.record("entropy", entropy)

        if self.sample:
            assert self.rng is not None
            return self.rng.binomial(n=1, p=model_probs).astype(np.float32)
        return model_probs

    def _reward_sums(self, fragment_pairs) -> Tuple[np.ndarray, np.ndarray]:
        rews1, rews2 = zip(
            *[
                (
                    rollout.discounted_sum(f1.rews, self.discount_factor),
                    rollout.discounted_sum(f2.rews, self.discount_factor),
                )
                for f1, f2 in fragment_pairs
            ],
        )
        return np.array(rews1, dtype=np.float32), np.array(rews2, dtype=np.float32)


def _trajectory_pair_includes_reward(fragment_pair: TrajectoryPair) -> bool:
    """Return true if and only if both fragments in the pair include rewards."""
    frag1, frag2 = fragment_pair
    return isinstance(frag1, TrajectoryWithRew) and isinstance(frag2, TrajectoryWithRew)


def get_base_model(reward_model: RewardNet) -> RewardNet:
    base_model = reward_model
    while hasattr(base_model, "base"):
        base_model = cast(RewardNet, base_model.base)

    return base_model

class PreferenceModel(nn.Module):
    """Class to convert two fragments' rewards into preference probability."""

    def __init__(
        self,
        model: RewardNet,
        noise_prob: float = 0.0,
        discount_factor: float = 0.9,
        threshold: float = 50,
    ) -> None:
        """Create Preference Prediction Model.

        Args:
            model: base model to compute reward.
            noise_prob: assumed probability with which the preference
                is uniformly random (used for the model of preference generation
                that is used for the loss).
            discount_factor: the model of preference generation uses a softmax
                of returns as the probability that a fragment is preferred.
                This is the discount factor used to calculate those returns.
                Default is 1, i.e. undiscounted sums of rewards (which is what
                the DRLHP paper uses).
            threshold: the preference model used to compute the loss contains
                a softmax of returns. To avoid overflows, we clip differences
                in returns that are above this threshold. This threshold
                is therefore in logspace. The default value of 50 means
                that probabilities below 2e-22 are rounded up to 2e-22.

        Raises:
            ValueError: if `RewardEnsemble` is wrapped around a class
                other than `AddSTDRewardWrapper`.
        """
        super().__init__()
        self.model = model
        self.noise_prob = noise_prob
        self.discount_factor = discount_factor
        self.threshold = threshold
        base_model = get_base_model(model)
        self.ensemble_model = None
        # if the base model is an ensemble model, then keep the base model as
        # model to get rewards from all networks
        if isinstance(base_model, RewardEnsemble):
            # reward_model may include an AddSTDRewardWrapper for RL training; but we
            # must train directly on the base model for reward model training.
            is_base = model is base_model
            is_std_wrapper = (
                isinstance(model, AddSTDRewardWrapper)
                and model.base is base_model
            )

            if not (is_base or is_std_wrapper):
                raise ValueError(
                    "RewardEnsemble can only be wrapped"
                    f" by AddSTDRewardWrapper but found {type(model).__name__}.",
                )
            self.ensemble_model = base_model
            self.member_pref_models = []
            for member in self.ensemble_model.members:
                member_pref_model = PreferenceModel(
                    cast(RewardNet, member),  # nn.ModuleList is not generic
                    self.noise_prob,
                    self.discount_factor,
                    self.threshold,
                )
                self.member_pref_models.append(member_pref_model)

    def forward(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """Computes the preference probability of the first fragment for all pairs.

        Note: This function passes the gradient through for non-ensemble models.
              For an ensemble model, this function should not be used for loss
              calculation. It can be used in case where passing the gradient is not
              required such as during active selection or inference time.
              Therefore, the EnsembleTrainer passes each member network through this
              function instead of passing the EnsembleNetwork object with the use of
              `ensemble_member_index`.

        Args:
            fragment_pairs: batch of pair of fragments.

        Returns:
            A tuple with the first element as the preference probabilities for the
            first fragment for all fragment pairs given by the network(s).
            If the ground truth rewards are available, it also returns gt preference
            probabilities in the second element of the tuple (else None).
            Reward probability shape - (num_fragment_pairs, ) for non-ensemble reward
            network and (num_fragment_pairs, num_networks) for an ensemble of networks.

        """
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        gt_reward_available = _trajectory_pair_includes_reward(fragment_pairs[0])
        if gt_reward_available:
            gt_probs = th.empty(len(fragment_pairs), dtype=th.float32)
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            rews1 = self.rewards(trans1)
            rews2 = self.rewards(trans2)
            probs[i] = self.probability(rews1, rews2)
            if gt_reward_available:
                frag1 = cast(TrajectoryWithRew, frag1)
                frag2 = cast(TrajectoryWithRew, frag2)
                gt_rews_1 = th.from_numpy(frag1.rews)
                gt_rews_2 = th.from_numpy(frag2.rews)
                gt_probs[i] = self.probability(gt_rews_1, gt_rews_2)

        return probs, (gt_probs if gt_reward_available else None)

    def rewards(self, transitions: Transitions) -> th.Tensor:
        """Computes the reward for all transitions.

        Args:
            transitions: batch of obs-act-obs-done for a fragment of a trajectory.

        Returns:
            The reward given by the network(s) for all the transitions.
            Shape - (num_transitions, ) for Single reward network and
            (num_transitions, num_networks) for ensemble of networks.
        """
        state = types.assert_not_dictobs(transitions.obs)
        action = transitions.acts
        next_state = types.assert_not_dictobs(transitions.next_obs)
        done = transitions.dones
        if self.ensemble_model is not None:
            rews_np = self.ensemble_model.predict_processed_all(
                state,
                action,
                next_state,
                done,
            )
            assert rews_np.shape == (len(state), self.ensemble_model.num_members)
            rews = util.safe_to_tensor(rews_np).to(self.ensemble_model.device)
        else:
            preprocessed = self.model.preprocess(state, action, next_state, done)
            rews = self.model(*preprocessed)
            assert rews.shape == (len(state),)
        return rews

    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        """Computes the Boltzmann rational probability the first trajectory is best.

        Args:
            rews1: array/matrix of rewards for the first trajectory fragment.
                matrix for ensemble models and array for non-ensemble models.
            rews2: array/matrix of rewards for the second trajectory fragment.
                matrix for ensemble models and array for non-ensemble models.

        Returns:
            The softmax of the difference between the (discounted) return of the
            first and second trajectory.
            Shape - (num_ensemble_members, ) for ensemble model and
            () for non-ensemble model which is a torch scalar.
        """
        # check rews has correct shape based on the model
        expected_dims = 2 if self.ensemble_model is not None else 1
        assert rews1.ndim == rews2.ndim == expected_dims
        # First, we compute the difference of the returns of
        # the two fragments. We have a special case for a discount
        # factor of 1 to avoid unnecessary computation (especially
        # since this is the default setting).
        if self.discount_factor == 1:
            returns_diff = (rews2 - rews1).sum(axis=0)  # type: ignore[call-overload]
        else:
            device = rews1.device
            assert device == rews2.device
            discounts = self.discount_factor ** th.arange(len(rews1), device=device)
            if self.ensemble_model is not None:
                discounts = discounts.reshape(-1, 1)
            returns_diff = (discounts * (rews2 - rews1)).sum(axis=0)
        # Clip to avoid overflows (which in particular may occur
        # in the backwards pass even if they do not in the forward pass).
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        # We take the softmax of the returns. model_probability
        # is the first dimension of that softmax, representing the
        # probability that fragment 1 is preferred.
        model_probability = 1 / (1 + returns_diff.exp())
        probability = self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability
        if self.ensemble_model is not None:
            assert probability.shape == (self.model.num_members,)
        else:
            assert probability.shape == ()
        return probability
    

class LossAndMetrics(NamedTuple):
    """Loss and auxiliary metrics for reward network training."""

    loss: th.Tensor
    metrics: Mapping[str, th.Tensor]
    
class RewardLoss(nn.Module, abc.ABC):
    """A loss function over preferences."""

    @abc.abstractmethod
    def forward(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
        preference_model: PreferenceModel,
    ) -> LossAndMetrics:
        """Computes the loss.

        Args:
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).
            preference_model: model to predict the preferred fragment from a pair.

        Returns: # noqa: DAR202
            loss: the loss
            metrics: a dictionary of metrics that can be logged
        """

# class CrossEntropyRewardLoss(RewardLoss):
#     """Compute the cross entropy reward loss."""

#     def __init__(self) -> None:
#         """Create cross entropy reward loss."""
#         super().__init__()

#     def forward(
#         self,
#         fragment_pairs: Sequence[TrajectoryPair],
#         preferences: np.ndarray,
#         preference_model: PreferenceModel,
#     ) -> LossAndMetrics:
#         """Computes the loss.

#         Args:
#             fragment_pairs: Batch consisting of pairs of trajectory fragments.
#             preferences: The probability that the first fragment is preferred
#                 over the second. Typically 0, 1 or 0.5 (tie).
#             preference_model: model to predict the preferred fragment from a pair.

#         Returns:
#             The cross-entropy loss between the probability predicted by the
#                 reward model and the target probabilities in `preferences`. Metrics
#                 are accuracy, and gt_reward_loss, if the ground truth reward is
#                 available.
#         """
#         probs, gt_probs = preference_model(fragment_pairs)
#         # TODO(ejnnr): Here and below, > 0.5 is problematic
#         #  because getting exactly 0.5 is actually somewhat
#         #  common in some environments (as long as sample=False or temperature=0).
#         #  In a sense that "only" creates class imbalance
#         #  but it's still misleading.
#         predictions = probs > 0.5
#         preferences_th = th.as_tensor(preferences, dtype=th.float32)
#         ground_truth = preferences_th > 0.5
#         metrics = {}
#         metrics["accuracy"] = (predictions == ground_truth).float().mean()
#         if gt_probs is not None:
#             metrics["gt_reward_loss"] = th.nn.functional.binary_cross_entropy(
#                 gt_probs,
#                 preferences_th,
#             )
#         metrics = {key: value.detach().cpu() for key, value in metrics.items()}
#         return LossAndMetrics(
#             loss=th.nn.functional.binary_cross_entropy(probs, preferences_th),
#             metrics=metrics,
#         )

class CrossEntropyRewardLoss(RewardLoss):
    """Compute the cross entropy reward loss."""

    def __init__(self) -> None:
        """Create cross entropy reward loss."""
        super().__init__()

    def forward(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
        preference_model: PreferenceModel,
    ) -> LossAndMetrics:
        """Computes the loss.

        Args:
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).
            preference_model: model to predict the preferred fragment from a pair.

        Returns:
            The cross-entropy loss between the probability predicted by the
                reward model and the target probabilities in `preferences`. Metrics
                are accuracy, and gt_reward_loss, if the ground truth reward is
                available.
        """
        probs, gt_probs = preference_model(fragment_pairs)
        
        # Handle the case when the preference is exactly 0.5
        preferences_th = th.as_tensor(preferences, dtype=th.float32)
        is_tie = (preferences_th == 0.5)
        
        # Compute predictions and ground truth
        predictions = probs > 0.5
        ground_truth = preferences_th > 0.5
        
        # Mask out tie cases from predictions and ground truth
        predictions = predictions[~is_tie]
        ground_truth = ground_truth[~is_tie]
        
        metrics = {}
        if len(predictions) > 0:
            metrics["accuracy"] = (predictions == ground_truth).float().mean()
        else:
            metrics["accuracy"] = th.tensor(0.0)
        
        if gt_probs is not None:
            # Mask out tie cases from gt_probs and preferences_th
            gt_probs = gt_probs[~is_tie]
            preferences_th = preferences_th[~is_tie]
            
            if len(gt_probs) > 0:
                metrics["gt_reward_loss"] = th.nn.functional.binary_cross_entropy(
                    gt_probs,
                    preferences_th,
                )
            else:
                metrics["gt_reward_loss"] = th.tensor(0.0)
        
        metrics = {key: value.detach().cpu() for key, value in metrics.items()}
        
        # Compute the cross-entropy loss
        if len(probs[~is_tie]) > 0:
            loss = th.nn.functional.binary_cross_entropy(
                probs[~is_tie],
                preferences_th[~is_tie],
            )
        else:
            loss = th.tensor(0.0)
        
        return LossAndMetrics(
            loss=loss,
            metrics=metrics,
        )

class RewardTrainer(abc.ABC):
    """Abstract base class for training reward models using preference comparisons.

    This class contains only the actual reward model training code,
    it is not responsible for gathering trajectories and preferences
    or for agent training (see :class: `PreferenceComparisons` for that).
    """

    def __init__(
        self,
        preference_model: PreferenceModel,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        """Initialize the reward trainer.

        Args:
            preference_model: the preference model to train the reward network.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self._preference_model = preference_model
        self._logger = custom_logger or imit_logger.configure()

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        return self._logger

    @logger.setter
    def logger(self, custom_logger: imit_logger.HierarchicalLogger) -> None:
        self._logger = custom_logger

    def train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0) -> None:
        """Train the reward model on a batch of fragment pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
            epoch_multiplier: how much longer to train for than usual
                (measured relatively).
        """
        with networks.training(self._preference_model.model):
            self._train(dataset, epoch_multiplier)

    @abc.abstractmethod
    def _train(self, dataset: PreferenceDataset, epoch_multiplier: float) -> None:
        """Train the reward model; see ``train`` for details."""


class BasicRewardTrainer(RewardTrainer):
    """Train a basic reward model."""

    regularizer: Optional[regularizers.Regularizer]

    def __init__(
        self,
        preference_model: PreferenceModel,
        loss: RewardLoss,
        rng: np.random.Generator,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        epochs: int = 1,
        lr: float = 1e-3,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        regularizer_factory: Optional[regularizers.RegularizerFactory] = None,
    ) -> None:
        """Initialize the reward model trainer.

        Args:
            preference_model: the preference model to train the reward network.
            loss: the loss to use
            rng: the random number generator to use for splitting the dataset into
                training and validation.
            batch_size: number of fragment pairs per batch
            minibatch_size: size of minibatch to calculate gradients over.
                The gradients are accumulated until `batch_size` examples
                are processed before making an optimization step. This
                is useful in GPU training to reduce memory usage, since
                fewer examples are loaded into memory at once,
                facilitating training with larger batch sizes, but is
                generally slower. Must be a factor of `batch_size`.
                Optional, defaults to `batch_size`.
            epochs: number of epochs in each training iteration (can be adjusted
                on the fly by specifying an `epoch_multiplier` in `self.train()`
                if longer training is desired in specific cases).
            lr: the learning rate
            custom_logger: Where to log to; if None (default), creates a new logger.
            regularizer_factory: if you would like to apply regularization during
                training, specify a regularizer factory here. The factory will be
                used to construct a regularizer. See
                ``imitation.regularization.RegularizerFactory`` for more details.

        Raises:
            ValueError: if the batch size is not a multiple of the minibatch size.
        """
        super().__init__(preference_model, custom_logger)
        self.loss = loss
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        self.epochs = epochs
        self.optim = th.optim.AdamW(self._preference_model.parameters(), lr=lr)
        self.rng = rng
        self.regularizer = (
            regularizer_factory(optimizer=self.optim, logger=self.logger)
            if regularizer_factory is not None
            else None
        )

    def _make_data_loader(self, dataset: data_th.Dataset) -> data_th.DataLoader:
        """Make a dataloader."""
        return data_th.DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            collate_fn=preference_collate_fn,
        )

    @property
    def requires_regularizer_update(self) -> bool:
        """Whether the regularizer requires updating.

        Returns:
            If true, this means that a validation dataset will be used.
        """
        return self.regularizer is not None and self.regularizer.val_split is not None

    def _train(
        self,
        dataset: PreferenceDataset,
        epoch_multiplier: float = 1.0,
    ) -> None:
        """Trains for `epoch_multiplier * self.epochs` epochs over `dataset`."""
        if self.regularizer is not None and self.regularizer.val_split is not None:
            val_length = int(len(dataset) * self.regularizer.val_split)
            train_length = len(dataset) - val_length
            if val_length < 1 or train_length < 1:
                raise ValueError(
                    "Not enough data samples to split into training and validation, "
                    "or the validation split is too large/small. "
                    "Make sure you've generated enough initial preference data. "
                    "You can adjust this through initial_comparison_frac in "
                    "PreferenceComparisons.",
                )
            train_dataset, val_dataset = data_th.random_split(
                dataset,
                lengths=[train_length, val_length],
                # we convert the numpy generator to the pytorch generator.
                generator=th.Generator().manual_seed(util.make_seeds(self.rng)),
            )
            dataloader = self._make_data_loader(train_dataset)
            val_dataloader = self._make_data_loader(val_dataset)
        else:
            dataloader = self._make_data_loader(dataset)
            val_dataloader = None

        epochs = round(self.epochs * epoch_multiplier)

        assert epochs > 0, "Must train for at least one epoch."
        with self.logger.accumulate_means("reward"):
            for epoch_num in tqdm(range(epochs), desc="Training reward model"):
                with self.logger.add_key_prefix(f"epoch-{epoch_num}"):
                    train_loss = 0.0
                    accumulated_size = 0
                    self.optim.zero_grad()
                    for fragment_pairs, preferences in dataloader:
                        with self.logger.add_key_prefix("train"):
                            loss = self._training_inner_loop(
                                fragment_pairs,
                                preferences,
                            )

                            # Renormalise the loss to be averaged over
                            # the whole batch size instead of the
                            # minibatch size. If there is an incomplete
                            # batch, its gradients will be smaller,
                            # which may be helpful for stability.
                            loss *= len(fragment_pairs) / self.batch_size

                        train_loss += loss.item()
                        if self.regularizer:
                            self.regularizer.regularize_and_backward(loss)
                        else:
                            loss.backward()

                        accumulated_size += len(fragment_pairs)
                        if accumulated_size >= self.batch_size:
                            self.optim.step()
                            self.optim.zero_grad()
                            accumulated_size = 0
                    if accumulated_size != 0:
                        self.optim.step()  # if there remains an incomplete batch

                    if not self.requires_regularizer_update:
                        continue
                    assert val_dataloader is not None
                    assert self.regularizer is not None

                    val_loss = 0.0
                    for fragment_pairs, preferences in val_dataloader:
                        with self.logger.add_key_prefix("val"):
                            val_loss += self._training_inner_loop(
                                fragment_pairs,
                                preferences,
                            ).item()
                    self.regularizer.update_params(train_loss, val_loss)

        # after training all the epochs,
        # record also the final value in a separate key for easy access.
        keys = list(self.logger.name_to_value.keys())
        outer_prefix = self.logger.get_accumulate_prefixes()
        for key in keys:
            base_path = f"{outer_prefix}reward/"  # existing prefix + accum_means ctx
            epoch_path = f"mean/{base_path}epoch-{epoch_num}/"  # mean for last epoch
            final_path = f"{base_path}final/"  # path to record last epoch
            pattern = rf"{epoch_path}(.+)"
            if regex_match := re.match(pattern, key):
                (key_name,) = regex_match.groups()
                val = self.logger.name_to_value[key]
                new_key = f"{final_path}{key_name}"
                self.logger.record(new_key, val)

    def _training_inner_loop(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
    ) -> th.Tensor:
        output = self.loss.forward(fragment_pairs, preferences, self._preference_model)
        loss = output.loss
        self.logger.record("loss", loss.item())
        for name, value in output.metrics.items():
            self.logger.record(name, value.item())
        return loss
    

class PreferenceComparisons(base.BaseImitationAlgorithm):
    """Main interface for reward learning using preference comparisons."""

    def __init__(
        self,
        trajectory_generator: TrajectoryGenerator,
        reward_model: RewardNet,
        num_iterations: int,
        fragmenter: Optional[Fragmenter] = None,
        preference_gatherer: Optional[PreferenceGatherer] = None,
        reward_trainer: Optional[RewardTrainer] = None,
        comparison_queue_size: Optional[int] = None,
        fragment_length: int = 100,
        transition_oversampling: float = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
        rng: Optional[np.random.Generator] = None,
        query_schedule: Union[str, type_aliases.Schedule] = "hyperbolic",
        policy_evaluator = None,
    ) -> None:
        """Initialize the preference comparison trainer.

        The loggers of all subcomponents are overridden with the logger used
        by this class.

        Args:
            trajectory_generator: generates trajectories while optionally training
                an RL agent on the learned reward function (can also be a sampler
                from a static dataset of trajectories though).
            reward_model: a RewardNet instance to be used for learning the reward
            num_iterations: number of times to train the agent against the reward model
                and then train the reward model against newly gathered preferences.
            fragmenter: takes in a set of trajectories and returns pairs of fragments
                for which preferences will be gathered. These fragments could be random,
                or they could be selected more deliberately (active learning).
                Default is a random fragmenter.
            preference_gatherer: how to get preferences between trajectory fragments.
                Default (and currently the only option) is to use synthetic preferences
                based on ground-truth rewards. Human preferences could be implemented
                here in the future.
            reward_trainer: trains the reward model based on pairs of fragments and
                associated preferences. Default is to use the preference model
                and loss function from DRLHP.
            comparison_queue_size: the maximum number of comparisons to keep in the
                queue for training the reward model. If None, the queue will grow
                without bound as new comparisons are added.
            fragment_length: number of timesteps per fragment that is used to elicit
                preferences
            transition_oversampling: factor by which to oversample transitions before
                creating fragments. Since fragments are sampled with replacement,
                this is usually chosen > 1 to avoid having the same transition
                in too many fragments.
            initial_comparison_frac: fraction of the total_comparisons argument
                to train() that will be sampled before the rest of training begins
                (using a randomly initialized agent). This can be used to pretrain the
                reward model before the agent is trained on the learned reward, to
                help avoid irreversibly learning a bad policy from an untrained reward.
                Note that there will often be some additional pretraining comparisons
                since `comparisons_per_iteration` won't exactly divide the total number
                of comparisons. How many such comparisons there are depends
                discontinuously on `total_comparisons` and `comparisons_per_iteration`.
            initial_epoch_multiplier: before agent training begins, train the reward
                model for this many more epochs than usual (on fragments sampled from a
                random agent).
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
            rng: random number generator to use for initializing subcomponents such as
                fragmenter.
                Only used when default components are used; if you instantiate your
                own fragmenter, preference gatherer, etc., you are responsible for
                seeding them!
            query_schedule: one of ("constant", "hyperbolic", "inverse_quadratic"), or
                a function that takes in a float between 0 and 1 inclusive,
                representing a fraction of the total number of timesteps elapsed up to
                some time T, and returns a potentially unnormalized probability
                indicating the fraction of `total_comparisons` that should be queried
                at that iteration. This function will be called `num_iterations` times
                in `__init__()` with values from `np.linspace(0, 1, num_iterations)`
                as input. The outputs will be normalized to sum to 1 and then used to
                apportion the comparisons among the `num_iterations` iterations.

        Raises:
            ValueError: if `query_schedule` is not a valid string or callable.
        """
        super().__init__(
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        # for keeping track of the global iteration, in case train() is called
        # multiple times
        self._iteration = 0

        self.model = reward_model
        self.rng = rng

        # are any of the optional args that require a rng None?
        has_any_rng_args_none = None in (
            preference_gatherer,
            fragmenter,
            reward_trainer,
        )

        if self.rng is None and has_any_rng_args_none:
            raise ValueError(
                "If you don't provide a random state, you must provide your own "
                "seeded fragmenter, preference gatherer, and reward_trainer. "
                "You can initialize a random state with `np.random.default_rng(seed)`.",
            )
        elif self.rng is not None and not has_any_rng_args_none:
            raise ValueError(
                "If you provide your own fragmenter, preference gatherer, "
                "and reward trainer, you don't need to provide a random state.",
            )

        if reward_trainer is None:
            assert self.rng is not None
            preference_model = PreferenceModel(reward_model)
            loss = CrossEntropyRewardLoss()
            self.reward_trainer = _make_reward_trainer(
                preference_model,
                loss,
                rng=self.rng,
            )
        else:
            self.reward_trainer = reward_trainer

        # If the reward trainer was created in the previous line, we've already passed
        # the correct logger. But if the user created a RewardTrainer themselves and
        # didn't manually set a logger, it would be annoying if a separate one was used.
        self.reward_trainer.logger = self.logger
        self.trajectory_generator = trajectory_generator
        self.trajectory_generator.logger = self.logger
        if fragmenter:
            self.fragmenter = fragmenter
        else:
            assert self.rng is not None
            self.fragmenter = RandomFragmenter(
                custom_logger=self.logger,
                rng=self.rng,
            )
        self.fragmenter.logger = self.logger
        if preference_gatherer:
            self.preference_gatherer = preference_gatherer
        else:
            assert self.rng is not None
            self.preference_gatherer = SyntheticGatherer(
                custom_logger=self.logger,
                rng=self.rng,
            )

        self.preference_gatherer.logger = self.logger

        self.fragment_length = fragment_length
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.num_iterations = num_iterations
        self.transition_oversampling = transition_oversampling
        if callable(query_schedule):
            self.query_schedule = query_schedule
        elif query_schedule in QUERY_SCHEDULES:
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            raise ValueError(f"Unknown query schedule: {query_schedule}")

        self.dataset = PreferenceDataset(max_size=comparison_queue_size)
        self.policy_evaluator = policy_evaluator

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> Mapping[str, Any]:
        """Train the reward model and the policy if applicable.

        Args:
            total_timesteps: number of environment interaction steps
            total_comparisons: number of preferences to gather in total
            callback: callback functions called at the end of each iteration

        Returns:
            A dictionary with final metrics such as loss and accuracy
            of the reward model.
        """
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons -= initial_comparisons

        # Compute the number of comparisons to request at each iteration in advance.
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(probs * total_comparisons)
        schedule = [initial_comparisons] + shares.tolist()
        print(f"Query schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps,
            self.num_iterations,
        )
        reward_loss = None
        reward_accuracy = None

        for i, num_pairs in enumerate(schedule):
            ##########################
            # Gather new preferences #
            ##########################
            num_steps = math.ceil(
                self.transition_oversampling * 2 * num_pairs * self.fragment_length,
            )
            print(f"Number of steps: {num_steps}")
            
            self.logger.log(
                f"Collecting {2 * num_pairs} fragments ({num_steps} transitions)",
            )
            trajectories = self.trajectory_generator.sample(num_steps)
            # This assumes there are no fragments missing initial timesteps
            # (but allows for fragments missing terminal timesteps).
            horizons = (len(traj) for traj in trajectories if traj.terminal)
            # TODO (joan): if we want to allow for easy camera model observability,
            # We should create the visibility_mask here and pass it to the fragmenter and gatherer.
            self._check_fixed_horizon(horizons)
            self.logger.log("Creating fragment pairs")
            if self.fragmenter.get_limits:
                fragments, limits = self.fragmenter(trajectories, self.fragment_length, num_pairs)
            else:
                fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)
            # Limits are the indices of the last transition in each fragment.
            # We can use them to place the camera correctly.

            with self.logger.accumulate_means("preferences"):
                self.logger.log("Gathering preferences")
                self.logger.log(f"Gatherer type: {self.preference_gatherer.__class__.__name__}")

                # TODO (joan): if we want to allow for easy camera model observability,
                # a more elegant solution can be probably thought of.

                if self.fragmenter.get_limits:
                    preferences = self.preference_gatherer(fragments, limits)
                else:
                    preferences = self.preference_gatherer(fragments)

            self.dataset.push(fragments, preferences)
            self.logger.log(f"Dataset now contains {len(self.dataset)} comparisons")

            ##########################
            # Train the reward model #
            ##########################

            # On the first iteration, we train the reward model for longer,
            # as specified by initial_epoch_multiplier.
            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier

            self.reward_trainer.train(self.dataset, epoch_multiplier=epoch_multiplier)
            base_key = self.logger.get_accumulate_prefixes() + "reward/final/train"
            assert f"{base_key}/loss" in self.logger.name_to_value
            assert f"{base_key}/accuracy" in self.logger.name_to_value
            reward_loss = self.logger.name_to_value[f"{base_key}/loss"]
            reward_accuracy = self.logger.name_to_value[f"{base_key}/accuracy"]

            ###################
            # Train the agent #
            ###################
            num_steps = timesteps_per_iteration
            # if the number of timesteps per iterations doesn't exactly divide
            # the desired total number of timesteps, we train the agent a bit longer
            # at the end of training (where the reward model is presumably best)
            if i == self.num_iterations - 1:
                num_steps += extra_timesteps
            with self.logger.accumulate_means("agent"):
                self.logger.log(f"Training agent for {num_steps} timesteps")
                self.trajectory_generator.train(steps=num_steps)

            
            ###################
            # Log information #
            ###################

            if self.policy_evaluator is not None:
                with networks.evaluating(self.model):
                    prop_bad, prop_bad_per_condition = self.policy_evaluator.evaluate(
                        policy=self.trajectory_generator.policy,
                        env=self.trajectory_generator.env,
                        num_trajs=100,
                    )
                    self.logger.record("policy_behavior/prop_bad_rollouts", prop_bad)
                    for condition, prop in prop_bad_per_condition.items():
                        self.logger.record(f"policy_behavior/prop_bad_rollouts_{condition}", prop)

            self.logger.dump(self._iteration)

            ########################
            # Additional Callbacks #
            ########################
            if callback:
                #callback(self._iteration)
                callback(self)
            self._iteration += 1

        # save the final reward model and agent policy
        
        self.logger.log("Training complete")
        self.logger.dump(self._iteration)



        return {"reward_loss": reward_loss, "reward_accuracy": reward_accuracy}
