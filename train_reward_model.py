# Copy of code from experiment ipython notebook

import os

import numpy as np
import torch as th
from imitation.util import logger as imit_logger

import wandb
from evaluate_reward_model import full_visibility_evaluator_factory, partial_visibility_evaluator_factory, camera_visibility_evaluator_factory
from imitation_modules import (
    BasicScalarFeedbackRewardTrainer,
    DeterministicMDPTrajGenerator,
    MSERewardLoss,
    NoisyObservationGathererWrapper,
    NonImageCnnRewardNet,
    RandomSingleFragmenter,
    ScalarFeedbackModel,
    ScalarRewardLearner,
    SyntheticScalarFeedbackGatherer,
)
from imitation_modules import (
    PreferenceComparisons,
    PreferenceModel,
    BasicRewardTrainer,
    CrossEntropyRewardLoss,
    SyntheticGatherer,
    RandomFragmenter,
    PreferenceComparisonNoisyObservationGathererWrapper,
)

from stealing_gridworld import PartialGridVisibility, DynamicGridVisibility, StealingGridworld, PartialGridVisibility_HF

#######################################################################################################################
##################################################### Run params ######################################################
#######################################################################################################################


GPU_NUMBER = 0
N_ITER = 100
N_COMPARISONS = 10_000
TESTING = False


#######################################################################################################################
##################################################### Expt params #####################################################
#######################################################################################################################


config = {
    "environment": {
        "name": "StealingGridworld",
        "grid_size": 5,
        "horizon": 30,
        "reward_for_depositing": 100,
        "reward_for_picking_up": 1,
        "reward_for_stealing": -200,
    },
    "reward_model": {
        "type": "NonImageCnnRewardNet",
        "hid_channels": [32, 32],
        "kernel_size": 3,
    },
    "seed": 42,
    "dataset_max_size": 100_000,
    # If fragment_length is None, then the whole trajectory is used as a single fragment.
    "fragment_length": 10,
    "transition_oversampling": 10,
    "initial_epoch_multiplier": 4.0,
    "feedback": {
        "type": "preference",
        "belief_direction": "optimistic",
        "belief_strength": 1.0,
        "belief_early": 1.0,
    },
    "trajectory_generator": {
        "epsilon": 0.1,
    },
    "visibility": {
        "visibility": "partial",
        # Available visibility mask keys:
        # "full": All of the grid is visible. Not actually used, but should be set for easier comparison.
        # "(n-1)x(n-1)": All but the outermost ring of the grid is visible.
        #"visibility_mask_key": "(n-1)x(n-1)",
        "visibility_mask_key": "camera",
        #"visibility_mask_key": "full",
    },
    "reward_trainer": {
        "num_epochs": 5,
    },
}

# Some validation

if config["feedback"]["type"] not in ("scalar", "preference"):
    raise NotImplementedError("Only scalar and preference feedback are supported at the moment.")

if config["feedback"]["belief_strength"] < 0 or config["feedback"]["belief_strength"] > 1:
    raise ValueError(f'Belief strength must be between 0 and 1. Instead, it is {config["feedback"]["belief_strength"]}.')

if config["feedback"]["belief_direction"] not in ["optimistic", "pessimistic", "neutral"]:
    raise ValueError(
        f'Unknown belief direction {config["feedback"]["belief_direction"]}.'
        f'Belief direction must be "optimistic", "pessimistic", or "neutral".'
    )
if config["feedback"]["belief_early"] < 0 or config["feedback"]["belief_early"] > 1:
    raise ValueError(f'Belief early must be between 0 and 1. Instead, it is {config["feedback"]["belief_early"]}.')


if config["visibility"]["visibility"] == "full" and config["visibility"]["visibility_mask_key"] != "full":
    raise ValueError(
        f'If visibility is "full", then visibility mask key must be "full".'
        f'Instead, it is {config["visibility"]["visibility_mask_key"]}.'
    )

if config["visibility"]["visibility"] not in ["full", "partial"]:
    raise ValueError(
        f'Unknown visibility {config["visibility"]["visibility"]}.' f'Visibility must be "full" or "partial".'
    )

if config["reward_model"]["type"] != "NonImageCnnRewardNet":
    raise ValueError(f'Unknown reward model type {config["reward_model"]["type"]}.')

available_visibility_mask_keys = ["full", "(n-1)x(n-1)", "camera"]
if config["visibility"]["visibility_mask_key"] not in available_visibility_mask_keys:
    raise ValueError(
        f'Unknown visibility mask key {config["visibility"]["visibility_mask_key"]}.'
        f"Available visibility mask keys are {available_visibility_mask_keys}."
    )

if config["fragment_length"] == None:
    config["fragment_length"] = config["environment"]["horizon"]
    print("Fragment length unspecified... setting it to ", config["environment"]["horizon"])

wandb.login()
run = wandb.init(
    project="assisting-bounded-humans",
    notes="Partial Observability - 5x5 Frame - Optimistic",
    name="PO_5x5_Camera_NewEVAL",
    tags=[
        "Partial Observability" if config["visibility"]["visibility"] == "partial" else "Full Observability",
        "Mask : Frame" if config["visibility"]["visibility_mask_key"] == "(n-1)x(n-1)" else "Mask : Camera" if config["visibility"]["visibility_mask_key"] == "camera" else "Mask : Full",
        f"{config['environment']['grid_size']}x{config['environment']['grid_size']}",
        f"{config['environment']['horizon']} horizon",
        f"Epochs: {config['reward_trainer']['num_epochs']}",
        f"Fragment length: {config['fragment_length']}",
        f"Max dataset size: {config['dataset_max_size']}",
        "Halt = 5"
    ],
    config=config,
    mode="disabled" if TESTING else "online",
)

#######################################################################################################################
################################################## Create everything ##################################################
#######################################################################################################################


env = StealingGridworld(
    grid_size=wandb.config["environment"]["grid_size"],
    horizon=wandb.config["environment"]["horizon"],
    reward_for_depositing=wandb.config["environment"]["reward_for_depositing"],
    reward_for_picking_up=wandb.config["environment"]["reward_for_picking_up"],
    reward_for_stealing=wandb.config["environment"]["reward_for_stealing"],
    seed=wandb.config["seed"],
)


reward_net = NonImageCnnRewardNet(
    env.observation_space,
    env.action_space,
    hid_channels=wandb.config["reward_model"]["hid_channels"],
    kernel_size=wandb.config["reward_model"]["kernel_size"],
)

rng = np.random.default_rng(wandb.config["seed"])

if GPU_NUMBER is not None:
    device = th.device(f"cuda:{GPU_NUMBER}" if th.cuda.is_available() else 'cpu')
    reward_net.to(device)
    print(f"Reward net on {device}.")

if config["feedback"]["type"] == 'scalar':
    fragmenter = RandomSingleFragmenter(rng=rng)
    gatherer = SyntheticScalarFeedbackGatherer(rng=rng)
else:
    fragmenter = RandomFragmenter(rng=rng)
    gatherer = SyntheticGatherer(rng=rng)

if wandb.config["visibility"]["visibility"] == "partial":
    if wandb.config["visibility"]["visibility_mask_key"] == "(n-1)x(n-1)":
        #observation_function = PartialGridVisibility_HF(env, mask_key = wandb.config["visibility"]["visibility_mask_key"], feedback=config["feedback"]["type"], 
        #                                                 belief_direction=config["feedback"]["belief_direction"], belief_strength=config["feedback"]["belief_strength"], belief_early=config["feedback"]["belief_early"])
        observation_function = PartialGridVisibility(env, mask_key = wandb.config["visibility"]["visibility_mask_key"], feedback=config["feedback"]["type"])
        policy_evaluator = partial_visibility_evaluator_factory(observation_function.visibility_mask)
    elif wandb.config["visibility"]["visibility_mask_key"] == "camera":
        fragmenter = RandomFragmenter(rng=rng, get_limits=True)
        observation_function = DynamicGridVisibility(env, feedback=config["feedback"]["type"], halt=3)
        #from stealing_gridworld import DynamicGridVisibility_OJ
        #observation_function = DynamicGridVisibility_OJ(env, feedback=config["feedback"]["type"])
        policy_evaluator = camera_visibility_evaluator_factory(observation_function)

    if wandb.config["feedback"]["type"] == 'scalar':
        gatherer = NoisyObservationGathererWrapper(gatherer, observation_function)
    elif wandb.config["feedback"]["type"] == 'preference':
        gatherer = PreferenceComparisonNoisyObservationGathererWrapper(gatherer, observation_function)

    #policy_evaluator = partial_visibility_evaluator_factory(observation_function.visibility_mask)

elif wandb.config["visibility"]["visibility"] == "full":
    policy_evaluator = full_visibility_evaluator_factory()

if config["feedback"]["type"] == 'scalar':
    feedback_model = ScalarFeedbackModel(model=reward_net)
    reward_trainer = BasicScalarFeedbackRewardTrainer(
        feedback_model=feedback_model,
        loss=MSERewardLoss(),  # Will need to change this for preference learning
        rng=rng,
        epochs=wandb.config["reward_trainer"]["num_epochs"],
    )

else:
    feedback_model = PreferenceModel(reward_net)
    reward_trainer = BasicRewardTrainer(
        preference_model=feedback_model,
        loss=CrossEntropyRewardLoss(),
        rng=rng,
        epochs=wandb.config["reward_trainer"]["num_epochs"],
    )

### I think that as long as we are in ValueIteration, this can stay like this?
trajectory_generator = DeterministicMDPTrajGenerator(
    reward_fn=reward_net,
    env=env,
    rng=None,  # This doesn't work yet
    epsilon=wandb.config["trajectory_generator"]["epsilon"],
)



logger = imit_logger.configure(format_strs=["stdout", "wandb"])


def save_model_params_and_dataset_callback(reward_learner):
    data_dir = os.path.join(wandb.run.dir, "saved_reward_models")
    latest_checkpoint_path = os.path.join(data_dir, "latest_checkpoint.pt")
    latest_dataset_path = os.path.join(data_dir, "latest_dataset.pkl")
    checkpoints_dir = os.path.join(data_dir, "checkpoints")
    checkpoint_iter_path = os.path.join(checkpoints_dir, f"model_weights_iter{reward_learner._iteration}.pt")
    dataset_iter_path = os.path.join(checkpoints_dir, f"dataset_iter{reward_learner._iteration}.pkl")

    os.makedirs(checkpoints_dir, exist_ok=True)
    th.save(reward_learner.model.state_dict(), latest_checkpoint_path)
    th.save(reward_learner.model.state_dict(), checkpoint_iter_path)
    reward_learner.dataset.save(latest_dataset_path)
    reward_learner.dataset.save(dataset_iter_path)

if config["feedback"]["type"] == 'scalar':
    reward_learner = ScalarRewardLearner(
        trajectory_generator=trajectory_generator,
        reward_model=reward_net,
        num_iterations=N_ITER,
        fragmenter=fragmenter,
        feedback_gatherer=gatherer,
        feedback_queue_size=wandb.config["dataset_max_size"],
        reward_trainer=reward_trainer,
        fragment_length=wandb.config["fragment_length"],
        transition_oversampling=wandb.config["transition_oversampling"],
        initial_epoch_multiplier=wandb.config["initial_epoch_multiplier"],
        policy_evaluator=policy_evaluator,
        custom_logger=logger,
    )

else:
    reward_learner = PreferenceComparisons(
        trajectory_generator=trajectory_generator,
        reward_model=reward_net,
        num_iterations=N_ITER,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        comparison_queue_size=wandb.config["dataset_max_size"],
        reward_trainer=reward_trainer,
        fragment_length=wandb.config["fragment_length"],
        transition_oversampling=wandb.config["transition_oversampling"],
        initial_epoch_multiplier=wandb.config["initial_epoch_multiplier"],
        initial_comparison_frac=0.1,
        query_schedule="hyperbolic",
        policy_evaluator=policy_evaluator,
        custom_logger=logger,
    )

#######################################################################################################################
####################################################### Training ######################################################
#######################################################################################################################

if config["feedback"]["type"] == 'scalar':
    result = reward_learner.train(
        # Just needs to be bigger then N_ITER * HORIZON. Value iteration doesn't really use this.
        total_timesteps=10 * N_ITER * wandb.config["environment"]["horizon"],
        total_queries=N_COMPARISONS,
        callback=save_model_params_and_dataset_callback,
    )

else:
    result = reward_learner.train(
        # Just needs to be bigger then N_ITER * HORIZON. Value iteration doesn't really use this.
        total_timesteps=10 * N_ITER * wandb.config["environment"]["horizon"],
        total_comparisons=N_COMPARISONS,
        callback=save_model_params_and_dataset_callback,
    )
    