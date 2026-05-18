# RoboMIND experiment — Cosmos3 2B pretrained base
#
# Base experiment (policy mode) + mode variants (fd, id, policy, i2v, joint).
# Plus embodiment-type variants and keep-aspect-ratio variants.


from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action._experiment_helpers import (
    register_embodiment_type,
    register_modes,
)
from configs.base.experiment.action.midtrain_config.action_datasets import (
    DATASET_ROBOMIND_FRANKA_480,
    DATASET_ROBOMIND_FRANKA_DUAL_480,
)
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from cosmos.data.vfm.action.robomind_ur_dataset import RoboMINDURDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Dataset definition (shared across all modes)
# ---------------------------------------------------------------------------
ROBOMIND_DATASET = [
    L(dataset_entry)(
        name="robomind",
        dataset=L(RoboMINDURDataset)(chunk_length=16, split="train"),
        ratio=1.0,
    ),
]

ROBOMIND_FRANKA_DATASET = [
    DATASET_ROBOMIND_FRANKA_480,
]

ROBOMIND_FRANKA_DUAL_DATASET = [
    DATASET_ROBOMIND_FRANKA_DUAL_480,
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, 4k iters (policy mode default)
# ---------------------------------------------------------------------------
robomind = make_2b_experiment(
    exp_name="robomind",
    datasets=ROBOMIND_DATASET,
    training_iterations=4_000,
)
robomind["job"]["group"] = "robomind"

cs.store("robomind", robomind, group="experiment", package="_global_")
register_modes(cs, "robomind", robomind, dataloader_key="action_data")


# ---------------------------------------------------------------------------
# Embodiment-type variants
# ---------------------------------------------------------------------------
robomind_franka = make_2b_experiment(
    exp_name="robomind_franka",
    datasets=ROBOMIND_FRANKA_DATASET,
    training_iterations=4_000,
)
robomind_franka["job"]["group"] = "robomind"
cs.store("robomind_franka", robomind_franka, group="experiment", package="_global_")
register_modes(cs, "robomind_franka", robomind_franka, dataloader_key="action_data")

robomind_franka_dual = make_2b_experiment(
    exp_name="robomind_franka_dual",
    datasets=ROBOMIND_FRANKA_DUAL_DATASET,
    training_iterations=4_000,
)
robomind_franka_dual["job"]["group"] = "robomind"
cs.store("robomind_franka_dual", robomind_franka_dual, group="experiment", package="_global_")
register_modes(cs, "robomind_franka_dual", robomind_franka_dual, dataloader_key="action_data")

robomind_ur = register_embodiment_type(cs, "robomind", "ur", "robomind-ur", robomind, dataloader_key="action_data")
register_modes(cs, "robomind_ur", robomind_ur, dataloader_key="action_data")
