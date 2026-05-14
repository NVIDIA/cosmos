# Fractal (Google RT-1) experiment — Cosmos3 2B pretrained base
#
# Base experiment (policy mode) + mode variants (fd, id, policy, i2v, joint).

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action._experiment_helpers import register_modes
from configs.base.experiment.action.pretrained_config.cosmos3_2b import (
    make_2b_experiment,
)
from cosmos.data.vfm.action.fractal import FractalLeRobotDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Dataset definition (shared across all modes)
# ---------------------------------------------------------------------------
FRACTAL_DATASET = [
    L(dataset_entry)(
        name="fractal",
        dataset=L(FractalLeRobotDataset)(chunk_length=16, split="train"),
        ratio=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, 4k iters (policy mode default)
# ---------------------------------------------------------------------------
fractal = make_2b_experiment(
    exp_name="fractal",
    datasets=FRACTAL_DATASET,
    training_iterations=4_000,
)
fractal["job"]["group"] = "fractal"

cs.store("fractal", fractal, group="experiment", package="_global_")
register_modes(cs, "fractal", fractal, dataloader_key="action_data")
