# OneLogger

`You must use OneLogger or we will not let you run any jobs greater than one node.` OneLogger is a tool that measures our end-to-end job efficiency and the measures will be used for future resources governance. For all the training jobs we run in NVIDIA clusters, we need to enable OneLogger to report our utilization. Currently, slurm clusters (`aws-iad-cs-001`, `cw-pdx-cs-001`) are supported.

* Related docs
  * [Example integration](https://gitlab-master.nvidia.com/hwinf-dcm/one-logger-utils)

## Usage

* OneLogger logs the resource utilization by using wandb. Please set `WANDB_API_KEY` environment variable.
```bash
WANDB_API_KEY=xxx torchrun --nproc_per_node 1 -m scripts.train --config projects/dv_diffusion/config/cifar10.py
```

* By default, OneLogger is automatically enabled by detecting the job environment - Slurm, NGC, Run:AI, and Local. Environment variable `ENABLE_ONELOGGER` is there to provide explicit control by the value between `TRUE` and `FALSE`. We assume `ENABLE_ONELOGGER` is set from [launcher](packages/launcher/README.md) Executor and users will always submit imaginaire4 jobs with [launcher](packages/launcher/README.md).
  * NGC: `launcher.NGCExecutor` always sets `ENABLE_ONELOGGER=FALSE`.
  * Slurm: `launcher.SlurmExecutor` sets `ENABLE_ONELOGGER=TRUE` by default. It will follow user input if provided.
  * Run:AI: `launcher.RunAIExecutor` always sets `ENABLE_ONELOGGER=FALSE`.
  * Local: `launcher.Executor` does not set `ENABLE_ONELOGGER` and will follow user input if provided.

* OneLogger supports 2 modes - `production` and `test`.
  * By default, `launcher.SlurmExecutor` sets `ONE_LOGGER_JOB_CATEGORY=production`.
  * If necessary, users are allowed to set `ONE_LOGGER_JOB_CATEGORY=test` through launcher. `test` mode is for jobs with abnormal behaviors such as interactive debugging jobs.

* OneLogger is implemented in imaginaire4 in the form of Callback. `OneLoggerCallback` is defined in [imaginaire/utils/callback.py](imaginaire/utils/callback.py). `OneLoggerCallback` is added to the config callbacks before initializing `ImaginaireTrainer`.

`scripts/train.py`
```python
config = override_one_logger_callback(config)
```

`imaginaire/trainer.py`
```python
# OneLogger - initialize one_logger before instantiating CallBackGroup
enable_one_logger = os.environ.get("ENABLE_ONELOGGER", "FALSE").lower() == "true"
if enable_one_logger:
    initialize_one_logger_from_imaginaire_config(config)
# Initialize the callback functions.
self.callbacks = callback.CallBackGroup(config=config, trainer=self)
```

## Requirements

* W&B

As OneLogger relies on wandb to log efficiency graphs, everyone who is using OneLogger / training models should belong to the corresponding wandb team, [hwinf_dcm](https://wandb.ai/hwinf_dcm). Please make sure you can find your name from the users list. If you are not there, please ask in slack channel [#hwinf-mlwfo-e2e-support](https://nvidia.enterprise.slack.com/archives/C0730DFM6UC) to onboard yourself.

* python dependency (already included in imaginaire4 dockers images)
```bash
pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger
```

## Implementation details

* New files are added for OneLogger utility.
  * cosmos/utils/one_logger/one_logger_global_vars.py
  * cosmos/utils/one_logger/one_logger_utils.py
* global variable `one_logger` is initialized from `ImaginaireTrainer` and most logging functions are implemented in the form of callbacks.
* `OneLoggerCallback` is implemented with trainer callback functions.
  * `__init__()`
  * `on_train_start()`
  * `on_training_step_start()`
  * `on_optimizer_init_start()` (**new**)
  * `on_optimizer_init_end()` (**new**)
  * `on_training_step_end()`
  * `on_validation_start()`
  * `on_validation_step_start()`
  * `on_validation_step_end()`
  * `on_validation_end()`
  * `on_load_checkpoint_start()` (**new**)
  * `on_load_checkpoint_end()` (**new**)
  * `on_save_checkpoint_start()` (**new**)
  * `on_save_checkpoint_end()` (**new**)
  * `on_save_checkpoint_success()` (**new**)
  * `on_train_end()`
  * `on_app_end()` (**new**)
* Above callbacks marked (**new**) are new callback functions that did not exist before.
* Exceptions are
  * Dataloader callbacks are called from the context manager data_loader_init:
    * `one_logger.on_dataloader_init_start()`
    * `one_logger.on_dataloader_init_end()`
  * model callbacks are called from the context manager model_init:
    * `one_logger.on_model_init_start()`
    * `one_logger.on_model_init_end()`
* **Important**
  * `app_tag` and `app_tag_run_name` are the variables necessary for resource efficiency tracking.
  * Current formats are
```python
app_tag_run_name = f"{config.job.project}/{config.job.group}/{config.job.name}"
app_tag_run_version = "0.0.0"  # hard-coded app_tag_run version.
app_tag = f"{app_tag_run_name}/{job_size}/ENV_{job_environment}_GPU_{gpu_name}"
```

OneLogger implementation is tested with the following command on Slurm, NGC, and Local environments.
```bash
WANDB_API_KEY=xxx torchrun --nproc_per_node 1 -m scripts.train --config projects/dv_diffusion/config/cifar10.py
```
