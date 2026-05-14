# Training Telemetry

Training Telemetry is a utility [library](https://gitlab-master.nvidia.com/ai-efficiency/training_telemetry) for tracking and recording training metrics, events, and performance data during model training. The code in this folder is a wrapper for this library, primarily to adapt it to existing imaginaire4 code through a function decorator, a callback implementation, and context managers.

The code in this folder also gracefully handles cases where the library is not available or training telemetry is disabled.

The library provides a standardized way to monitor and analyze the training process that backend infrastructure components such as Heimdall can use to manage applications and report training KPIs.

This is done by either logging to stdout on rank 0, creating a JSON file for each rank, or both.

The library also inserts [NVTX marks](https://github.com/NVIDIA/NVTX), which by default are a no-op, but when running with the NVIDIA Nsight profiler are useful to know where the code is spending time without relying on the slower torch profiler.

## Enabling the library

The library must be installed in the container or manually if not already available:

```bash
pip install training-telemetry --index-url https://__token__:{gitlab_token}@gitlab-master.nvidia.com/api/v4/projects/166461/packages/pypi/simple
```

where `{gitlab_token}` is a GitLab token with read access to the [package registry](https://gitlab-master.nvidia.com/ai-efficiency/training_telemetry/-/packages). Anyone in NVIDIA with a GitLab account should have access to this package registry. Therefore, if you have an existing GitLab token, it should work. If you do not have one, follow these [instructions](https://docs.gitlab.com/user/profile/personal_access_tokens/) to create one. The token needs to have at least `read_repository` access, or higher `read` access.

To enable the library, set this environment variable:

```bash
export ENABLE_TELEMETRY=true
```

If this variable is set to true and the library is not installed, then an error will be logged, and telemetry will be disabled.

To specify which telemetry backends to use, set the following environment variable with a list of comma separated backends:

```bash
export TELEMETRY_BACKENDS=logger,nvtx,file
```

This will enable all 3 backends but by default, only the logger backend is enabled.

It is recommended to enable training telemetry with the logger backend when running the training process with Heimdall, formerly known as APS.

You should also consider enabling the nvtx backend when running the training process with NVIDIA Nsight Systems profiler.

The file backend will make every rank generate a json file with telemetry events, which could be processed to extract KPIs for each rank.

## Features

The library uses the following mechanisms to intercept training events:

* Top-level function decorator
* Context managers
* Callback implementation

### Top-level function decorator

This is defined in [telemetry.py](telemetry.py) and performs the library initialization. It also overrides the config to inject the callback.

Most train.py files have already been annotated, but if any are missing, simply add `@telemetry.monitor` to the main launch functions.

Training telemetry can be imported as:

```
from cosmos.utils.training_telemetry import telemetry
```

### Context managers

They are defined in [context_managers.py](./context_managers.py) and are called by higher-level context managers that also wrap other telemetry code (one-logger and timers), defined in [../context_managers.py](../context_managers.py). They are as follows:

* `data_loader_init`: can be used to wrap code that initializes the data loader
* `model_init`: can be used to wrap code that initializes a model
* `distributed_init`: can be used to wrap code that initializes distributed communication

The context managers capture events that are not currently available in the callback.

### Callback

This is defined in [callback.py](./callback.py) and implements most of the functions defined by the Imaginaire4 callback to capture the remaining training events.

## Training events

The following training events are captured:

* Application running time and other information:
  * Timezone
  * Node name
  * World size
  * Rank
  * Total iterations
  * Checkpoint strategy and other information
  * More data can be added in [telemetry.py](telemetry.py)
* Data loader init
* Model init
* Distributed init
* Optimizer init
* Entire training loop duration
* Training step duration (NVTX only)
* Model forward (NVTX only)
* Model backward (NVTX only)
* Data loading (NVTX only)
* Training iterations - every `trainer.logging_iter` the following is logged:
  * Avg iteration time
  * Avg forward time
  * Avg backward time
  * Avg data loading time
  * Current iteration
  * Number of iterations since previous logging
  * Loss
  * Batch size
  * FLOPS - TODO (?)
* Validation
  * Total - logged
  * Single step (NVTX only)  
* Checkpoint load
* Checkpoint save

## Output

The library outputs to stdout along with the main logger on rank zero, but also to a separate `stdout.log` file in the `telemetry` folder in the main output location.

The library also outputs one JSON file per rank in the `telemetry` folder in the main output location. In future, these json files could be ingested to be processed instead of logs, or analyzed to generate rank-level latency and other performance KPIs.

As already mentioned, the library also inserts NVTX code marks.
