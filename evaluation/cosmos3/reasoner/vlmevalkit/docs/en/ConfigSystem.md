# Config System

By default, VLMEvalKit launches the evaluation by setting the model name(s) (defined in `/vlmeval/config.py`) and dataset name(s) (defined in `vlmeval/dataset/__init__.py` or `vlmeval/dataset/video_dataset_config.py`) in the `run.py` script with the `--model` and `--data` arguments. Such approach is simple and efficient in most scenarios, however, it may not be flexible enough when the user wants to evaluate multiple models / datasets with different settings.

To address this, VLMEvalKit provides a more flexible config system. The user can specify the model and dataset settings in a json file, and pass the path to the config file to the `run.py` script with the `--config` argument. Here is a sample config json:

```json
{
    "model": {
        "GPT4o_20240806_T00_HIGH": {
            "class": "GPT4V",
            "model": "gpt-4o-2024-08-06",
            "temperature": 0,
            "img_detail": "high"
        },
        "GPT4o_20240806_T10_Low": {
            "class": "GPT4V",
            "model": "gpt-4o-2024-08-06",
            "temperature": 1.0,
            "img_detail": "low"
        },
        "GPT4o_20241120": {}
    },
    "data": {
        "MME-RealWorld-Lite": {
            "class": "MMERealWorld",
            "dataset": "MME-RealWorld-Lite"
        },
        "MMBench_DEV_EN_V11": {
            "class": "ImageMCQDataset",
            "dataset": "MMBench_DEV_EN_V11"
        },
        "MMBench_Video_8frame_nopack":{},
        "Video-MME_16frame_subs": {
            "class": "VideoMME",
            "dataset": "Video-MME",
            "nframe": 16,
            "use_subtitle": true
        }
    }
}
```

Explanation of the config json:

1. Now we support two fields: `model` and `data`, each of which is a dictionary. The key of the dictionary is the name of the model / dataset (set by the user), and the value is the setting of the model / dataset.
2. For items in `model`, the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class name defined in `vlmeval/vlm/__init__.py` (open-source models) or `vlmeval/api/__init__.py` (API models).
    - Other kwargs: Other kwargs are model-specific parameters, please refer to the definition of the model class for detailed usage. For example, `model`, `temperature`, `img_detail` are arguments of the `GPT4V` class. It's noteworthy that the `model` argument is required by most model classes.
    - Tip: The defined model in the `supported_VLM` of `vlmeval/config.py` can be used as a shortcut, for example, `GPT4o_20241120: {}` is equivalent to `GPT4o_20241120: {'class': 'GPT4V', 'model': 'gpt-4o-2024-11-20', 'temperature': 0, 'img_size': -1, 'img_detail': 'high', 'retry': 10, 'verbose': False}`
3. For the dictionary `data`, we suggest users to use the official dataset name as the key (or part of the key), since we frequently determine the post-processing / judging settings based on the dataset name. For items in `data`, the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class name defined in `vlmeval/dataset/__init__.py`.
    - Other kwargs: Other kwargs are dataset-specific parameters, please refer to the definition of the dataset class for detailed usage. Typically, the `dataset` argument is required by most dataset classes. It's noteworthy that the `nframe` argument or `fps` argument is required by most video dataset classes.
    - Tip: The defined dataset in the `supported_video_datasets` of `vlmeval/dataset/video_dataset_config.py` can be used as a shortcut, for example, `MMBench_Video_8frame_nopack: {}` is equivalent to `MMBench_Video_8frame_nopack: {'class': 'MMBenchVideo', 'dataset': 'MMBench-Video', 'nframe': 8, 'pack': False}`.
Saving the example config json to `config.json`, you can launch the evaluation by:

```bash
python run.py --config config.json
```

That will generate the following output files under the working directory `$WORK_DIR` (Following the format `{$WORK_DIR}/{$MODEL_NAME}/{$MODEL_NAME}_{$DATASET_NAME}_*`):

- `$WORK_DIR/GPT4o_20240806_T00_HIGH/GPT4o_20240806_T00_HIGH_MME-RealWorld-Lite*`
- `$WORK_DIR/GPT4o_20240806_T10_Low/GPT4o_20240806_T10_Low_MME-RealWorld-Lite*`
- `$WORK_DIR/GPT4o_20240806_T00_HIGH/GPT4o_20240806_T00_HIGH_MMBench_DEV_EN_V11*`
- `$WORK_DIR/GPT4o_20240806_T10_Low/GPT4o_20240806_T10_Low_MMBench_DEV_EN_V11*`
...
