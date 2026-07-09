
# 配置系统

默认情况下，VLMEvalKit通过在`run.py`脚本中使用`--model`和`--data`参数设置模型名称（在`/vlmeval/config.py`中定义）和数据集名称（在`vlmeval/dataset/__init__.py` 或 `vlmeval/dataset/video_dataset_config.py` 中定义）来启动评估。这种方法在大多数情况下简单且高效，但当用户希望使用不同设置评估多个模型/数据集时，可能不够灵活。

为了解决这个问题，VLMEvalKit提供了一个更灵活的配置系统。用户可以在json文件中指定模型和数据集设置，并通过`--config`参数将配置文件的路径传递给`run.py`脚本。以下是一个示例配置json：

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

配置json的解释：

1. 现在我们支持两个字段：`model`和`data`，每个字段都是一个字典。字典的键是模型/数据集的名称（由用户设置），值是模型/数据集的设置。
2. 对于`model`中的项目，值是一个包含以下键的字典：
    - `class`：模型的类名，应该是`vlmeval/vlm/__init__.py`（开源模型）或`vlmeval/api/__init__.py`（API模型）中定义的类名。
    - 其他kwargs：其他kwargs是模型特定的参数，请参考模型类的定义以获取详细用法。例如，`model`、`temperature`、`img_detail`是`GPT4V`类的参数。值得注意的是，大多数模型类都需要`model`参数。
    - Tip：在位于`vlmeval/config.py`的变量`supported_VLM`中的已经被定义的模型可以作为`model`的键，而不需要填对应的值即可启动。例如，`GPT4o_20240806_T00_HIGH: {}`是等价于`GPT4o_20240806_T00_HIGH: {'class': 'GPT4V', 'model': 'gpt-4o-2024-08-06', 'temperature': 0, 'img_size': -1, 'img_detail': 'high', 'retry': 10, 'verbose': False}`。
3. 对于字典`data`，我们建议用户使用官方数据集名称作为键（或键的一部分），因为我们经常根据数据集名称确定后处理/判断设置。对于`data`中的项目，值是一个包含以下键的字典：
    - `class`：数据集的类名，应该是`vlmeval/dataset/__init__.py`中定义的类名。
    - 其他kwargs：其他kwargs是数据集特定的参数，请参考数据集类的定义以获取详细用法。通常，大多数数据集类都需要`dataset`参数。大多数视频数据集类都需要 `nframe` 或 `fps` 参数。
    - Tip：在位于`vlmeval/dataset/video_dataset_config.py`的变量`supported_video_dataset`中的已经被定义的数据集可以作为`data`的键，而不需要填对应的值即可启动。例如，`MMBench_Video_8frame_nopack: {}`是等价于`MMBench_Video_8frame_nopack: {'class': 'MMBenchVideo', 'dataset': 'MMBench-Video', 'nframe': 8, 'pack': False}`。

将示例配置json保存为`config.json`，您可以通过以下命令启动评估：

```bash
python run.py --config config.json
```

这将在工作目录`$WORK_DIR`下生成以下输出文件（格式为`{$WORK_DIR}/{$MODEL_NAME}/{$MODEL_NAME}_{$DATASET_NAME}_*`）：

- `$WORK_DIR/GPT4o_20240806_T00_HIGH/GPT4o_20240806_T00_HIGH_MME-RealWorld-Lite*`
- `$WORK_DIR/GPT4o_20240806_T10_Low/GPT4o_20240806_T10_Low_MME-RealWorld-Lite*`
- `$WORK_DIR/GPT4o_20240806_T00_HIGH/GPT4o_20240806_T00_HIGH_MMBench_DEV_EN_V11*`
- `$WORK_DIR/GPT4o_20240806_T10_Low/GPT4o_20240806_T10_Low_MMBench_DEV_EN_V11*`
......
