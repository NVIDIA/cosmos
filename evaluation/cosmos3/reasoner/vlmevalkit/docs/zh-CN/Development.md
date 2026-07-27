# 🛠️ 如何在 VLMEvalKit 中实现一个新的 Benchmark 或多模态模型（VLM）

## 实现一个新的 benchmark

示例 PR: **添加 Math-Vision Benchmark** ([#292](https://github.com/open-compass/VLMEvalKit/pull/292/files))

目前在 VLMEvalKit 中，benchmark 以数据集类的形式呈现，当你新增一个 benchmark 时，你可以选择复用现有的数据集类 (如单选题 benchmark 可复用 `ImageMCQDataset`)，或是实现新的数据集类。你的数据集类必须支持以下两种方法 (复用父类或自行实现):

- `build_prompt(self, line)`: 方法输入 `line` 类型为 int (对应数据 index) 或 `pd.Series` (对应数据原始 record)。方法输出一条 `multi-modal message` 作为多模态模型输入，`multi-modal message` 是一个图文交错的列表，如以下格式 (一图一文): `[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`。
- `evaluate(self, eval_file, **judge_kwargs)`: 方法输入 `eval_file` 为多模态模型的预测结果 (多以 `.xlsx` 格式存在)，如 benchmark evaluation 需要大语言模型 (一般为 GPT) 辅助，则 `judge_kwargs` 传入大语言模型的参数。方法输出 benchmark 的评测结果，以 `dict` 或 `pd.DataFrame` 的形式。

以下，我们简述新增数据集的通常步骤：

### 1. TSV 数据文件准备 (图文评测集)

目前，我们将每一个 benchmark 数据集设置为一个单独的 TSV 文件。在推理过程中，数据文件将从数据集定义的 `DATASET_URL` 链接地址自动下载到 `$LMUData` 中（如果没有明确设置的话，默认路径是 `$HOME/LMUData`）。你可以将准备好的 TSV 文件上传到一个可下载的地址（如：huggingface），或发送给我们 <opencompass@pjlab.org.cn>，我们将帮助上传数据集到服务器中。此外，你也可以在环境变量中自定义设置下载路径 `LMUData=/path/to/your/data`。

TSV 文件中的内容组成为：

| 数据集名称 \ 字段  | index | image | image_path | question | hint | multi-choice<br>options | answer | category | l2-category | split |
| ---------------------- | ----- | ----- | ---------- | -------- | ---- | ----------------------- | ------ | -------- | ----------- | ----- |
| MMBench_DEV_[CN/EN]    | ✅     | ✅     |            | ✅        | ✅    | ✅                       | ✅      | ✅        | ✅           | ✅     |
| MMBench_TEST_[CN/EN]   | ✅     | ✅     |            | ✅        | ✅    | ✅                       |        | ✅        | ✅           | ✅     |
| CCBench                | ✅     | ✅     |            | ✅        |      | ✅                       | ✅      | ✅        |             |       |
| SEEDBench_IMG          | ✅     | ✅     |            | ✅        |      | ✅                       | ✅      | ✅        |             |       |
| MME                    | ✅     | ✅     |            | ✅        |      |                         | ✅      | ✅        |             |       |
| MMVet                  | ✅     | ✅     |            | ✅        |      |                         | ✅      | ✅        |             |       |
| MMMU_DEV_VAL           | ✅     | ✅     | ✅          | ✅        |      | ✅                       | ✅      | ✅        | ✅           | ✅     |
| COCO_VAL               | ✅     | ✅     |            |          |      |                         | ✅      |          |             |       |
| OCRVQA_[TEST/TESTCORE] | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |
| TextVQA_VAL            | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |
| VCR_[EN/ZH]\_[EASY/HARD]_[ALL/500/100]            | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |

<div align="center"><b>表 1. 支持的数据集的 TSV 字段。</b></div>

**TSV 中必须字段的介绍：**

- **index:** 一个整数，`tsv` 中每一行的唯一标识
- **image:** 图片的 base64 编码，你可以使用 `vlmeval/smp/vlm.py` 中实现的API进行编码和解码：
    - 编码：`encode_image_to_base64`（对于PIL Image）/ `encode_image_file_to_base64`（对于图片文件路径）
    - 解码：`decode_base64_to_image`（对于PIL Image）/ `decode_base64_to_image_file`（对于图片文件路径）
- **question:** 针对图像所提取出的问题，类型为字符串
- **answer:** 问题的答案，类型为字符串，Test 集可缺失这一字段

### 2. 自定义数据集的 prompt 构建

`ImageBaseDataset` 定义了默认的 prompt 格式。如果需要针对数据集添加 prompt，或给模型输入 `Interleave` 的数据格式，可以通过 `build_prompt(line)` 函数实现。该函数输入为，每次给定 TSV 文件中的一行，包含 index, image, question 等内容作为 line。该函数将返回一个多模态消息 `msg` 的字典列表 `[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`，包括图片路径和将被输入到 VLMs 的文本 prompt。对于 interleave 类型输入，可以直接将图片路径的字典放置到 image token 位置。

### 3. 自定义数据集的指标实现

增加对 benchmark 的评测需要自定义一个该数据集的 class 对象，从而实现数据集的指标计算。图文多模态数据集均继承自 `vlmeval/dataset/image_base.py` 中的 `ImageBaseDataset` 对象。其中 `TYPE` 定义了数据集的类型；`DATASET_URL` 为数据集的下载地址；`DATASET_MD5` 为数据集文件的 md5 一致性编码检查。

在 class 中**需要实现** `evaluate(eval_file, **judge_kwargs)` 类函数，对自定义的数据集结果进行指标计算和结果输出。函数输入 `eval_file` 为模型预测结果 `{model_name}_{dataset}.xlsx` 的路径。可以通过 `load(eval_file)` 文件将其读取为 panda.DataFrames 类型，其中包含 index, question, answer, category, prediction 等字段。`judge_kwargs` 参数将传递一个评测相关的字典，如：judge 模型的名称，api 请求线程数等。**函数的返回值**为评估完成的准确度等指标，其格式为由 list 组成的字典，并组织成 panda.DataFrames 类型。

## 实现一个新的模型

示例 PR: **支持 LLaVA-Next-Interleave** ([#294](https://github.com/open-compass/VLMEvalKit/pull/294))

**1. 支持 `generate_inner` API (必须)**

现有所有的模型都在 `vlmeval/vlm` 中实现。对于一个最基本的模型，你的模型类**应该实现方法** `generate_inner(msgs, dataset=None)`。这个函数将向 VLM 输入一个多模态数据，并返回 VLM 的预测（一个字符串）。可选参数 `dataset` 可以用作模型在不同推理策略之间切换的标志。

其中多模态消息 `msgs` 是一个字典列表，每个字典有两个键：类型和值：
- `type`：我们目前支持两种类型，选项是 ["image", "text"]。
- `value`：当类型为 `text` 时，值是文本消息（一个字符串）；当类型为 `image` 时，值可以是图像文件的本地路径，或者是图像的URL。

> 目前，一个多模态消息可能包含任意交错的图像和文本。如果你的模型不支持这一点，我们推荐的做法是取第一张图像和连接的文本消息作为模型的输入。你可以在模型的 class 中设置 `INTERLEAVE = False` 并调用 `self.message_to_promptimg(message, dataset=dataset)` 函数来获取你的 prompt 和第一张图片的地址。

一些多模态消息的例子:

```python
IMAGE_PTH = 'assets/apple.jpg'
IMAGE_URL = 'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'
msg1 = [
    dict(type='image', value=IMAGE_PTH),
    dict(type='text', value='What is in this image?')
]
msg2 = [
    dict(type='image', value=IMAGE_URL),
    dict(type='image', value=IMAGE_URL),
    dict(type='text', value='How many apples are there in these images?')
]
response = model.generate(msg1)
```

为了方便起见，我们还支持接受字符串列表作为输入。在这种情况下，我们将检查一个字符串是图像路径还是图像 URL，并自动将其转换为 `list[dict]` 格式：

```python
IMAGE_PTH = 'assets/apple.jpg'
IMAGE_URL = 'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'
msg1 = [IMAGE_PTH, 'What is in this image?']
msg2 = [IMAGE_URL, IMAGE_URL,  'How many apples are there in these images?']
response = model.generate(msg1)
```

**2. 支持自定义提示词构建 (可选)**

此外，你的模型可以通过实现两个可选方法来支持自定义提示构建：`use_custom_prompt(dataset)` 和 `build_prompt(line, dataset=None)`。

- `use_custom_prompt(dataset)` 将返回一个布尔值，指示模型是否应使用自定义提示构建策略。
- 如果`use_custom_prompt(dataset)`返回 True，`build_prompt(line, dataset)` 应该为相应的数据集返回一个自定义构建的多模态消息，line 数据是一个包含数据样本所需信息的字典。如果`use_custom_prompt(dataset)` 返回False，则将使用默认的 prompt 构建策略。

**3. 支持多轮对话 (可选)**

你可以通过支持 `chat_inner(message, dataset)` API 为你的模型新增多轮对话功能并兼容多轮对话评测。这个 API 输出一个字符串型回复，`message` 包含一个聊天记录的列表，格式如下：

```python
# Assume msg1, msg2, msg3, ... are multi-modal messages following the previously described format
# `chat_inner` take the following chat history list as input:
message = [
    dict(role='user', content=msg1),
    dict(role='assistant', content=msg2),
    dict(role='user', content=msg3),
    dict(role='assistant', content=msg4),
	......
    dict(role='user', content=msgn),
]
# `message` should contain an odd number of chat utterances, the role of utterances should be interleaved "user" and "assistant", with the role of the last utterance to be "user".
# The chat function will call `chat_inner`
response = model.chat(message)
```

### 示例 PRs：

- 不支持交错的图像和文本，且不使用自定义提示的VLM：[[模型] 支持 glm-4v-9b](https://github.com/open-compass/VLMEvalKit/pull/221)
- 支持交错的图像和文本及自定义提示的VLM：[添加 MiniCPM-Llama3-V-2.5](https://github.com/open-compass/VLMEvalKit/pull/205)
- VLM API：[特征添加 glmv](https://github.com/open-compass/VLMEvalKit/pull/201)

## 为 VLMEvalKit 贡献代码

如果你想为 **VLMEvalKit** 贡献代码，请在提交PR之前进行预提交检查。这有助于保持代码整洁。

```bash
# 在VLMEvalKit的目录下，安装预提交 hook:
pip install pre-commit
pre-commit install
pre-commit run --all-files
# 然后提交你的代码。
```
