# Develop new Benchmark / MLLM

>  🛠️ How to implement a new Benchmark / VLM in VLMEvalKit?

## Implement a new benchmark

Example PR: **Math-Vision Benchmark** ([#292](https://github.com/open-compass/VLMEvalKit/pull/292/files))

In VLMEvalKit, benchmarks are organized as dataset classes. When you try to implement a new benchmark, you can either reuse existing dataset classes (*e.g.*, You can reuse `ImageMCQDataset` when implementing a new multi-choice benchmark), or support a new dataset class. Each dataset must have the following two member functions (either reuse the one of the parent class or implement your own):

- `build_prompt(self, line)`: The function input `line` is an integer (the sample index) or a `pd.Series` object (the raw record of the sample). The function outputs a `multi-modal message`, serving as the input of an MLLM. The `multi-modal message` is an interleaved list of multi-modal messages adopting the following format (the example includes an image and a text message): `[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`.
- `evaluate(self, eval_file,  **judge_kwargs)`: The function input `eval_file` is the MLLM prediction (typically in `.xlsx` format). If the benchmark requires an external LLM (typically GPT) for evaluation, then `judge_kwargs` can pass the arguments for the LLM. The function outputs the benchmark evaluation results (metrics) in the form of `dict` or `pd.DataFrame`.

We then brief the typical steps to implement a new benchmark under VLMEvalKit:

### 1. Prepare your benchmark tsv file

Currently, we organize a benchmark as one single TSV file. During inference, the data file will be automatically downloaded from the definited `DATASET_URL` link to `$LMUData` file (default path is `$HOME/LMUData`, if not set explicitly). You can upload the prepared TSV file to a downloadable address (e.g., Huggingface) or send it to us at <opencompass@pjlab.org.cn>. We will assist in uploading the dataset to the server. You can also customize `LMUData` path in the environment variable `LMUData=/path/to/your/data`.

The contents of the TSV file consist of:

| Dataset Name \ Fields                   | index | image | image_path | question | hint | multi-choice<br>options | answer | category | l2-category | split |
| --------------------------------------- | ----- | ----- | ---------- | -------- | ---- | ----------------------- | ------ | -------- | ----------- | ----- |
| MMBench_DEV_[CN/EN]                     | ✅     | ✅     |            | ✅        | ✅    | ✅                       | ✅      | ✅        | ✅           | ✅     |
| MMBench_TEST_[CN/EN]                    | ✅     | ✅     |            | ✅        | ✅    | ✅                       |        | ✅        | ✅           | ✅     |
| CCBench                                 | ✅     | ✅     |            | ✅        |      | ✅                       | ✅      | ✅        |             |       |
| SEEDBench_IMG                           | ✅     | ✅     |            | ✅        |      | ✅                       | ✅      | ✅        |             |       |
| MME                                     | ✅     | ✅     |            | ✅        |      |                         | ✅      | ✅        |             |       |
| MMVet                                   | ✅     | ✅     |            | ✅        |      |                         | ✅      | ✅        |             |       |
| MMMU_DEV_VAL                            | ✅     | ✅     | ✅          | ✅        |      | ✅                       | ✅      | ✅        | ✅           | ✅     |
| COCO_VAL                                | ✅     | ✅     |            |          |      |                         | ✅      |          |             |       |
| OCRVQA_[TEST/TESTCORE]                  | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |
| TextVQA_VAL                             | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |
| VCR_[EN/ZH]\_[EASY/HARD]\_[ALL/500/100] | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |
| MMMB_[en/cn/pt/ar/tr/ru] | ✅     | ✅     |            | ✅        | ✅     | ✅     | ✅      | ✅         |             |✅       |
| MMBench_dev_[en/cn/pt/ar/tr/ru] | ✅     | ✅     |            | ✅        | ✅     | ✅     | ✅      | ✅         | ✅            |✅       |

<div align="center"><b>Table 1. TSV fields of supported datasets.</b></div>

**Intro to mandatory fields in the `TSV` file:**

- **index:** Integer, Unique for each line in `tsv`
- **image:** The base64 of the image, you can use APIs implemented in `vlmeval/smp/vlm.py` for encoding and decoding:
  - Encoding: `encode_image_to_base64 `(for PIL Image) / `encode_image_file_to_base64` (for image file path)
  - Decoding: `decode_base64_to_image`(for PIL Image) / `decode_base64_to_image_file` (for image file path)
- **question**: The question corresponding to the image, a string
- **answer**: The answer to the question, a string. The `test` split does not need this field

### 2. Cutomize your benchmark prompt

`ImageBaseDataset` defines the default prompt format. If you need to add prompts specific to the dataset or input data in the `Interleave` format to the model, you can implement this through the `build_prompt(line)` function. This function takes a line from a TSV file as input, containing fields such as index, image, question, etc. The function returns a dictionary list of multimodal messages `msg` in the format `[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`, including the image path and the text prompt to be input into VLMs. For interleave type inputs, you can directly place the dictionary of the image path at the image token position.

### 3. Cutomize your benchmark metrics

To add evaluation for a new benchmark, you need to customize a class object to implement the dataset’s metrics calculation. Multimodal datasets inherit from the `ImageBaseDataset` object in `vlmeval/dataset/image_base.py`. The TYPE defines the type of dataset, `DATASET_URL` is the download address of the dataset, and `DATASET_MD5` is the MD5 checksum for consistency checking of the dataset file.

In this class, **you need to implement** the `evaluate(eval_file, **judge_kwargs)` class function to calculate metrics and output results for the custom dataset. The function input `eval_file` is the path to the model prediction results file `{model_name}_{dataset}.xlsx`. This file can be read as a pandas.DataFrame using the `load(eval_file)` method, containing fields such as index, question, answer, category, prediction, etc. The judge_kwargs will pass a dictionary related to evaluation, such as the name of the `judge model`, the number of API request threads, etc. **The return value** of the function is the calculated accuracy and other metrics, formatted as a dictionary composed of lists, organized into a pandas.DataFrame.

## Implement a new model

Example PR: **Support LLaVA-Next-Interleave** ([#294](https://github.com/open-compass/VLMEvalKit/pull/294))

**1. Support `generate_inner` API (mandatory).**

All existing models are implemented in `vlmeval/vlm`. For a minimal model, your model class **must implement the method** `generate_inner(msgs, dataset=None)`. In this function, you feed a multi-modal message to your VLM and return the VLM prediction (which is a string). The optional argument `dataset` can be used as the flag for the model to switch among various inference strategies.

The multi-modal messages `msgs` is a list of dictionaries, each dictionary has two keys: type and value:
- `type`: We currently support two types, choices are ["image", "text"].
- `value`: When type=='text' , the value is the text message (a single string); when type=='image', the value can be the local path of an image file, or the image URL.

Currently a multi-modal message may contain arbitrarily interleaved images and texts. If your model do not support that, a practice can be taking the 1st image and concatenated text messages as the input. You can set the `INTERLEAVE = False` in your model class and use `self.message_to_promptimg(message, dataset=dataset)` to build your prompt and the first image's path.

Here are some examples of multi-modal messages:

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

For convenience sake, we also support to take a list of string as inputs. In that case, we will check if a string is an image path or image URL and automatically convert it to the list[dict] format:

```python
IMAGE_PTH = 'assets/apple.jpg'
IMAGE_URL = 'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'
msg1 = [IMAGE_PTH, 'What is in this image?']
msg2 = [IMAGE_URL, IMAGE_URL,  'How many apples are there in these images?']
response = model.generate(msg1)
```

**Support Custom Prompt (optional).**

Besides, your model can support **custom prompt building** by implementing two optional methods: `use_custom_prompt(dataset)` and `build_prompt(line, dataset=None)`.

Both functions take the dataset name as the input：

-  `use_custom_prompt(dataset)` returns a boolean flag, indicating whether the model should use the custom prompt building strategy.
- If `use_custom_prompt(dataset)` returns True, `build_prompt(line, dataset)` should return a customly bulit multimodal message for the corresponding `dataset`, given `line`, which is a dictionary that includes the necessary information of a data sample. If `use_custom_prompt(dataset)` returns False, the default prompt building strategy will be used.

**Support multi-turn chatting (optional).**

You can also support the multi-turn chatting and evaluation with your VLM by supporting the `chat_inner(message, dataset)` function. The function outputs a single string response, and the `message` is a list of chat history, following the below format.

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

### Example PRs:

- VLM that doesn't support interleaved images and texts, and does not use custom prompts: [[Model] Support glm-4v-9b](https://github.com/open-compass/VLMEvalKit/pull/221)
- VLM that supports interleaved images and texts and custom prompts: [Add MiniCPM-Llama3-V-2.5](https://github.com/open-compass/VLMEvalKit/pull/205)
- VLM API: [Feature add glmv](https://github.com/open-compass/VLMEvalKit/pull/201)

## Contribute to VLMEvalKit

If you want to contribute codes to **VLMEvalKit**, please do the pre-commit check before you submit a PR. That helps to keep the code tidy.

```bash
# Under the directory of VLMEvalKit, install the pre-commit hook:
pip install pre-commit
pre-commit install
pre-commit run --all-files
# Then you can commit your code.
```
