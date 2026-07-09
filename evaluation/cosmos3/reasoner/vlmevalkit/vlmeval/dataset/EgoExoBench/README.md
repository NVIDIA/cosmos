# EgoExoBench

This is the official repository of [EgoExoBench: A
Benchmark for First- and Third-person View Video
Understanding in MLLMs]()

## 📊 Benchmark Overview

**EgoExoBench** is a large-scale benchmark designed to evaluate cross-view video understanding in multimodal large language models (MLLMs). It contains paired egocentric–exocentric videos and over **7,300 multiple-choice questions** across **11 subtasks**, covering three key dimensions of ego–exo reasoning:

* **Ego-Exo Relation**
* **Ego-Exo View Transition**
* **Ego-Exo Temporal Reasoning**

## 📝 Data Preparation

### Video Data

EgoExoBench builds upon six publicly available ego–exo datasets.

* [Ego-Exo4D](https://ego-exo4d-data.org/)
* [LEMMA](https://sites.google.com/view/lemma-activity)
* [EgoExoLearn](https://huggingface.co/datasets/hyf015/EgoExoLearn)
* [TF2023](https://github.com/ziweizhao1993/PEN)
* [EgoMe](https://huggingface.co/datasets/HeqianQiu/EgoMe)
* [CVMHAT](https://github.com/RuizeHan/CVMHT)

The script will automatically download the processed video data, **except Ego-Exo4D**, due to license restrictions. You need to manually download it from the [official website](https://ego-exo4d-data.org/) and organize it as shown below.

If you prefer to download all datasets manually, you can simply create empty `processed_videos/` and `processed_frames/` folders and organize the datasets in the following structure:

```
[LMUData]/videos/EgoExoBench
├── CVMHAT/
│   └── data/
├── EgoExo4D/
│   └── takes/
├── EgoExoLearn/
├── EgoMe/
├── LEMMA/
├── TF2023/
│   └── data/
├── processed_frames/
└── processed_videos/
```
### Multiple-Choice Questions (MCQs)

The script will automatically download the EgoExoBench **multiple-choice questions (MCQs)** file from this [link](https://huggingface.co/datasets/Heleun/EgoExoBench_MCQ).

## 🚀 Model Evaluation

Use the following commands to evaluate your VLMs on EgoExoBench:

```shell
# For lightweight vision-language models
torchrun --nproc-per-node=1 run.py \
    --data EgoExoBench_MCQ \
    --model Qwen2.5-VL-7B-Instruct-ForVideo

# For larger models with higher memory usage
python run.py \
    --data EgoExoBench_MCQ \
    --model Qwen2.5-VL-72B-Instruct-ForVideo
```

To skip evaluation on the **Ego-Exo4D** portion of the benchmark, specify the `EgoExoBench_64frame_skip_EgoExo4D` configuration with the **`--data`** argument.

```
# Example command to skip Ego-Exo4D
torchrun --nproc-per-node=1 run.py \
    --data EgoExoBench_64frame_skip_EgoExo4D \
    --model [Your_Model_Name]
```

> 💡 Note: If you encounter errors related to stacking videos with varying frame counts, try using `transformers==4.49.0` as a temporary workaround.

## 🙏 Acknowledgements

EgoExoBench builds upon publicly available ego–exo datasets: [Ego-Exo4D](https://ego-exo4d-data.org/), [LEMMA](https://sites.google.com/view/lemma-activity), [EgoExoLearn](https://huggingface.co/datasets/hyf015/EgoExoLearn), [TF2023](https://github.com/ziweizhao1993/PEN), [EgoMe](https://huggingface.co/datasets/HeqianQiu/EgoMe), [CVMHAT](https://github.com/RuizeHan/CVMHT). Thanks for open-sourcing!
