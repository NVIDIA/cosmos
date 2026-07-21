# MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks [ICLR 2025]

![image](https://github.com/user-attachments/assets/5fd44fa9-0ec2-4298-ad0c-e883cb1edf7f)

MEGA-Bench contains 505 multimodal tasks with diverse data sources, input/output formats, and skill requirements. The taxonomy tree is derived from the application dimension, which guides and calibrates the annotation process. The benchmark is equiped with a suite of 45 evaluation metrics to handle various output formats beyond multiple-choice questions.

Following this doc, the evaluation result contains the final scores and multi-dimensional breakdown, which has a consistent format as [MEGA-Bench Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MEGA-Bench). Below is an example from evaluating `Qwen-2-VL-7B-Instruct` on the core set.


## Step-1: Install requirements for MEGA-Bench metrics to obtain the evaluation scores and breakdown analysis

```bash
pip install -r vlmeval/dataset/utils/megabench/requirements.txt
```


## Step-2: Get the model response and evaluation score files with VLMEvalKit

```bash
# Core set (440 tasks, in 16-frame setting)
python3 run.py \
    --data MEGABench_core_16frame \
    --model Qwen2-VL-7B-Instruct \
    --work-dir your/work/dir \

# Open-ended set (65 tasks, in 16-frame setting)
python3 run.py \
    --data MEGABench_open_16frame \
    --model Qwen2-VL-7B-Instruct \
    --work-dir your/work/dir \
```
Note: please set up the `OPENAI_API_KEY` in the .env file to evaluate the open set.

Then you can have 2 score files in the directory like: 

```bash
your/work/dir/Qwen-2-VL-7B-Instruct/T20250706_Gbf63ab2c/megabench_score_core.json
your/work/dir/Qwen-2-VL-7B-Instruct/T20250707_Gbf63ab2c/megabench_score_open.json
```

## Step-3(Optional): Run MEGA-Bench scripts to obtain the breakdown analysis

Move the 2 score files into the same directory, then run the script:

```bash
# Run the metrics for the open-ended set
cd vlmeval/dataset/utils/megabench/tools
python3 derive_breakdown_results.py  --input_dir your/dir/to/megabench_scores
```

The results in `your/dir/to/megabench_scores/analysis` are what used by [MEGA-Bench leaderboard](https://huggingface.co/spaces/TIGER-Lab/MEGA-Bench). The leaderboard can be updated by putting the files in the results directory of the leadboard's [HuggingFace space](https://huggingface.co/spaces/TIGER-Lab/MEGA-Bench/tree/main/static/eval_results/Default).
