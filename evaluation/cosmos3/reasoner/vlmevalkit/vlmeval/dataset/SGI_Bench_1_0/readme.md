## 整体说明
SGI-Bench-1.0 包含5个子数据集(deep research , dry experiment , wet experiment , experimental reasoning, idea generation)

## 注意事项
1. dry experiment , experimental reasoning和deep research以及idea generation使用了模型进行评估，需要设置`OPENAI_API_KEY`,以及`OPENAI_API_BASE`环境变量
2. dry experiment评测过程中需要下载文件，默认路径是`./outputs`，可以通过`--judge-args`命令行参数传入`work_dir`参数进行控制。<br> 评测之前还需要运行以下命令
3. idea generation 的评测需要额外安装`sentence_transformers`包
```
conda create -n dryexp python=3.10.18
conda activate dryexp
pip install -r vlmeval/dataset/SGI_Bench_1_0/dry_experiment_requirements.txt
```
