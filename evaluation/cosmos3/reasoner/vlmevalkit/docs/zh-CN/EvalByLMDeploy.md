# 使用 LMDeploy 加速评测推理

VLMEvalKit 支持测试由 LMDeploy 部署的 VLM 模型，下面以 InternVL2-8B 为例，展示如何测试模型

## 第0步 安装 LMDeploy

```bash
pip install lmdeploy
```

其他安装方式可以参考 LMDeploy 的[文档](https://github.com/InternLM/lmdeploy)

## 第1步 启动推理服务

```bash
lmdeploy serve api_server OpenGVLab/InternVL2-8B --model-name InternVL2-8B
```
> [!IMPORTANT]
> 因为 VLMEvalKit 中的模型对于不同数据集在构建 prompt 时可能有自定义行为，如 InternVL2 对于 HallusionBench 的处理，所以，server 端在启动的时候需要指定 `--model-name`，这样在使用 LMDEploy api 时可以根据名字选择合适的 prompt 构建策略。
>
> 如果指定了 `--server-port`，需要设置对应的环境变量 `LMDEPLOY_API_BASE`


## 第2步 评测

```bash
python run.py --data MMStar --model InternVL2-8B --verbose --api-nproc 64
```
