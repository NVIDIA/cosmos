# Cosmos3 vLLM Plugin

Start the vLLM server:

```shell
cd packages/vllm-cosmos3
VLLM_USE_DEEP_GEMM=0 uvx --torch-backend=auto --with-editable . vllm@latest serve nvidia/Cosmos3-Nano \
  --trust-remote-code \
  --port 8000
```

Wait for the server to start (takes ~5 minutes). You will see `Application startup complete.` in the log.

In a separate terminal, submit a request:

```shell
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Cosmos3-Nano",
    "messages": [{"role": "user", "content": "Give me a short introduction to large language model."}],
    "max_tokens": 4096
  }' | jq -r '.choices[0].message.content'
```

For more details, see:

- [Cosmos-Reason2 repository](https://github.com/nvidia-cosmos/cosmos-reason2)
- [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL#online-serving)
- [Qwen3-VL vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
