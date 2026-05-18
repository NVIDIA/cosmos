# Online Inference

> **Skill:** `.agents/skills/cosmos3-inference/SKILL.md`

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Setup](#setup)
- [Usage](#usage)

______________________________________________________________________

<!--TOC-->

**EXPERIMENTAL:** This guide is not tested. The code is provided for learning purposes only.

We recommend first reading [Inference](./inference.md).

Useful resources:

- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html).

## Setup

Install the serve dependencies:

```shell
uv pip install -e ".[serve]"
```

## Usage

Start the server using one of the following methods:

<details open><summary><b>Single model</b></summary>

```shell
python -m cosmos3.ray.serve \
    --parallelism-preset=latency \
    --keep-going \
    -o outputs/ray_serve \
    --checkpoint-path Cosmos3-Nano
```

To see all available arguments:

```shell
python -m cosmos3.ray.serve --help
```

</details>

<details><summary><b>Multiple models</b></summary>

```shell
serve run cosmos3/ray/configs/latency.yaml
```

</details>

To monitor, open the dashboard <http://localhost:8265/#/serve>.

Wait ~1 min until the server is ready. You can either monitor the dashboard or the log:

```
Application 'cosmos3_omni' is ready at http://127.0.0.1:8000/.
```

Models are not loaded until a request is submitted. In a separate terminal, submit requests using one of the following methods:

<details open><summary><b>Curl/Wget</b></summary>

```shell
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "", "name": "city", "prompt": "A bustling city street at night"}' \
  -N
```

</details>

<details open><summary><b>Submit CLI</b></summary>

```shell
python -m cosmos3.ray.submit \
    -i "inputs/omni/*.json" \
    -o outputs/omni_ray \
    --seed=0
```

To see all available arguments:

```shell
python -m cosmos3.ray.submit --help
```

</details>

<details open><summary><b>Gradio UI</b></summary>

Launch the gradio frontend at <http://localhost:8080>:

```shell
python -m cosmos3.ray.gradio --port=8080
```

To see all available arguments:

```shell
python -m cosmos3.ray.gradio --help
```

</details>
