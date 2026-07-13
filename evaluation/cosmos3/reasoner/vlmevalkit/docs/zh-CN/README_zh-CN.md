<div align="center">

![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)

<b>VLMEvalKit: 一种多模态大模型评测工具 </b>

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

[English](/README.md) | 简体中文 | [日本語](/docs/ja/README_ja.md)

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OpenCompass 排行榜 </a> •
<a href="#%EF%B8%8F-quickstart">🏗️ 快速开始 </a> •
<a href="#-datasets-models-and-evaluation-results">📊 数据集和模型 </a> •
<a href="#%EF%B8%8F-development-guide">🛠️ 开发指南 </a> •
<a href="#-the-goal-of-vlmevalkit">🎯 我们的目标 </a> •
<a href="#%EF%B8%8F-citation">🖊️ 引用 </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HuggingFace 排行榜 (存档全部性能) </a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 原始评测记录</a> •
<a href="https://discord.gg/evDT4GZmxN">🔊 Discord</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 技术报告 </a>
</div>

**VLMEvalKit** (python 包名为 **vlmeval**) 是一款专为大型视觉语言模型 (Large Vision-Language Models， LVLMs) 评测而设计的开源工具包。该工具支持在各种基准测试上对大型视觉语言模型进行**一键评估**，无需进行繁重的数据准备工作，让评估过程更加简便。在 VLMEvalKit 中，我们对所有大型视觉语言模型生成的结果进行评测，并提供基于**精确匹配**与基于 **LLM 的答案提取**两种评测结果。

## 🆕 更新

- **[2025-04-29]** 优化 `torchrun` 启动逻辑：目前 `torchrun` 启动时，若进程数为 M，机器 GPU 卡数为 N，将会自动调整每个进程分配的 GPU 数量为 `N // M`。目前此分配方式适用于 `transformers`, `lmdeploy` 推理后端，`vllm` 推理后端仅支持使用 python 启动 🔥🔥🔥
- **[2025-02-20]** 支持新模型：**InternVL2.5 series, QwenVL2.5 series, QVQ-72B, Doubao-VL, Janus-Pro-7B, MiniCPM-o-2.6, InternVL2-MPO, LLaVA-CoT, Hunyuan-Standard-Vision, Ovis2, Valley, SAIL-VL, Ross, Long-VITA, EMU3, SmolVLM**。支持新基准：**MMMU-Pro, WeMath, 3DSRBench, LogicVista, VL-RewardBench, CC-OCR, CG-Bench, CMMMU, WorldSense**。请参考[**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)以获取更多信息。感谢社区的各位贡献者 🔥🔥🔥
- **[2024-11-21]** 集成了一个新的配置系统，以实现更灵活的评估设置。查看[文档](/docs/zh-CN/ConfigSystem.md)或运行`python run.py --help`了解更多详情 🔥🔥🔥
- **[2024-11-21]** 支持 **[QSpatial](https://andrewliao11.github.io/spatial_prompt/)**，一个用于定量空间推理的多模态基准（例如，确定大小/距离），感谢 **[andrewliao11](https://github.com/andrewliao11)** 提供官方支持 🔥🔥🔥
- **[2024-11-21]** 支持 **[MM-Math](https://github.com/kge-sun/mm-math)**，一个包含约6K初中多模态推理数学问题的新多模态数学基准。GPT-4o-20240806在该基准上达到了22.5%的准确率 🔥🔥🔥
- **[2024-11-16]** 支持 **[OlympiadBench](https://github.com/OpenBMB/OlympiadBench)**，一个多模态基准，包含奥林匹克级别的数学和物理问题 🔥🔥🔥
- **[2024-11-16]** 支持 **[WildVision](https://huggingface.co/datasets/WildVision/wildvision-bench)**，一个基于多模态竞技场数据的主观多模态基准 🔥🔥🔥
- **[2024-11-13]** 支持 **[MIA-Bench](https://arxiv.org/abs/2407.01509)**，一个多模态指令跟随基准 🔥🔥🔥
- **[2024-11-08]** 支持 **[Aria](https://arxiv.org/abs/2410.05993)**，一个多模态原生 MoE 模型，感谢 **[teowu](https://github.com/teowu)** 🔥🔥🔥
- **[2024-11-04]** 支持 **[WorldMedQA-V](https://www.arxiv.org/abs/2410.12722)**，该基准包含 1000 多个医学 VQA 问题，涵盖巴西、以色列、日本、西班牙等四个国家的语言，以及它们的英文翻译 🔥🔥🔥

## 🏗️ 快速开始 <a id="quickstart"></a>

请参阅[**快速开始**](/docs/zh-CN/Quickstart.md)获取入门指南。

## 📊 评测结果，支持的数据集和模型 <a id="data-model-results"></a>

### 评测结果

**[OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)**: **[下载全部细粒度测试结果](http://opencompass.openxlab.space/assets/OpenVLM.json)**.

请查看[**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)中的 **Supported Benchmarks** 标签，以查看所有支持的图像和视频基准（70+）。

请查看[**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)中的 **Supported LMMs** 标签，以查看所有支持的 LMMs，包括商业 API、开源模型等（200+）。

### 其他

**Transformers 的版本推荐:**

**请注意**，某些 VLM 可能无法在某些特定的 transformers 版本下运行，我们建议使用以下设置来评估对应的VLM:

- **请用** `transformers==4.33.0` **来运行**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`, `VXVERSE`.
- **请用** `transformers==4.37.0 ` **来运行**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`, `Cambrian Series`, `VILA Series`, `Llama-3-MixSenseV1_1`, `Parrot-7B`, `PLLaVA Series`.
- **请用** `transformers==4.40.0 ` **来运行**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **请用** `transformers==4.42.0 ` **来运行**: `AKI`.
- **请用** `transformers==latest` **来运行**: `LLaVA-Next series`, `PaliGemma-3B`, `Chameleon series`, `Video-LLaVA-7B-HF`, `Ovis series`, `Mantis series`, `MiniCPM-V2.6`, `OmChat-v2.0-13B-sinlge-beta`, `Idefics-3`, `GLM-4v-9B`, `VideoChat2-HD`.

**如何测试一个 VLM 是否可以正常运行:**

```python
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# 前向单张图片
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # 这张图片上有一个带叶子的红苹果
# 前向多张图片
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # 提供的图片中有两个苹果
```

## 🛠️ 开发指南 <a id="development"></a>

要开发自定义评测数据集，支持其他 VLMs，或为 VLMEvalKit 贡献代码，请参阅[**开发指南**](/docs/zh-CN/Development_zh-CN.md)。

为激励来自社区的共享并分享相应的 credit，在下一次 report 更新中，我们将：

- 致谢所有的 contribution
- 具备三个或以上主要贡献 (支持新模型、评测集、或是主要特性) 的贡献者将可以加入技术报告的作者列表 。合条件的贡献者可以创建 issue 或是在 [VLMEvalKit Discord Channel](https://discord.com/invite/evDT4GZmxN) 私信 kennyutc，我们将进行跟进

## 🎯 VLMEvalKit 的目标 <a id="goal-of-vlmevalkit"></a>

**该代码库的设计目标是：**

1. 提供一个**易于使用**的**开源评估工具包**，方便研究人员和开发人员评测现有的多模态大模型，并使评测结果**易于复现**。
2. 使 VLM 开发人员能够轻松地评测自己的模型。在多个支持的基准测试上评估 VLM，只需实现一个 `generate_inner()` 函数，所有其他工作负载（数据下载、数据预处理、预测推理、度量计算）都由代码库处理。

**该代码库的设计目标不是:**

复现所有**第三方基准测试**原始论文中报告的准确数字。有两个相关的原因:
1. VLMEvalKit 对所有 VLMs 使用基于生成的评估（可选使用基于 LLM 的答案提取）。同时，一些基准测试可能官方使用不同的方法（*例如，SEEDBench 使用基于 PPL 的评估*）。对于这些基准测试，我们在相应的结果中比较两个得分。我们鼓励开发人员在代码库中支持其他评估范式。
2. 默认情况下，我们对所有多模态模型使用相同的提示模板来评估基准测试。同时，**一些多模态模型可能有他们特定的提示模板**（目前可能未在代码库中涵盖）。我们鼓励 VLM 的开发人员在 VLMEvalKit 中实现自己的提示模板，如果目前未覆盖。这将有助于提高可复现性。

## 🖊️ 引用 <a id="citation"></a>

如果我们的工作对您有所帮助，请考虑 **star🌟** VLMEvalKit。感谢支持！

[![Stargazers repo roster for @open-compass/VLMEvalKit](https://reporoster.com/stars/open-compass/VLMEvalKit)](https://github.com/open-compass/VLMEvalKit/stargazers)

如果您在研究中使用了 VLMEvalKit，或希望参考已发布的开源评估结果，请使用以下 BibTeX 条目以及与您使用的特定 VLM / 基准测试相对应的 BibTex 条目。

```bib
@misc{duan2024vlmevalkit,
      title={VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models},
      author={Haodong Duan and Junming Yang and Yuxuan Qiao and Xinyu Fang and Lin Chen and Yuan Liu and Xiaoyi Dong and Yuhang Zang and Pan Zhang and Jiaqi Wang and Dahua Lin and Kai Chen},
      year={2024},
      eprint={2407.11691},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11691},
}
```

<p align="right"><a href="#top">🔝回到顶部</a></p>

[github-contributors-link]: https://github.com/open-compass/VLMEvalKit/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/VLMEvalKit?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/VLMEvalKit/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/VLMEvalKit?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/VLMEvalKit/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/VLMEvalKit?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/VLMEvalKit/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/VLMEvalKit?color=white&labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/VLMEvalKit/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/VLMEvalKit?color=ffcb47&labelColor=black&style=flat-square
