<div align="center">

![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)

<b>VLMEvalKit: 大規模視覚言語モデルの評価ツールキット</b>

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

[English](/README.md) | [简体中文](/docs/zh-CN/README_zh-CN.md) | 日本語

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OpenCompass Learderboard </a> •
<a href="#-datasets-models-and-evaluation-results">📊Datasets & Models </a> •
<a href="#%EF%B8%8F-quickstart">🏗️Quickstart </a> •
<a href="#%EF%B8%8F-development-guide">🛠️Development </a> •
<a href="#-the-goal-of-vlmevalkit">🎯Goal </a> •
<a href="#%EF%B8%8F-citation">🖊️Citation </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HF Leaderboard</a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 Evaluation Records</a> •
<a href="https://discord.gg/evDT4GZmxN">🔊 Discord Channel</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 Technical Report</a>
</div>

**VLMEvalKit**（pythonパッケージ名は**vlmeval**）は、**大規模視覚言語モデル（LVLMs）**の**オープンソース評価ツールキット**です。このツールキットは、複数のリポジトリでのデータ準備という重労働なしに、さまざまなベンチマークでLVLMsの**ワンコマンド評価**を可能にします。VLMEvalKitでは、すべてのLVLMsに対して**生成ベースの評価**を採用し、**正確なマッチング**と**LLMベースの回答抽出**の両方で得られた評価結果を提供します。

PS: 日本語の README には最新のアップデートがすべて含まれていない場合があります。英語版をご確認ください。

## 📊 データセット、モデル、および評価結果

**公式のマルチモーダルリーダーボードでのパフォーマンス数値は、ここからダウンロードできます！**

[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [すべての詳細な結果をダウンロード](http://opencompass.openxlab.space/assets/OpenVLM.json)。

**Supported Benchmarks** in [**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb) を確認して、すべてのサポートされているベンチマーク（70以上）を表示してください。

**Supported LMMs** in [**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb) を確認して、すべてのサポートされている LMMs（200以上）を表示してください。

**Transformersバージョンの推奨事項:**

特定のtransformerバージョンで一部のVLMが実行できない可能性があることに注意してください。各VLMを評価するために、以下の設定を推奨します：

- **`transformers==4.33.0`を使用してください**: `Qwenシリーズ`, `Monkeyシリーズ`, `InternLM-XComposerシリーズ`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICSシリーズ`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4シリーズ`, `InstructBLIPシリーズ`, `PandaGPT`, `VXVERSE`, `GLM-4v-9B`.
- **`transformers==4.37.0`を使用してください**: `LLaVAシリーズ`, `ShareGPT4Vシリーズ`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLMシリーズ`, `EMU2シリーズ`, `Yi-VLシリーズ`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VLシリーズ`, `InternVLシリーズ`, `Cambrianシリーズ`, `VILA-VLシリーズ`.
- **`transformers==4.40.0`を使用してください**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **`transformers==4.42.0`を使用してください**: `AKI`.
- **`transformers==latest`を使用してください**: `LLaVA-Nextシリーズ`, `PaliGemma-3B`, `Chameleon-VLシリーズ`, `Video-LLaVA-7B-HF`, `Ovis1.5シリーズ`, `Mantisシリーズ`, `MiniCPM-V2.6`.

```python
# デモ
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# 単一画像のフォワード
ret = model.generate(['assets/apple.jpg', 'この画像には何がありますか？'])
print(ret)  # この画像には葉がついた赤いリンゴがあります。
# 複数画像のフォワード
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', '提供された画像にはリンゴが何個ありますか？'])
print(ret)  # 提供された画像にはリンゴが2個あります。
```

## 🏗️ クイックスタート

クイックスタートガイドについては、[クイックスタート](/docs/en/Quickstart.md)を参照してください。

## 🛠️ 開発ガイド

カスタムベンチマーク、VLMsを開発するか、単に**VLMEvalKit**に他のコードを貢献する場合は、[開発ガイド](/docs/en/Development.md)を参照してください。

コミュニティからの共有を奨励し、それに応じたクレジットを共有するために、次回のレポート更新では以下のことを実施します：

- 全ての貢献に対して感謝の意を示します
- 新しいモデル、評価セット、または主要な機能への3つ以上の主要な貢献を持つ貢献者は、テクニカルレポートの著者リストに加わることができます。適格な貢献者は、issueを作成するか、または[VLM評価キット ディスコードチャンネル](https://discord.com/invite/evDT4GZmxN)で kennyutc にDMを送ることができます。私たちはそれに応じてフォローアップします。

## 🎯 VLMEvalKitの目標

**このコードベースは以下を目的として設計されています：**

1. 研究者や開発者が既存のLVLMsを評価し、評価結果を**簡単に再現できるようにする**ための**使いやすい**、**オープンソースの評価ツールキット**を提供します。
2. VLMの開発者が自分のモデルを簡単に評価できるようにします。複数のサポートされているベンチマークでVLMを評価するには、単一の`generate_inner()`関数を**実装するだけで**、他のすべてのワークロード（データのダウンロード、データの前処理、予測の推論、メトリックの計算）はコードベースによって処理されます。

**このコードベースは以下を目的として設計されていません：**

1. すべての**第三者ベンチマーク**の元の論文で報告された正確な精度数値を再現すること。その理由は2つあります：
   1. VLMEvalKitは、すべてのVLMに対して**生成ベースの評価**を使用します（オプションで**LLMベースの回答抽出**を使用）。一方、一部のベンチマークは異なるアプローチを使用する場合があります（SEEDBenchはPPLベースの評価を使用します）。これらのベンチマークについては、対応する結果で両方のスコアを比較します。開発者には、コードベースで他の評価パラダイムをサポートすることをお勧めします。
   2. デフォルトでは、すべてのVLMに対して同じプロンプトテンプレートを使用してベンチマークを評価します。一方、**一部のVLMには特定のプロンプトテンプレートがある**場合があります（現時点ではコードベースでカバーされていない場合があります）。VLMの開発者には、現在カバーされていない場合でも、VLMEvalKitで独自のプロンプトテンプレートを実装することをお勧めします。これにより、再現性が向上します。

## 🖊️ 引用

この作業が役立つ場合は、このリポジトリに**スター🌟**を付けてください。サポートありがとうございます！

[![Stargazers repo roster for @open-compass/VLMEvalKit](https://reporoster.com/stars/open-compass/VLMEvalKit)](https://github.com/open-compass/VLMEvalKit/stargazers)

研究でVLMEvalKitを使用する場合、または公開されたオープンソースの評価結果を参照する場合は、以下のBibTeXエントリと、使用した特定のVLM/ベンチマークに対応するBibTexエントリを使用してください。

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

<p align="right"><a href="#top">🔝Top に戻る</a></p>

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
