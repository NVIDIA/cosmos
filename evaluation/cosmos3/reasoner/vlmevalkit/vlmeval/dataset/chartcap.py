import warnings

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from vlmeval.smp import dump, load
from vlmeval.smp.vlm import encode_image_to_base64
from .image_base import ImageBaseDataset


def prepare_chartcap():
    print("Loading ChartCap dataset from HuggingFace...")
    # Load the dataset
    ds = load_dataset("junyoung-00/ChartCap", split="test")

    # We will collect the data rows here
    data_rows = []

    # Default question as per requirements
    default_question = 'Please provide a detailed caption for the chart.'

    print("Processing samples...")
    for idx, sample in enumerate(tqdm(ds)):
        # Extract fields
        # sample in HuggingFace dataset usually acts like a dict

        # Prepare the base64 image
        if 'image' in sample:
            img = sample['image']
            img_b64 = encode_image_to_base64(img)
        else:
            print(f"Warning: No image found for sample {idx}, skipping.")
            continue

        # Prepare answer (ground truth caption)
        # The dataset likely has a caption field. Let's inspect the keys if we can't be sure
        # But for now I'll assume standard naming or check `sample` content dynamically if needed.
        # Based on typical HF datasets, it might be 'caption' or 'text'.
        # However, looking at junyoung-00/ChartCap on HF (hypothetically provided link),
        # usually it has 'image' and 'caption'.
        # If 'label' or 'ground_truth' exists, use that.
        # I will dump all original keys as requested.

        # Checking common keys for caption in such datasets
        answer = sample.get('caption', sample.get('text', ''))

        row = {
            'index': idx,
            'image': img_b64,
            'question': default_question,
            'answer': answer
        }

        # Add all original data information
        for k, v in sample.items():
            if k not in row and k != 'image':  # Don't duplicate image or overwrite our fields if they match
                row[k] = v

        data_rows.append(row)

    print(f"Creating TSV with {len(data_rows)} samples...")
    df = pd.DataFrame(data_rows)

    # Ensure columns order: index, image, question, answer, ... others
    cols = ['index', 'image', 'question', 'answer']
    remaining_cols = [c for c in df.columns if c not in cols]
    df = df[cols + remaining_cols]

    output_file = 'ChartCap.tsv'
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved to {output_file}")


class ChartCapDataset(ImageBaseDataset):

    TYPE = 'Caption'
    DATASET_URL = {
        'ChartCap': 'https://huggingface.co/datasets/alfassy/chart_cap_vlmevalkit/blob/main/ChartCap.tsv',
    }

    DATASET_MD5 = {
        'ChartCap': '10a0292079120f748eae81af5a1e19da',
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def evaluate(self, eval_file, **judge_kwargs):
        import evaluate
        from bert_score import BERTScorer
        from sentence_transformers import SentenceTransformer  # noqa: F401

        from .utils.megabench.scoring.sacrebleu_bleu import Bleu

        data = load(eval_file)

        # Prepare predictions and references
        # Ensure data is sorted or aligned by index if needed, but usually index is preserved
        # data usually contains 'prediction' and 'answer' columns

        predictions = [str(x) for x in data['prediction']]
        references = [str(x) for x in data['answer']]

        # Metric 1: sacreBLEU
        # Reference implementation uses corpus_bleu
        # vlmeval wrapper: Bleu.match(pred, gt) returns single score or expected aggregated?
        # Looking at Bleu.match source: it calls corpus_bleu(corr, [resp]).score / 100
        # It seems to be designed for single sample or list of list?
        # Let's use sacrebleu directly for corpus level or the wrapper if it supports batch properly.
        # wrapper `Bleu.match(response, correct_answer)` handles lists.
        # But it returns scalar 0-1 (divided by 100).

        bleu_score = Bleu.match(predictions, references) * 100

        # Metric 2: ROUGE-L through 'evaluate'
        try:
            rouge = evaluate.load("rouge")
            rouge_results = rouge.compute(predictions=predictions, references=references)
            rouge_l = rouge_results['rougeL']
        except Exception as e:
            warnings.warn(f"Failed to compute ROUGE: {e}")
            rouge_l = 0.0

        # Metric 3: METEOR through 'evaluate'
        try:
            meteor = evaluate.load("meteor")
            meteor_results = meteor.compute(predictions=predictions, references=references)
            meteor_score = meteor_results['meteor']
        except Exception as e:
            warnings.warn(f"Failed to compute METEOR: {e}")
            meteor_score = 0.0

        # Metric 4: BERTScore
        # Refer to uni_svg.py usage
        try:
            # Device handling
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Using roberta-large-mnli or similar is common, but let's stick to default or what uni_svg uses
            # uni_svg uses "en" (which defaults to roberta-large presumably)
            bert_scorer = BERTScorer(lang="en", rescale_with_baseline=False, device=device)
            P, R, F1 = bert_scorer.score(predictions, references)
            bert_score_val = F1.mean().item()
        except Exception as e:
            warnings.warn(f"Failed to compute BERTScore: {e}")
            bert_score_val = 0.0

        results = {
            'BLEU_4': bleu_score,
            'ROUGE_L': rouge_l,
            'METEOR': meteor_score,
            'BERTScore': bert_score_val
        }

        # Format as requested: dictionary composed of lists, organized into a pandas.DataFrame
        # or just a dictionary of scores?
        # Base class says: "The return value of the function is the calculated accuracy and other metrics,
        # formatted as a dictionary composed of lists, organized into a pandas.DataFrame."
        # However, for captioning, we usually return a single scalar per metric for the whole dataset.
        # But typical evaluate returns a dict (often converted to DF later).
        # Let's make it a DF with one row or just the dict if allowed.
        # Looking at ImageCaptionDataset, it returns a dict of scores.
        # Let's return the dict, usually frame_eval handles it.

        print("Evaluation Results for ChartCap:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

        # Create CSV output file
        import pandas as pd
        result_df = pd.DataFrame([results])
        result_file = eval_file.replace(f".{eval_file.split('.')[-1]}", "_acc.csv")
        dump(result_df, result_file)

        return results


if __name__ == '__main__':
    prepare_chartcap()
