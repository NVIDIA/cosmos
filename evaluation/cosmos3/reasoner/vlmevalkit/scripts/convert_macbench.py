import argparse
import string
import ast
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset
import pandas as pd
# from vlmeval.smp.vlm import encode_image_to_base64
from concurrent.futures import ProcessPoolExecutor

prog_description = """\
Convert original MaCBench dataset to TSV.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('out', type=Path, help='output path')
    args = parser.parse_args()
    return args


CONFIGS = [
    'afm-image', 'chem-lab-basic', 'chem-lab-comparison',
    'chem-lab-equipments', 'chirality', 'cif-atomic-species',
    'cif-crystal-system', 'cif-density', 'cif-symmetry', 'cif-volume',
    'electronic-structure', 'handdrawn-molecules', 'isomers',
    'mof-adsorption-strength-comparison', 'mof-adsorption-strength-order',
    'mof-capacity-comparison', 'mof-capacity-order', 'mof-capacity-value',
    'mof-henry-constant-comparison', 'mof-henry-constant-order',
    'mof-working-capacity-comparison', 'mof-working-capacity-order',
    'mof-working-capacity-value', 'org-schema', 'org-schema-wo-smiles',
    'organic-molecules', 'spectral-analysis', 'tables-qa', 'us-patent-figures',
    'us-patent-plots', 'xrd-pattern-matching', 'xrd-pattern-shape',
    'xrd-peak-position', 'xrd-relative-intensity'
]


def process_row(row):
    example = ast.literal_eval(row['examples'][0])

    image = example['qentries_modality']['image']['entry1']['value'].partition(',')[-1]
    assert len(image) > 1000
    output = {'image': image}

    entries = example['qentries_modality']['image']
    question = example['input']
    for k, entry in entries.items():
        if entry['type'] == 'text':
            question = question.replace('{' + k + '}', entry['value'])
        else:
            question = question.replace('{' + k + '}', '{image}')
        output['question'] = question

    if 'target' in example:
        answer = example['target']
        output['answer'] = str(answer)
    elif 'target_scores' in example:
        answer = []
        for i, (option, grade) in enumerate(example['target_scores'].items()):
            column = string.ascii_uppercase[i]
            output[column] = option
            if grade:
                answer.append(column)
        output['answer'] = ','.join(answer)

    if row['relative_tolerance'] is not None:
        output['relative_tolerance'] = row['relative_tolerance']
    return output


def process_single_config(name):
    ds: Dataset = load_dataset('jablonkagroup/MaCBench', name)['train']
    processed_rows = []
    for row in ds:
        processed_rows.append(process_row(row))
    df = pd.DataFrame(processed_rows)
    df['category'] = str(name)
    return df


def main():
    args = parse_args()
    all_dfs = []
    with ProcessPoolExecutor() as executor:
        tasks = []
        for name in CONFIGS:
            tasks.append(executor.submit(process_single_config, name))
        for task in tqdm(tasks):
            sub_df = task.result()
            all_dfs.append(sub_df)

    df = pd.concat(all_dfs, ignore_index=True).reset_index()
    df.to_csv(args.out, sep='\t', index=True)


if __name__ == "__main__":
    main()
