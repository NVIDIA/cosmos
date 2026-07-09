"""Convert SPAR-Bench from HuggingFace to TSV format."""
import argparse
import os
import base64
from io import BytesIO
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def convert_spar_bench_to_tsv(dataset_name='jasonzhango/SPAR-Bench', output_dir='./data'):
    print(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name)
    test_data = dataset['test']
    
    tsv_data = []
    for idx, example in enumerate(tqdm(test_data)):
        images = example['image']
        if isinstance(images, list):
            image_b64 = encode_image_to_base64(images[0])
        else:
            image_b64 = encode_image_to_base64(images)
        
        tsv_data.append({
            'index': idx,
            'image': image_b64,
            'question': example['question'],
            'answer': example['answer'],
            'task': example.get('task', 'unknown'),
            'img_type': example.get('img_type', 'single_view'),
            'format_type': example.get('format_type', 'unknown'),
            'source': example.get('source', 'unknown'),
        })
    
    df = pd.DataFrame(tsv_data)
    os.makedirs(output_dir, exist_ok=True)
    output_name = 'SPAR-Bench-Tiny.tsv' if 'Tiny' in dataset_name else 'SPAR-Bench.tsv'
    output_path = os.path.join(output_dir, output_name)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"✅ Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='jasonzhango/SPAR-Bench')
    parser.add_argument('--output_dir', default='./data')
    args = parser.parse_args()
    convert_spar_bench_to_tsv(args.dataset, args.output_dir)
