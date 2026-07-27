import json
import os
import re
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from timeout_decorator import timeout

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, get_logger, load
from vlmeval.utils.mp_util import track_progress_rich

logger = get_logger(__name__)
# ============================================================================
# CONFIGURATION
# ============================================================================
# Model Paths
CLIP_MODEL_PATH = "openai/clip-vit-large-patch14-336"
SBERT_MODEL_PATH = "all-MiniLM-L6-v2"
lpips_model = None
clip_model = None
clip_processor = None
sbert_model = None
bert_scorer = None
# ============================================================================


def _init_models():
    """Lazy load models."""
    global lpips_model, clip_model, clip_processor
    global sbert_model, bert_scorer, device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading LPIPS...")
    import lpips
    model_path = os.path.join(LMUDataRoot(), 'aux_models', 'alex.pth')
    lpips_model = lpips.LPIPS(net='alex', model_path=model_path).to(device)

    print("Loading CLIP...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)

    print("Loading Sentence-BERT...")
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer(SBERT_MODEL_PATH).to(device)

    print("Loading BERTScore...")
    from bert_score import BERTScorer
    bert_scorer = BERTScorer(
        lang="en", rescale_with_baseline=True, device=device
    )

    return device


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_numbers(text):
    return [float(num) for num in re.findall(r'\d+\.\d+|\d+', str(text))]


def parse_json_safe(val):
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except (ValueError, TypeError):
        return {}


def fix_svg(svg_content):
    svg_content = re.sub(r'<[^>]*$', '', svg_content)
    open_tags = re.findall(r'<([a-zA-Z]+)[^>]*>', svg_content)
    close_tags = re.findall(r'</([a-zA-Z]+)>', svg_content)

    open_tag_stack = []
    ignored_tags = [
        'path', 'rect', 'circle', 'ellipse', 'line',
        'polyline', 'polygon', 'image', 'stop'
    ]

    for tag in open_tags:
        if tag in ignored_tags:
            continue
        if tag not in close_tags:
            open_tag_stack.append(tag)
        else:
            close_tags.remove(tag)

    while open_tag_stack:
        tag = open_tag_stack.pop()
        svg_content += f'</{tag}>'

    if not svg_content.strip().endswith('</svg>'):
        svg_content += '</svg>'
    return svg_content


def extract_svg_from_text(text):
    text = str(text)
    svg_match = re.search(r'<svg[^>]*>(.*?)</svg>', text, re.DOTALL)
    if svg_match:
        return svg_match.group(0)
    svg_part = re.search(r'<svg[^>]*>.*', text, re.DOTALL)
    if svg_part:
        return fix_svg(svg_part.group(0))
    return None


def save_svg_and_convert_to_png(svg_content, output_base_path):
    """
    Saves SVG and converts to PNG.
    Returns the PNG path if successful, None otherwise.
    """
    import cairosvg
    svg_path = output_base_path + '.svg'
    png_path = output_base_path + '.png'

    if not svg_content:
        return None

    try:
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        # Standard UniSVG resolution 336x336
        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            output_width=336,
            output_height=336,
            background_color='white'
        )
    except Exception:
        # print(f"Error converting SVG: {e}")
        return None
    return png_path


def evaluate_images(original_image_path, generated_image_path, device):
    import lpips
    from skimage.metrics import structural_similarity as ssim

    original_image = Image.open(original_image_path).convert('RGB')
    generated_image = Image.open(generated_image_path).convert('RGB')

    if original_image.size != generated_image.size:
        generated_image = generated_image.resize(original_image.size)

    original_image_np = np.array(original_image)
    generated_image_np = np.array(generated_image)

    min_dim = min(original_image_np.shape[0], original_image_np.shape[1])
    win_size = min(7, min_dim // 2 * 2 + 1)

    ssim_value = ssim(
        original_image_np,
        generated_image_np,
        channel_axis=2,
        win_size=win_size
    )

    mse_value = np.mean((original_image_np - generated_image_np) ** 2)
    if mse_value == 0:
        psnr_value = float('inf')
    else:
        psnr_value = 10 * np.log10((255 ** 2) / mse_value)

    original_image_tensor = lpips.im2tensor(original_image_np).to(device)
    generated_image_tensor = lpips.im2tensor(generated_image_np).to(device)
    with torch.no_grad():
        lpips_value = lpips_model(
            original_image_tensor, generated_image_tensor
        ).item()

    inputs = clip_processor(
        text=["a photo"],
        images=[original_image, generated_image],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        clip_similarity = torch.cosine_similarity(
            image_features[0], image_features[1], dim=0
        ).item()

    return ssim_value, psnr_value, lpips_value, clip_similarity


basic_shapes = [
    "rect", "circle", "ellipse", "line",
    "polygon", "polyline", "path", "text"
]


def parse_shapes(text):
    shape_counts = {shape: 0 for shape in basic_shapes}
    pattern = re.compile(r"(\d+)\s+(" + "|".join(basic_shapes) + r")s?")
    matches = pattern.findall(str(text))
    for count, shape in matches:
        shape_counts[shape] = int(count)
    return shape_counts


def calculate_shape_accuracy(gt_text, model_answer):
    gt_shapes = parse_shapes(gt_text)
    model_shapes = parse_shapes(model_answer)
    correct_shapes = 0
    total_shapes = sum(1 for shape in basic_shapes if gt_shapes[shape] > 0)
    for shape in basic_shapes:
        if gt_shapes[shape] > 0:
            if gt_shapes[shape] == model_shapes[shape]:
                correct_shapes += 1
    if total_shapes == 0:
        return 1.0 if sum(model_shapes.values()) == 0 else 0.0
    return correct_shapes / total_shapes


def extract_dimensions(text):
    numbers = extract_numbers(text)
    if len(numbers) >= 2:
        return numbers[:2]
    return None, None


def extract_transformations(text):
    transformations = [
        "translate", "rotate", "scale", "skewX", "skewY", "matrix"
    ]
    transform_counts = {transform: 0 for transform in transformations}
    for transform in transformations:
        transform_counts[transform] = len(re.findall(transform, str(text)))
    return transform_counts


def evaluate_row(idx, model_answer, gt_answer, q_type, metadata, tmpdir, device):
    try:
        return _evaluate_row(idx, model_answer, gt_answer, q_type, metadata, tmpdir, device)
    except Exception as e:
        logger.warning(f'Error sample {idx}: {repr(e)}')
        return idx, None


@timeout(60., use_signals=False)
def _evaluate_row(idx, model_answer, gt_answer, q_type, metadata, tmpdir, device):
    from sentence_transformers import util as sbert_util

    metrics = {
        'ssim': 0,
        'psnr': 0,
        'lpips': 0,
        'clip_score': 0,
        'bertscore': 0,
        'sbert_score': 0,
        'accuracy': 0
    }

    model_answer = str(model_answer)
    gt_answer = str(gt_answer)

    if q_type in ["CSVGUN_color", "ISVGUN_color"]:
        kw = metadata.get('keywords', {})
        colors_gt = kw.get('colors', {}).get('colors', [])
        if not colors_gt and 'colors' in metadata:
            colors_gt = metadata['colors'].get('colors', [])
        correct_colors = set(
            c.lower() for c in colors_gt if c.lower() != 'white'
        )
        model_colors = set(
            re.findall(r'\b\w+\b', model_answer.lower())
        )
        matched_colors = sum(
            1 for c in model_colors if c in correct_colors
        )
        if len(correct_colors) > 0:
            acc = matched_colors / len(correct_colors)
        else:
            acc = 0
        metrics['accuracy'] += acc

    # --- CATEGORY ---
    elif q_type in ["CSVGUN_category", "ISVGUN_category"]:
        cats_gt = metadata.get('keywords', {}).get('category', [])
        if not cats_gt and 'category' in metadata:
            cats_gt = metadata['category']
        correct_lower = set(c.lower() for c in cats_gt)
        correct_cap = set(c.capitalize() for c in cats_gt)
        matched = (
            any(c in model_answer for c in correct_lower)
            or any(c in model_answer for c in correct_cap)
        )
        metrics['accuracy'] += 1 if matched else 0

    # --- GENERATION (ISVGEN, TSVGEN) ---
    elif q_type in ["ISVGEN", "TSVGEN"]:
        created_files = []
        try:
            # 1. Render Model Prediction
            model_svg = extract_svg_from_text(model_answer)
            if not model_svg:
                model_svg = (
                    '<svg xmlns="http://www.w3.org/2000/svg" '
                    'width="336" height="336"></svg>'
                )

            gen_base_path = os.path.join(tmpdir, f"gen_{idx}")
            gen_img_path = save_svg_and_convert_to_png(
                model_svg, gen_base_path
            )

            if gen_img_path:
                created_files.append(gen_img_path)
                created_files.append(gen_base_path + '.svg')

            # 2. Render Ground Truth
            gt_svg = extract_svg_from_text(gt_answer)
            gt_img_path = None
            if gt_svg:
                gt_base_path = os.path.join(tmpdir, f"gt_{idx}")
                gt_img_path = save_svg_and_convert_to_png(
                    gt_svg, gt_base_path
                )
                if gt_img_path:
                    created_files.append(gt_img_path)
                    created_files.append(gt_base_path + '.svg')

            # 3. Evaluate
            valid_gen = gen_img_path and os.path.exists(gen_img_path)
            valid_gt = gt_img_path and os.path.exists(gt_img_path)

            if valid_gen and valid_gt:
                try:
                    ssim_v, psnr_v, lpips_v, clip_v = evaluate_images(
                        gt_img_path, gen_img_path, device
                    )
                except Exception as e:
                    print(f"Error evaluating images {idx}: {e}")
                    ssim_v, psnr_v, lpips_v, clip_v = 0, 0, 1.0, 0

                metrics['ssim'] += ssim_v
                val = psnr_v if psnr_v != float('inf') else 100
                metrics['psnr'] += val
                metrics['lpips'] += lpips_v
                metrics['clip_score'] += clip_v
            else:
                metrics['lpips'] += 1.0

        finally:
            for fpath in created_files:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass

    # --- TEXT ---
    elif q_type in [
        "CSVGUN_usage", "ISVGUN_usage", "CSVGUN_rect",
        "CSVGUN_circle", "CSVGUN_description", "ISVGUN_description"
    ]:
        if model_answer and gt_answer:
            P, R, F1 = bert_scorer.score(
                [model_answer], [gt_answer]
            )
            b_score = F1.mean().item()
            mod_emb = sbert_model.encode(
                model_answer, convert_to_tensor=True
            )
            gt_emb = sbert_model.encode(
                gt_answer, convert_to_tensor=True
            )
            s_score = sbert_util.pytorch_cos_sim(
                mod_emb, gt_emb
            ).item()
            metrics['bertscore'] += b_score
            metrics['sbert_score'] += s_score

    # --- SHAPE ---
    elif q_type == "CSVGUN_shape":
        acc = calculate_shape_accuracy(gt_answer, model_answer)
        metrics['accuracy'] += acc

    # --- SIZE ---
    elif q_type == "CSVGUN_size":
        mw, mh = extract_dimensions(model_answer)
        gw, gh = extract_dimensions(gt_answer)
        acc = 0
        if mw == gw and mh == gh:
            acc = 1
        elif mw == gw or mh == gh:
            acc = 0.5
        metrics['accuracy'] += acc

    # --- TRANSFORM ---
    elif q_type == "CSVGUN_transform":
        mt = extract_transformations(model_answer)
        gt = extract_transformations(gt_answer)
        correct_c = 0
        total_c = sum(gt.values())
        for k in gt:
            if k in mt:
                correct_c += min(mt[k], gt[k])
        if total_c > 0:
            acc = correct_c / total_c
        else:
            acc = 1.0 if correct_c == 0 else 0.0
        metrics['accuracy'] += acc

    return q_type, metrics


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_uni_svg(eval_file, **kwargs):
    device = _init_models()
    data = load(eval_file)

    # Create a base temporary directory for this evaluation run
    run_temp_dir = tempfile.mkdtemp(prefix="unisvg_eval_")
    print(f"Temporary workspace created at: {run_temp_dir}")

    type_metrics = {}
    type_samples = defaultdict(int)
    missing_samples = []

    tmp_file = get_intermediate_file_path(eval_file, '_EVAL', 'pkl')
    total_count = len(data)

    try:
        tups = []
        indices = []
        for idx, row in data.iterrows():
            model_answer = row['prediction']
            gt_answer = row['answer']
            q_type = row['category']
            metadata = parse_json_safe(row.get('l2-category', '{}'))

            if pd.isna(model_answer):
                missing_samples.append(idx)
                if q_type in ["ISVGEN", "TSVGEN"]:
                    type_metrics[q_type]['lpips'] += 1
                type_samples[q_type] += 1
                continue

            tups.append((idx, model_answer, gt_answer, q_type, metadata, run_temp_dir, device))
            indices.append(row['index'])

        ans = {}
        if os.path.exists(tmp_file):
            ans = load(tmp_file)
        tups = [x for x, i in zip(tups, indices) if i not in ans]
        indices = [i for i in indices if i not in ans]
        results = track_progress_rich(evaluate_row,
                                      tups,
                                      nproc=None,
                                      save=tmp_file,
                                      keys=indices,
                                      use_process=False)

        for result in results:
            if result[1] is None:
                missing_samples.append(result[0])
            q_type, metrics = result
            type_metrics.setdefault(
                q_type, {
                    'ssim': 0,
                    'psnr': 0,
                    'lpips': 0,
                    'clip_score': 0,
                    'bertscore': 0,
                    'sbert_score': 0,
                    'accuracy': 0
                })
            type_samples[q_type] += 1
            for k, v in metrics.items():
                type_metrics[q_type][k] += v
    finally:
        # Final cleanup
        if os.path.exists(run_temp_dir):
            try:
                shutil.rmtree(run_temp_dir)
                print(f"Cleaned up temporary workspace: {run_temp_dir}")
            except OSError as e:
                print(f"Error cleaning up temp dir: {e}")

    # ========================================================================
    # AGGREGATION
    # ========================================================================

    final_results = {
        'total_samples': total_count,
        'type_samples': type_samples,
        'type_metrics': {}
    }

    for t in type_metrics:
        count = type_samples[t]
        if count == 0:
            continue

        final_results['type_metrics'][t] = {}
        metrics = type_metrics[t]

        if 'accuracy' in metrics and metrics['accuracy'] > 0:
            final_results['type_metrics'][t]['accuracy'] = (
                metrics['accuracy'] / count
            )

        if 'ssim' in metrics:
            final_results['type_metrics'][t].update({
                'ssim': metrics['ssim'] / count,
                'psnr': metrics['psnr'] / count,
                'lpips': metrics['lpips'] / count,
                'clip_score': metrics['clip_score'] / count,
            })

        has_bert = metrics.get('bertscore', 0) != 0
        has_sbert = metrics.get('sbert_score', 0) != 0
        if 'bertscore' in metrics and (has_bert or has_sbert):
            final_results['type_metrics'][t].update({
                'bertscore': metrics['bertscore'] / count,
                'sbert_score': metrics['sbert_score'] / count,
            })

    def safe_get_metric(type_name, metric_name, default=0.0):
        m = final_results['type_metrics'].get(type_name, {})
        return float(m.get(metric_name, default))

    def calc_gen_score(type_name):
        if type_name not in final_results['type_metrics']:
            return 0.0
        ssim = safe_get_metric(type_name, 'ssim')
        lpips = safe_get_metric(type_name, 'lpips', default=1.0)
        clip = safe_get_metric(type_name, 'clip_score')
        return (ssim * 0.2) + ((1.0 - lpips) * 0.2) + (clip * 0.6)

    isvgen_score = calc_gen_score("ISVGEN")
    tsvgen_score = calc_gen_score("TSVGEN")

    easy_types = ["CSVGUN_size", "CSVGUN_shape", "CSVGUN_transform"]
    easy_acc_sum = 0
    easy_denom = 0
    for t in easy_types:
        if t in final_results['type_metrics']:
            easy_acc_sum += safe_get_metric(t, 'accuracy') * 50
            easy_denom += 50
    easy_acc = easy_acc_sum / easy_denom if easy_denom > 0 else 0.0

    hard_types = [
        "ISVGUN_color", "CSVGUN_color", "ISVGUN_category", "CSVGUN_category"
    ]
    hard_acc_sum = 0
    hard_denom = 0
    for t in hard_types:
        if t in final_results['type_metrics']:
            hard_acc_sum += safe_get_metric(t, 'accuracy') * 100
            hard_denom += 100
    hard_acc = hard_acc_sum / hard_denom if hard_denom > 0 else 0.0

    bert_sbert_weights = [
        ("CSVGUN_usage", 200), ("CSVGUN_description", 200),
        ("ISVGUN_usage", 200), ("ISVGUN_description", 200),
        ("CSVGUN_rect", 100), ("CSVGUN_circle", 100)
    ]

    total_bert = 0.0
    total_sbert = 0.0
    total_w = 0.0

    for t, w in bert_sbert_weights:
        if t in final_results['type_metrics']:
            total_bert += safe_get_metric(t, 'bertscore') * w
            total_sbert += safe_get_metric(t, 'sbert_score') * w
            total_w += w

    bertscore = total_bert / total_w if total_w > 0 else 0.0
    sbert_score = total_sbert / total_w if total_w > 0 else 0.0

    final_score = (
        isvgen_score * 0.45
        + tsvgen_score * 0.45
        + easy_acc * 0.01
        + hard_acc * 0.02
        + bertscore * 0.035
        + sbert_score * 0.035
    )

    final_results['isvgen_score'] = isvgen_score
    final_results['tsvgen_score'] = tsvgen_score
    final_results['easy_acc'] = easy_acc
    final_results['hard_acc'] = hard_acc
    final_results['bertscore'] = bertscore
    final_results['sbert_score'] = sbert_score
    final_results['final_score'] = final_score

    score_file = get_intermediate_file_path(eval_file, '_score', 'json')
    dump(final_results, score_file)

    print(f"\n{'=' * 50}")
    print("UniSVG Evaluation Results")
    print(f"{'=' * 50}")
    print(f"ISVGEN Score: {isvgen_score:.4f}")
    print(f"TSVGEN Score: {tsvgen_score:.4f}")
    print(f"Easy Accuracy: {easy_acc:.4f}")
    print(f"Hard Accuracy: {hard_acc:.4f}")
    print(f"BERTScore:     {bertscore:.4f}")
    print(f"SBERT Score:   {sbert_score:.4f}")
    print(f"Final Score:   {final_score:.4f}")
    print(f"{'=' * 50}\n")

    return final_results
