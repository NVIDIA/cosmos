import re
from fractions import Fraction

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load


def _clean_extracted_number(number_str):
    if not number_str:
        return None
    cleaned = str(number_str).strip().lstrip('$').strip()
    cleaned = cleaned.replace(',', '')
    if re.match(r'^-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?$', cleaned):
        return cleaned
    return None


def _get_last_paragraph(text):
    paragraphs = re.split(r'\n\s*\n', text.strip())
    if paragraphs:
        return paragraphs[-1]
    return text[-500:] if len(text) > 500 else text


def extract_numerical_answer(text):
    if not isinstance(text, str):
        return None
    text = str(text).strip()

    final_answer_patterns = [
        r"(?:FINAL\s+ANSWER|Final\s+Answer)\s*[:\-=]?\s*\$?\s*(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
        r"(?:FINAL\s+ANSWER|Final\s+Answer)\s*[:\-=]?\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)",
        r"\\boxed\\{(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)\\}",  # noqa: W605
        r"\\boxed\\{(-?[0-9,]+(?:\.[0-9]+)?)\\}",  # noqa: W605
        r"####\s*(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
        r"####\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)",
        r"(?:The\s+)?(?:[Aa]nswer|result)\s+is\s*[:\-=]?\s*\$?\s*(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
        r"(?:The\s+)?(?:[Aa]nswer|result)\s+is\s*[:\-=]?\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)",
        r"答案\s*[:\-=]?\s*\$?\s*(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
        r"答案\s*[:\-=]?\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)",
        r"(?:Therefore|Thus|Hence)[^.]*?[:\-=]?\s*\$?\s*(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
        r"(?:Therefore|Thus|Hence)[^.]*?[:\-=]?\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)",
    ]

    for pattern in final_answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1]
            cleaned = _clean_extracted_number(answer)
            if cleaned is not None:
                try:
                    return _normalize_to_float(cleaned)
                except Exception:
                    pass

    lines = text.split('\n')
    for i in range(min(5, len(lines))):
        line = lines[-(i + 1)].strip()
        if not line:
            continue

        standalone_patterns = [
            r'^(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)$',
            r'^(-?[0-9,]+(?:\.[0-9]+)?)$',
            r'^\$\s*(-?[0-9,]+(?:\.[0-9]+)?)$',
            r'^(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)\s*(?:dollars?|cents?)?$',
        ]

        for pattern in standalone_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                cleaned = _clean_extracted_number(match.group(1))
                if cleaned is not None:
                    try:
                        return _normalize_to_float(cleaned)
                    except Exception:
                        pass

    last_paragraph = _get_last_paragraph(text)
    contextual_patterns = [
        r"(?:answer|result|solution)\s*(?:is|=|:)?\s*\$?\s*(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
        r"(?:answer|result|solution)\s*(?:is|=|:)?\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)",
        r"=\s*\$?\s*(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
        r"=\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)",
        r"makes?\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?)?",
        r"total\s*(?:of|is)?\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)",
        r"costs?\s*\$?\s*(-?[0-9,]+(?:\.[0-9]+)?)",
    ]

    for pattern in contextual_patterns:
        matches = re.findall(pattern, last_paragraph, re.IGNORECASE)
        if matches:
            answer = matches[-1]
            cleaned = _clean_extracted_number(answer)
            if cleaned is not None:
                try:
                    return _normalize_to_float(cleaned)
                except Exception:
                    pass

    for i in range(min(3, len(lines))):
        line = lines[-(i + 1)].strip()

        numbers = re.findall(r'-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?', line)
        if numbers:
            cleaned = _clean_extracted_number(numbers[-1])
            if cleaned is not None:
                try:
                    return _normalize_to_float(cleaned)
                except Exception:
                    pass

        comma_numbers = re.findall(r'-?[0-9,]+(?:\.[0-9]+)?', line)
        if comma_numbers:
            cleaned = _clean_extracted_number(comma_numbers[-1])
            if cleaned is not None:
                try:
                    return _normalize_to_float(cleaned)
                except Exception:
                    pass

    return None


def _normalize_to_float(number_str):
    if '/' in number_str:
        try:
            frac = Fraction(number_str)
            return float(frac)
        except (ValueError, ZeroDivisionError):
            pass

    return float(number_str)


def is_correct(prediction, ground_truth, tolerance=1e-3):
    if prediction is None or ground_truth is None:
        return False

    try:
        pred_num = float(prediction)
        gt_num = float(ground_truth)

        if abs(pred_num - gt_num) <= tolerance:
            return True

        rel_diff = abs(pred_num - gt_num) / max(abs(gt_num), 1e-10)
        if rel_diff < 1e-6:
            return True

        return False
    except Exception:
        try:
            pred_str = str(prediction).replace(',', '')
            gt_str = str(ground_truth).replace(',', '')

            if '/' in pred_str or '/' in gt_str:
                pred_frac = Fraction(pred_str) if '/' in pred_str else Fraction(float(pred_str))
                gt_frac = Fraction(gt_str) if '/' in gt_str else Fraction(float(gt_str))
                return pred_frac == gt_frac
        except (ValueError, TypeError, ZeroDivisionError):
            pass

        return False


def evaluate_gsm8k_v(eval_file, dataset_mode='visual_implicit'):

    data = load(eval_file)
    data['extracted_answer'] = data['prediction'].apply(extract_numerical_answer)
    data['gt_answer'] = data['answer'].apply(lambda x: float(x) if x and str(x) != 'nan' else None)
    data['is_correct'] = data.apply(
        lambda row: is_correct(row['extracted_answer'], row['gt_answer']),
        axis=1
    )
    total_count = len(data)
    correct_count = data['is_correct'].sum()
    overall_acc = correct_count / total_count if total_count > 0 else 0
    results = {
        'Overall': overall_acc,
        'Total': total_count,
        'Correct': int(correct_count),
    }
    if 'category' in data.columns:
        categories = data['category'].unique()
        for cat in categories:
            if pd.notna(cat) and cat != '':
                cat_data = data[data['category'] == cat]
                cat_total = len(cat_data)
                cat_correct = cat_data['is_correct'].sum()
                cat_acc = cat_correct / cat_total if cat_total > 0 else 0
                results[f'Category_{cat}'] = cat_acc
    if 'subcategory' in data.columns:
        subcategories = data['subcategory'].unique()
        for subcat in subcategories:
            if pd.notna(subcat) and subcat != '':
                subcat_data = data[data['subcategory'] == subcat]
                subcat_total = len(subcat_data)
                subcat_correct = subcat_data['is_correct'].sum()
                subcat_acc = subcat_correct / subcat_total if subcat_total > 0 else 0
                results[f'Subcategory_{subcat}'] = subcat_acc

    score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
    detailed_file = get_intermediate_file_path(eval_file, '_detailed', 'xlsx')
    summary_data = []
    for key, value in results.items():
        if key not in ['Total', 'Correct']:
            summary_data.append({
                'Metric': key,
                'Accuracy': value,
                'Mode': dataset_mode
            })
    summary_df = pd.DataFrame(summary_data)
    dump(summary_df, score_file)
    dump(data, detailed_file)
    print(f"\n{'=' * 50}")
    print(f"GSM8K-V Evaluation Results ({dataset_mode} mode)")
    print(f"{'=' * 50}")
    print(f"Overall Accuracy: {overall_acc:.4f} ({correct_count}/{total_count})")
    if 'category' in data.columns:
        print("\nPer-Category Results:")
        categories = sorted([cat for cat in data['category'].unique() if pd.notna(cat) and cat != ''])
        for cat in categories:
            if f'Category_{cat}' in results:
                print(f"  {cat}: {results[f'Category_{cat}']:.4f}")
    print(f"{'=' * 50}\n")
    return results
