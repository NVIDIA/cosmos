import ast
import math
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from vlmeval.smp.file import get_intermediate_file_path
from vlmeval.smp.vlm import gpt_key_set
from ..judge_util import build_judge
from .llm_extract import parallel_llm_extract
from .matching_func import can_match_na, can_match_option
from .tools.files import build_eval_paths, get_judge_tag_from_score_fn


def exact_match(pred: str, target: str) -> float:
    pred = str(pred).strip().lower()
    target = str(target).strip().lower()
    return 1. if pred == target else 0.


def abs_dist_norm(pred: float, target: float) -> float:
    if target == 0.0:
        return abs(pred - target)
    else:
        return abs((pred - target) / target)


def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    # TODO：check this, should be + 1, but in vsi code this is + 2
    num_pts = int((end - start) / interval + 2)
    conf_intervs = np.linspace(start, end, num_pts)
    err = abs_dist_norm(pred, target)
    ok = (err <= (1 - conf_intervs)).astype(float)
    return float(ok.mean())


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _safe_len_candidates(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, list):
        return len(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return len(parsed)
        except Exception:
            return None
    return None


def _ensure_options_count_row(row, default_choices=4):
    n = None
    if 'candidates' in row:
        n = _safe_len_candidates(row['candidates'])
    elif 'options' in row:
        n = _safe_len_candidates(row['options'])
    return n if (isinstance(n, int) and n > 0) else default_choices


# ---------- Rule-based scoring ----------
def compute_mcq_score(df: pd.DataFrame) -> pd.DataFrame:
    preds_extracted, acc = [], []
    for _, r in df.iterrows():
        pred_raw = str(r['prediction'])
        gt_raw = str(r['answer']).strip()

        pred = can_match_option(pred_raw)
        gt = can_match_option(gt_raw)

        preds_extracted.append(pred)
        acc.append(exact_match(pred, gt))

    df = df.copy()
    df['pred_extracted'] = preds_extracted
    df['hit'] = acc
    return df


def compute_na_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Mean Relative Accuracy (MRA) for numerical-answer (NA) items,
    following the VSI codebase.

    Definition:
        For prediction ŷ, ground-truth y, and confidence threshold θ,
        the relative accuracy is 1[ |ŷ - y| / |y| < 1 - θ ].
        MRA averages this relative accuracy over θ ∈ {0.50, 0.55, ..., 0.95}.

    Reference:
        Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces. (https://arxiv.org/pdf/2412.14171)  # noqa: E501
    """
    preds_extracted, mra_scores = [], []

    for _, r in df.iterrows():
        pred_num = can_match_na(str(r['prediction']))
        gt_num = to_float(r['answer'])

        preds_extracted.append(pred_num)

        if pred_num is None or gt_num is None or math.isnan(gt_num):
            mra_scores.append(0.0)          # WORST_CASE
        else:
            mra_scores.append(mean_relative_accuracy(pred_num, gt_num, .5, .95, .05))

    df = df.copy()
    df['pred_extracted'] = preds_extracted
    df['MRA:.5:.95:.05'] = mra_scores
    return df


def compute_caa_score(df_all: pd.DataFrame, default_choices: int = 4) -> float:
    """
    Compute Class-Adjusted Accuracy (CAA) for Multiple Choice Questions.

    Definition:
    For each item i with n_i options and correctness indicator X_i ∈ {0, 1},
        CAA = (Σ_i X_i - Σ_i (1 / n_i)) / (N - Σ_i (1 / n_i))
            - N   : total number of evaluated items.
            - X_i : 1 if the prediction for item i is correct, otherwise 0.
            - n_i : number of answer options for item i.

    Reference:
        SITE: Towards Spatial Intelligence Thorough Evaluation. (https://arxiv.org/pdf/2505.05456)

    """
    if len(df_all) == 0:
        return 0.0
    n_list = df_all.apply(lambda r: _ensure_options_count_row(r, default_choices), axis=1)
    xi = df_all['hit'].astype(int)
    N = len(df_all)
    sum_Xi = xi.sum()
    sum_1_ni = (1.0 / n_list).sum()
    denom = N - sum_1_ni
    return float((sum_Xi - sum_1_ni) / denom) if denom != 0 else 0.0


# High-level evaluation core for MCQ-style datasets
def eval_mcq_score(
    *,
    load_fn,
    eval_file: str,
    score_fn,
    group_col: str | list[str] = 'category',
    order: list[str] | dict[str, list[str]] | None = None,
    dataset_name: str = 'MCQ',
    return_scored: bool = False,
):
    judge_tag = get_judge_tag_from_score_fn(score_fn)
    result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag)

    # only effective for LLM-based score_fn; rule-based is no-op
    attach_score_cache(
        score_fn=score_fn,
        eval_file=eval_file,
        judge_tag=judge_tag,
        key_col='index',
    )

    data = load_fn(eval_file)
    if 'index' in data.columns:
        data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]

    mcq_scored = score_fn(data.copy())

    # ---------- group_cols / order_map ----------
    if isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = list(group_col)

    if isinstance(order, dict) or order is None:
        order_map: dict[str, list[str]] = order or {}
    else:
        order_map = {group_cols[0]: order}

    summary = OrderedDict()
    overall_acc = float(mcq_scored['hit'].mean()) if len(mcq_scored) else 0.0
    summary['overall'] = overall_acc * 100.0

    # ---------- category && tasks ----------
    for gc in group_cols:
        if gc not in mcq_scored.columns:
            continue

        preferred = order_map.get(gc, []) or []
        present = list(mcq_scored[gc].dropna().unique().tolist())
        remain = [c for c in present if c not in preferred]
        cat_order = preferred + remain

        prefix = '' if len(group_cols) == 1 else f'{gc}.'

        for cat in cat_order:
            sub = mcq_scored[mcq_scored[gc] == cat]
            if len(sub):
                acc = float(sub['hit'].mean()) * 100.0
                summary[f'{prefix}{cat}_accuracy'] = acc

    tab_keys = ', '.join(list(summary.keys()))
    tab_vals = ', '.join([f'{v:.3f}' for v in summary.values()])
    summary['tabulated_keys'] = tab_keys
    summary['tabulated_results'] = tab_vals

    # ---------- pkl ----------
    try:
        import pickle
        with open(result_file, 'wb') as f:
            pickle.dump({'mcq_scored': mcq_scored, 'summary': summary}, f)
        print(f'[save] result saved to {result_file}')
    except Exception as e:
        warnings.warn(f'[save] failed to save result to {result_file}: {e}')

    # ---------- extract_matching.xlsx ----------
    try:
        prefer_front = [
            'index', 'question_type',
            group_cols[0] if group_cols else None,
            'prediction', 'pred_extracted', 'answer', 'hit'
        ]
        prefer_front = [c for c in prefer_front if c is not None]

        merged = mcq_scored.copy()
        ordered_cols = [c for c in prefer_front if c in merged.columns] + \
                       [c for c in merged.columns if c not in prefer_front]
        merged = merged[ordered_cols]
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            merged.to_excel(writer, sheet_name='ALL', index=False)
        print(f'[save] extract & matching saved to {xlsx_path}')
    except Exception as e:
        warnings.warn(f'[save] failed to save extract xlsx to {xlsx_path}: {e}')

    # ---------- acc.tsv ----------
    try:
        acc_df = pd.DataFrame(
            [(k, v) for k, v in summary.items()
             if k not in ('tabulated_keys', 'tabulated_results')],
            columns=['metric', 'value']
        )

        metric_order = ['overall']

        for gc in group_cols:
            preferred = order_map.get(gc, []) or []
            prefix = '' if len(group_cols) == 1 else f'{gc}.'
            metric_order += [f'{prefix}{c}_accuracy' for c in preferred]

        metric_order += [k for k in acc_df['metric'].tolist()
                         if k not in metric_order]

        acc_df = acc_df.set_index('metric').reindex(metric_order).dropna(subset=['value'])
        wide = acc_df.T
        wide.to_csv(acc_tsv_path, sep='\t', index=False, float_format='%.4f')

        print(f'[save] accuracy table saved to {acc_tsv_path}')
    except Exception as e:
        warnings.warn(f'[save] failed to save acc tsv to {acc_tsv_path}: {e}')

    print(f'[{dataset_name}] summary: {summary}')
    if return_scored:
        return summary, mcq_scored
    return summary


# ---------- LLM-based scoring ----------
def compute_score_llm(
    df: pd.DataFrame,
    model,
    *,
    mode: str = 'mcq',
    max_retry: int = 3,
    nproc: int = 4,
    **extra,
) -> pd.DataFrame:
    """
    LLM-based MCQ/VQA scoring.

    Args:
        df: input dataframe (must contain at least question / prediction / answer).
        model: judge model with .generate(prompt: str) -> str
        mode: 'mcq' or 'vqa'
        max_retry: max retry times per sample
        nproc: number of worker threads for parallel judging
        extra:
            - cache_file: optional cache pkl path
            - key_col: column name used as cache key (default: 'index')
    """
    cache_file: str | None = extra.get('cache_file', None)
    key_col: str = extra.get('key_col', 'index')

    df = df.copy()
    grades, extracted_list = parallel_llm_extract(
        df=df,
        model=model,
        mode=mode,
        max_retry=max_retry,
        nproc=nproc,
        cache_file=cache_file,
        key_col=key_col,
    )

    hits = [1 if g == 'A' else 0 for g in grades]

    df['judge_grade'] = grades          # 'A' / 'B' / 'C'
    df['pred_extracted'] = extracted_list
    df['hit'] = hits
    return df


def compute_na_score_llm(
    df: pd.DataFrame,
    model,
    *,
    mode: str = 'vqa',
    max_retry: int = 3,
    nproc: int = 4,
    **extra,
) -> pd.DataFrame:
    """
    LLM-based NA scoring.

    Workflow:
      - Use the LLM to extract the final numeric answer (mode='vqa')
      - Then compute MRA from extracted number and ground truth.
    """

    cache_file: str | None = extra.get('cache_file', None)
    key_col: str = extra.get('key_col', 'index')

    df = df.copy()
    grades, extracted_list = parallel_llm_extract(
        df=df,
        model=model,
        mode=mode,
        max_retry=max_retry,
        nproc=nproc,
        cache_file=cache_file,
        key_col=key_col,
    )

    pred_nums: list[float | None] = []
    mra_list: list[float] = []

    for grade, extracted, (_, row) in zip(
        grades, extracted_list, df.iterrows()
    ):
        pred_num = to_float(extracted)
        gt_num = to_float(row.get('answer', None))

        if (
            pred_num is None
            or gt_num is None
            or math.isnan(gt_num)
            or grade == 'C'
        ):
            mra = 0.0
        else:
            mra = mean_relative_accuracy(pred_num, gt_num, 0.5, 0.95, 0.05)

        pred_nums.append(pred_num)
        mra_list.append(mra)

    df['judge_grade'] = grades              # 'A' / 'B' / 'C'
    df['pred_extracted'] = pred_nums        # float or None
    df['MRA:.5:.95:.05'] = mra_list
    return df


# ---------- Factory func ----------
def attach_score_cache(
    score_fn,
    eval_file: str,
    judge_tag: str,
    *,
    key_col: str = 'index',
    sub_tag: str | None = None,
):
    """
    Attach a cache file to an LLM-based score_fn for resume.

    cache_file naming convention:
        <eval_file base>_{judge_tag}[_<sub_tag>]_judge.<ext>

    This function is no-op if:
        - score_fn is None
        - score_fn.judge_mode != 'llm'
        - score_fn does not have 'llm_cache' attribute
    """
    if score_fn is None:
        return None

    if getattr(score_fn, 'judge_mode', 'rule') != 'llm':
        return None

    llm_cache = getattr(score_fn, 'llm_cache', None)
    if llm_cache is None:
        return None

    suffix_parts = [f'_{judge_tag}']
    if sub_tag:
        suffix_parts.append(f'_{sub_tag}')
    suffix = ''.join(suffix_parts) + '_judge'

    cache_file = get_intermediate_file_path(
        eval_file,
        suffix=suffix,
        target_format='pkl',
    )

    llm_cache['file'] = cache_file
    llm_cache['key_col'] = key_col
    return cache_file


def _build_llm_judge(judge_kwargs: dict, *, task_name: str):
    """
    Try to build an LLM judge from judge_kwargs.

    Returns:
        - model instance if everything is OK;
        - None if API key is missing or the judge is not working.
    """
    if not gpt_key_set():
        warnings.warn(
            f'OPENAI_API_KEY is not set properly, fallback to rule-based {task_name} scoring.'
        )
        return None

    model = build_judge(**judge_kwargs)
    if (model is None) or (hasattr(model, "working") and not model.working()):
        warnings.warn(
            f'LLM judge is not working properly, fallback to rule-based {task_name} scoring.'
        )
        return None

    return model


def _build_score_fn(
    *,
    task_name: str,
    judge_kwargs: dict,
    rule_fn: callable,
    llm_fn: callable,
    mode: str | None = None,
):
    """
    Generic factory to choose between rule-based scoring and LLM-based scoring.

    Args:
        task_name: for logging only, e.g. "MCQ" / "NA".
        judge_kwargs: kwargs used to build the judge model.
        rule_fn: rule-based scorer, signature: rule_fn(df) -> df.
        llm_fn: LLM-based scorer, signature:
            llm_fn(df, model, mode=..., max_retry=..., nproc=...) -> df
        mode: if not None, passed as mode=mode to llm_fn (for MCQ).
    """
    model_name = judge_kwargs.get('model', None)

    def _make_rule_score_fn() -> callable:
        def score_fn(df: pd.DataFrame) -> pd.DataFrame:
            return rule_fn(df)

        score_fn.judge_mode = 'rule'
        # for rule-based path, if model_name is None, we treat it as 'extract_matching'
        score_fn.judge_model = model_name or 'extract_matching'
        return score_fn

    # 1. Rule-based path
    if model_name is None or model_name in ('exact_matching', 'extract_matching'):
        return _make_rule_score_fn()

    # 2. Build LLM judge
    model = _build_llm_judge(judge_kwargs, task_name=task_name)
    if model is None:
        return _make_rule_score_fn()

    max_retry = judge_kwargs.get('retry', 3)
    nproc = judge_kwargs.get('nproc', judge_kwargs.get('api_nproc', 1) or 1)

    llm_cache = {
        'file': None,
        'key_col': 'index',
    }

    # 3. Wrap into df -> df scorer
    def score_fn(df: pd.DataFrame) -> pd.DataFrame:
        kwargs = dict(
            df=df,
            model=model,
            mode=mode,
            max_retry=max_retry,
            nproc=nproc,
            cache_file=llm_cache['file'],
            key_col=llm_cache['key_col'],
        )
        return llm_fn(**kwargs)

    score_fn.judge_mode = 'llm'
    score_fn.judge_model = model_name
    score_fn.llm_cache = llm_cache
    return score_fn


def build_mcq_score_fn(**judge_kwargs):
    """
    Build an MCQ scoring function based on judge_kwargs['model'].
    """
    return _build_score_fn(
        task_name='MCQ',
        judge_kwargs=judge_kwargs,
        rule_fn=compute_mcq_score,
        llm_fn=compute_score_llm,   # note: this is the generic LLM scorer
        mode='mcq',
    )


def build_na_score_fn(**judge_kwargs):
    """
    Build an NA scoring function based on judge_kwargs['model'].
    """
    return _build_score_fn(
        task_name='NA',
        judge_kwargs=judge_kwargs,
        rule_fn=compute_na_score,
        llm_fn=compute_na_score_llm,
        mode='vqa',
    )
