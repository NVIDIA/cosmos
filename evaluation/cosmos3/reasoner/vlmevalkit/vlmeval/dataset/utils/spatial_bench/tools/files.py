from vlmeval.smp.file import get_intermediate_file_path


def _judge_tag_from_mode_and_model(judge_mode: str | None, judge_model: str | None) -> str:
    """
    Map (judge_mode, judge_model) to a judge_tag string used in filenames.

    Returns examples:
        - 'extract_matching'
        - 'llm_gpt-4o' / 'llm_matching'
    """
    if judge_mode == 'llm':
        return f'llm_{judge_model}' if judge_model else 'llm_matching'
    return 'extract_matching'


def get_judge_tag_from_score_fn(score_fn) -> str:
    """
    Infer judge_tag from attributes attached to score_fn.

    This relies on _build_score_fn setting:
        score_fn.judge_mode
        score_fn.judge_model
    """
    judge_mode = getattr(score_fn, 'judge_mode', 'rule')
    judge_model = getattr(score_fn, 'judge_model', None)
    return _judge_tag_from_mode_and_model(judge_mode, judge_model)


def build_eval_paths(eval_file: str, judge_tag: str):
    """
    Build unified evaluation-related file paths from eval_file and judge_tag.

    It returns:
        - result_file: *_result.pkl
        - xlsx_path  : *_{judge_tag}.xlsx
        - acc_path   : *_acc.{EVAL_FORMAT or default csv}
    """
    result_file = get_intermediate_file_path(
        eval_file,
        suffix='_result',
        target_format='pkl'
    )

    xlsx_path = get_intermediate_file_path(
        eval_file,
        suffix=f'_{judge_tag}',
        target_format='xlsx'
    )

    acc_path = get_intermediate_file_path(
        eval_file,
        suffix=f'_{judge_tag}_acc'
        # target_format=None -> resolved via suffix '_acc' -> get_eval_file_format()
    )

    return result_file, xlsx_path, acc_path
