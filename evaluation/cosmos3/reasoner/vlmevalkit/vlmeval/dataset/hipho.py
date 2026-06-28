import json
import os.path as osp
import re
import warnings

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load, toliststr
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import build_judge
from .utils.hipho_prompt_inference import (JUDGE_GRADING_PROMPT_TEMPLATE, RETRY_WARNING_TEMPLATE,
                                           SYSTEM_PROMPTS_EN, SYSTEM_PROMPTS_ZH,
                                           TOTAL_SCORE_WARNING_TEMPLATE)
from .utils.hipho_verifier import answer_tag_reward_fn_for_r1

FAIL_MSG = 'Failed to obtain answer via API.'


def evaluate_hipho_single_problem(judge_model, row, index, judge_kwargs, dataset_instance):
    """Evaluate single HiPhO problem for parallel processing"""

    try:
        result = dataset_instance._evaluate_single_problem(judge_model, row, index, judge_kwargs)
        if result is None:
            return FAIL_MSG

        return {
            'success': True,
            'fine_grained_score': result['fine_grained_score'],
            'coarse_grained_score': result['coarse_grained_score'],
            'final_score': result['final_score'],
            'item_total_points': result['item_total_points'],
            'result': result
        }
    except Exception:
        return FAIL_MSG


class HiPhODataset(ImageBaseDataset):
    """
    HiPhO (High School Physics Olympiad) Benchmark Dataset

    Supports 13 physics olympiad competition datasets:
    - IPhO 2024/2025: International Physics Olympiad
    - EuPhO 2024/2025: European Physics Olympiad
    - APhO 2025: Asian Physics Olympiad
    - PanPhO 2024/2025: Pan-Asian Physics Olympiad
    - NBPhO 2024/2025: Nordic-Baltic Physics Olympiad
    - F_MA 2024/2025: US Physics Competition
    - PanMechanics 2024/2025: Pan-Asian Mechanics Competition

    Integrated with hipho_verifier for fine and coarse-grained evaluation
    """
    TYPE = 'VQA'  # Use VQA type uniformly

    # Dataset URL mapping - points to different splits of HuggingFace dataset
    DATASET_URL = {
        'IPhO_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'IPhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'EuPhO_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'EuPhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'APhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'PanPhO_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'PanPhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'NBPhO_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'NBPhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'F_MA_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'F_MA_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'PanMechanics_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'PanMechanics_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'HiPhO_ALL': 'https://huggingface.co/datasets/HY-Wan/HiPhO',  # All subsets combined
    }

    # MD5 values are empty as HuggingFace datasets are dynamically loaded
    DATASET_MD5 = {
        'IPhO_2024': '',
        'IPhO_2025': '',
        'EuPhO_2024': '',
        'EuPhO_2025': '',
        'APhO_2025': '',
        'PanPhO_2024': '',
        'PanPhO_2025': '',
        'NBPhO_2024': '',
        'NBPhO_2025': '',
        'F_MA_2024': '',
        'F_MA_2025': '',
        'PanMechanics_2024': '',
        'PanMechanics_2025': '',
        'HiPhO_ALL': '',
    }

    # All individual subsets for HiPhO_ALL
    ALL_SUBSETS = [
        'IPhO_2024', 'IPhO_2025', 'EuPhO_2024', 'EuPhO_2025', 'APhO_2025',
        'PanPhO_2024', 'PanPhO_2025', 'NBPhO_2024', 'NBPhO_2025',
        'F_MA_2024', 'F_MA_2025', 'PanMechanics_2024', 'PanMechanics_2025'
    ]

    # Dataset language mapping
    DATASET_LANGUAGES = {
        'PanMechanics_2024': 'zh',
        'PanMechanics_2025': 'zh',
        # All other datasets default to English
    }

    def __init__(self, dataset='IPhO_2025', skip_noimg=False, language=None):
        """Initialize dataset"""
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

        # Auto-detect language if not specified
        if language is None:
            self.language = self.DATASET_LANGUAGES.get(dataset, 'en')
        else:
            self.language = language

        self.is_all_mode = (dataset == 'HiPhO_ALL')

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def load_data(self, dataset):
        """Load dataset from HuggingFace"""
        from datasets import load_dataset

        if dataset == 'HiPhO_ALL':
            # Load all subsets and combine them
            all_data = []
            for subset in self.ALL_SUBSETS:
                hf_dataset = load_dataset('HY-Wan/HiPhO', split=subset)
                subset_data = hf_dataset.to_pandas()
                subset_data['SUB_DATASET'] = subset  # Add source identifier
                all_data.append(subset_data)

            data = pd.concat(all_data, ignore_index=True)
        else:
            # Load single subset
            hf_dataset = load_dataset('HY-Wan/HiPhO', split=dataset)
            data = hf_dataset.to_pandas()

        if 'image_question' in data.columns:
            no_image_placeholder = 'NO_IMAGE_PLACEHOLDER_' + 'x' * 50

            def process_base64_image(base64_data):
                if pd.isna(base64_data) or not str(base64_data).strip() or len(str(base64_data).strip()) < 100:
                    return no_image_placeholder
                return str(base64_data)

            data['image'] = data['image_question'].apply(process_base64_image)

        return data

    def build_prompt(self, line):
        """Build physics competition prompt"""
        if isinstance(line, int):
            line = self.data.iloc[line]

        def safe_str(val):
            return "" if pd.isna(val) or val == '' else str(val)

        context = safe_str(line.get('context', ''))
        question = safe_str(line['question'])
        information = safe_str(line.get('information', ''))

        system_prompt = (SYSTEM_PROMPTS_EN if self.language == 'en'
                         else SYSTEM_PROMPTS_ZH)
        formatted_prompt = (system_prompt.replace('{context}', context)
                            .replace('{problem}', question)
                            .replace('{information}', information))

        msgs = []

        # Check for real image data (excluding placeholders)
        image_val = str(line.get('image', '')).strip()

        if image_val and not image_val.startswith('NO_IMAGE_PLACEHOLDER_'):
            # Use standard VLMEvalKit image processing pipeline
            if self.meta_only:
                tgt_path = toliststr(line['image_path']) if 'image_path' in line else []
            else:
                tgt_path = self.dump_image(line)

            if tgt_path and tgt_path != ['']:
                if isinstance(tgt_path, list):
                    msgs.extend([dict(type='image', value=p) for p in tgt_path])
                else:
                    msgs.append(dict(type='image', value=tgt_path))

        msgs.append(dict(type='text', value=formatted_prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluation function"""
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        # Check if this is HiPhO_ALL mode
        if self.is_all_mode and 'SUB_DATASET' in data.columns:
            judge_name = judge_kwargs.get('model', 'exact_matching')
            all_tmp_file = get_intermediate_file_path(eval_file, f'_ALL_{judge_name}', 'pkl')
            all_score_file = get_intermediate_file_path(eval_file, f'_ALL_{judge_name}_score', 'json')
            nproc = judge_kwargs.pop('nproc', 4)

            if not osp.exists(all_score_file):

                # Load existing subset results if any
                subset_results = {}
                if osp.exists(all_tmp_file):
                    subset_results = load(all_tmp_file)

                all_detailed_results = []

                for subset in self.ALL_SUBSETS:
                    if subset not in data['SUB_DATASET'].values:
                        continue

                    # Check if this subset is already completed
                    if subset in subset_results:
                        continue

                    subset_data = data[data['SUB_DATASET'] == subset].copy()
                    subset_data = subset_data.reset_index(drop=True)

                    # Create subset-specific tmp file
                    subset_tmp_file = get_intermediate_file_path(eval_file, f'_{subset}_{judge_name}', 'pkl')

                    # Initialize judge model for this subset
                    judge_model = None
                    if judge_kwargs.get('model') and judge_kwargs.get('model') != 'exact_matching':
                        judge_kwargs.setdefault('timeout', 600)
                        judge_kwargs.setdefault('retry', 3)
                        judge_kwargs.setdefault('max_tokens', 4096)
                        judge_model = build_judge(**judge_kwargs)
                        if judge_model and not judge_model.working():
                            warnings.warn(f'Judge API not working for {subset}, skipping fine-grained evaluation')
                            judge_model = None

                    # Prepare evaluation tasks for this subset
                    lt = len(subset_data)
                    lines = [subset_data.iloc[i] for i in range(lt)]
                    tups = [(judge_model, line, i, judge_kwargs, self) for i, line in enumerate(lines)]
                    indices = [i for i in range(lt)]

                    # Load existing results for this subset if any
                    ans = {}
                    if osp.exists(subset_tmp_file):
                        ans = load(subset_tmp_file)
                        # Filter out failed results to allow retry
                        ans = {k: v for k, v in ans.items()
                               if (v != FAIL_MSG and isinstance(v, dict)
                                   and v.get('success', False))}

                    # Filter out completed tasks
                    tups = [x for x, i in zip(tups, indices) if i not in ans]
                    indices = [i for i in indices if i not in ans]

                    if len(indices):
                        _ = track_progress_rich(
                            evaluate_hipho_single_problem,
                            tups,
                            nproc=nproc,
                            chunksize=nproc,
                            keys=indices,
                            save=subset_tmp_file,
                        )

                    # Load final results for this subset
                    ans = load(subset_tmp_file)

                    # Aggregate results for this subset
                    fine_grained_total_score = 0.0
                    coarse_grained_total_score = 0.0
                    final_total_score = 0.0
                    max_possible_score = 0.0
                    detailed_results = []
                    failed_count = 0

                    for i in range(len(subset_data)):
                        if i not in ans:
                            failed_count += 1
                            continue

                        if ans[i] == FAIL_MSG or not isinstance(ans[i], dict) or not ans[i].get('success', False):
                            failed_count += 1
                            continue

                        row = subset_data.iloc[i]
                        result_data = ans[i]

                        fine_score = result_data['fine_grained_score']
                        coarse_score = result_data['coarse_grained_score']
                        final_score = result_data['final_score']
                        item_points = result_data['item_total_points']

                        fine_grained_total_score = round(fine_grained_total_score + fine_score, 2)
                        coarse_grained_total_score = round(coarse_grained_total_score + coarse_score, 2)
                        final_total_score = round(final_total_score + final_score, 2)
                        max_possible_score += item_points

                        detailed_item = self._build_result_item(row, i, result_data['result'])
                        detailed_item['subset'] = subset
                        detailed_results.append(detailed_item)
                        all_detailed_results.append(detailed_item)

                    # Note: failed_count tracking for debugging purposes

                    max_possible_score = round(max_possible_score, 2)
                    subset_result = self._build_final_results(
                        fine_grained_total_score, coarse_grained_total_score,
                        final_total_score, max_possible_score)
                    subset_result['subset_name'] = subset
                    subset_result['problem_count'] = len(subset_data)
                    subset_results[subset] = subset_result

                    # Save progress after each subset
                    dump(subset_results, all_tmp_file)

                # Final results
                final_results = {
                    'evaluation_type': 'HiPhO_ALL',
                    'total_subsets': len(subset_results),
                    'subset_results': subset_results
                }

                # Save final results
                dump(final_results, all_score_file)
                self._save_results(eval_file, final_results, all_detailed_results, data)

                # Results summary available in final_results
            else:
                final_results = load(all_score_file)

            return final_results

        # Single subset evaluation with resume support
        judge_name = judge_kwargs.get('model', 'exact_matching')
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge_name}', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{judge_name}_score', 'json')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            # Initialize judge model
            judge_model = None
            if judge_kwargs.get('model') and judge_kwargs.get('model') != 'exact_matching':
                judge_kwargs.setdefault('timeout', 1800)
                judge_kwargs.setdefault('retry', 3)
                judge_kwargs.setdefault('max_tokens', 4096)
                judge_model = build_judge(**judge_kwargs)
                if judge_model and not judge_model.working():
                    warnings.warn('Judge API not working, skipping fine-grained evaluation')
                    judge_model = None

            # Prepare evaluation tasks
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(judge_model, line, i, judge_kwargs, self) for i, line in enumerate(lines)]
            indices = [i for i in range(lt)]

            # Load existing results if any
            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
                # Filter out failed results to allow retry
                ans = {k: v for k, v in ans.items()
                       if (v != FAIL_MSG and isinstance(v, dict)
                           and v.get('success', False))}

            # Filter out completed tasks
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    evaluate_hipho_single_problem,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            # All problems already evaluated

            # Load final results
            ans = load(tmp_file)

            # Aggregate results
            fine_grained_total_score = 0.0
            coarse_grained_total_score = 0.0
            final_total_score = 0.0
            max_possible_score = 0.0
            detailed_results = []
            failed_count = 0

            for i in range(len(data)):
                if i not in ans:
                    failed_count += 1
                    continue

                if ans[i] == FAIL_MSG or not isinstance(ans[i], dict) or not ans[i].get('success', False):
                    failed_count += 1
                    continue

                row = data.iloc[i]
                result_data = ans[i]

                fine_score = result_data['fine_grained_score']
                coarse_score = result_data['coarse_grained_score']
                final_score = result_data['final_score']
                item_points = result_data['item_total_points']

                fine_grained_total_score = round(fine_grained_total_score + fine_score, 2)
                coarse_grained_total_score = round(coarse_grained_total_score + coarse_score, 2)
                final_total_score = round(final_total_score + final_score, 2)
                max_possible_score += item_points

                detailed_item = self._build_result_item(row, i, result_data['result'])
                detailed_results.append(detailed_item)

            # Note: failed_count tracking for debugging purposes

            max_possible_score = round(max_possible_score, 2)
            results = self._build_final_results(
                fine_grained_total_score, coarse_grained_total_score,
                final_total_score, max_possible_score)

            # Save final results
            dump(results, score_file)
            self._save_results(eval_file, results, detailed_results, data)
            self._print_summary(results)
        else:
            results = load(score_file)

        return results

    def _evaluate_single_problem(self, judge_model, row, index, judge_kwargs):
        """Evaluate single problem"""
        # Extract fields
        prediction = str(row['prediction']).strip()
        ground_truth = self._safe_parse_json_field(row.get('answer', ''))
        answer_type = self._safe_parse_json_field(row.get('answer_type', 'Open-End'))
        unit = self._safe_parse_json_field(row.get('unit', ''))
        points = self._safe_parse_points_field(row.get('points', 0))
        marking = self._safe_parse_json_field(row.get('marking', ''))

        item_total_points = sum(points) if points else 0.0

        # Fine-grained evaluation
        fine_grained_score, marking_detailed_scores = self._evaluate_fine_grained(
            prediction, marking, points, judge_model, row.get('question', '')
        )

        # Coarse-grained evaluation
        coarse_grained_score, extracted_pred = self._evaluate_coarse_grained(
            prediction, ground_truth, answer_type, unit, points,
            row.get('question', ''), judge_model
        )

        # Final score is the maximum of both
        final_score = max(fine_grained_score, coarse_grained_score)

        # Return single problem result
        return {
            'index': index,
            'fine_grained_score': fine_grained_score,
            'coarse_grained_score': coarse_grained_score,
            'final_score': final_score,
            'extracted_pred': extracted_pred,
            'marking_detailed_scores': marking_detailed_scores,
            'item_total_points': item_total_points,
            'ground_truth': ground_truth,
            'answer_type': answer_type,
            'unit': unit,
            'points': points,
            'marking': marking,
            'prediction': prediction
        }

    def _evaluate_fine_grained(self, prediction, marking, points, judge_model, question):
        """Fine-grained evaluation with retry mechanism"""

        if not marking:
            return 0.0, []

        # Check for multiple marking criteria sets
        if self._has_multiple_marking_sets(marking):
            return self._evaluate_multiple_marking_sets(prediction, marking, points, judge_model, question)

        # Handle single criterion set wrapped in a list
        if len(marking) == 1 and isinstance(marking[0], list):
            marking = marking[0]  # Unwrap single criterion set

        scoring_criteria = self._parse_marking_criteria(marking)
        max_possible_score = sum(points) if points else 0.0
        max_retries = 3  # Maximum retry attempts

        for attempt in range(max_retries + 1):
            scores = []
            detailed_scores = []

            api_failed_count = 0
            for i, criterion in enumerate(scoring_criteria):
                score, response = self._evaluate_single_criterion(
                    prediction, criterion, judge_model, question,
                    max_total_score=max_possible_score,
                    current_attempt=attempt
                )

                # Check if API failed
                if score is None or "API_FAILED" in str(response) or "EMPTY_RESPONSE" in str(response):
                    api_failed_count += 1
                    # For failed criteria, don't add to total score calculation
                    scores.append(0)  # Temporary placeholder, will retry later
                    detailed_scores.append({
                        'marking_criterion': criterion['description'],
                        'score': None,
                        'index': criterion['index'],
                        'attempt': attempt + 1,
                        'judge_response': response,
                        'evaluation_status': 'API_FAILED'
                    })
                else:
                    scores.append(score)
                detailed_scores.append({
                    'marking_criterion': criterion['description'],
                    'score': round(score, 2),
                    'index': criterion['index'],
                    'attempt': attempt + 1,
                    'judge_response': response,
                    'evaluation_status': 'SUCCESS'
                })

            # If API failed and not the last attempt, retry entire scoring process
            if api_failed_count > 0 and attempt < max_retries:
                continue

            # Calculate total score (only count successful evaluations)
            valid_scores = [s for s, ds in zip(scores, detailed_scores) if ds.get('evaluation_status') == 'SUCCESS']
            total_score = sum(valid_scores)

            # If all evaluations failed, return clear failure information
            if api_failed_count == len(scoring_criteria):
                return 0.0, [{
                    'marking_criterion': 'ALL_CRITERIA_FAILED',
                    'score': None,
                    'index': -1,
                    'judge_response': f'All {len(scoring_criteria)} evaluation criteria failed due to API issues',
                    'evaluation_status': 'TOTAL_FAILURE'
                }]

            if total_score <= max_possible_score or max_possible_score == 0:
                for detailed_score in detailed_scores:
                    detailed_score['retry_info'] = (
                        f"Attempt {attempt + 1} successful" if attempt > 0
                        else "First attempt successful")
                    detailed_score['final_success'] = True

                return round(total_score, 2), detailed_scores
            else:
                if attempt < max_retries:
                    continue  # Retry
                else:
                    # Force adjustment
                    scale_factor = max_possible_score / total_score
                    adjusted_scores = [score * scale_factor for score in scores]

                    for i, score in enumerate(adjusted_scores):
                        detailed_scores[i]['original_score'] = detailed_scores[i]['score']
                        detailed_scores[i]['score'] = round(score, 2)
                        detailed_scores[i]['forced_adjustment'] = True
                        detailed_scores[i]['scale_factor'] = round(scale_factor, 3)

                    return round(sum(adjusted_scores), 2), detailed_scores

        return 0.0, []

    def _evaluate_coarse_grained(self, prediction, ground_truth, answer_type, unit, points, question, judge_model=None):
        """Coarse-grained evaluation based on hipho_verifier answer matching"""
        extracted_pred = ""

        if ground_truth:
            # Use hipho_verifier
            total_score, total_point, extracted_preds, extracted_gts, scored_by_list = (
                answer_tag_reward_fn_for_r1(
                    prediction, ground_truth, problem=question, points=points,
                    use_xverify=True, debug=False, judge_model=judge_model))

            extracted_pred = ", ".join([str(p) for p in extracted_preds if p])
            return round(total_point, 2), extracted_pred

        return 0.0, extracted_pred

    def _evaluate_single_criterion(self, prediction, criterion, judge_model,
                                   question, max_total_score=None,
                                   current_attempt=0):
        """Evaluate single marking criterion using judge model"""

        # Build total score limit warning
        total_score_warning = ""
        if max_total_score is not None and max_total_score > 0:
            total_score_warning = TOTAL_SCORE_WARNING_TEMPLATE.format(
                max_total_score=max_total_score,
                current_attempt=current_attempt + 1
            )

        retry_warning = ""
        if current_attempt > 0:
            retry_warning = RETRY_WARNING_TEMPLATE

        # Use unified prompt template
        prompt = JUDGE_GRADING_PROMPT_TEMPLATE.format(
            question=question,
            prediction=prediction,
            criterion_description=criterion['description'],
            total_score_warning=total_score_warning,
            retry_warning=retry_warning
        )

        response = judge_model.generate(prompt).strip()

        # Check if it's an API failure message
        if 'Failed to obtain answer via API' in response:
            return None, f"API_FAILED: {response}"

        # Check if response is empty or invalid
        if not response or len(response.strip()) == 0:
            return None, "EMPTY_RESPONSE"

        score = self._extract_score_from_response(response)
        return score, response

    def _safe_parse_json_field(self, field_value):
        """Safely parse JSON field"""
        if pd.isna(field_value) or field_value == '':
            return []

        if isinstance(field_value, list):
            return field_value

        field_str = str(field_value).strip()
        if field_str.startswith('[') and field_str.endswith(']'):
            try:
                return json.loads(field_str)
            except json.JSONDecodeError:
                return [field_str]
        else:
            return [field_str] if field_str != 'nan' else []

    def _safe_parse_points_field(self, points_value):
        """Safely parse points field"""
        if pd.isna(points_value):
            return [0.0]

        if isinstance(points_value, list):
            return [float(p) for p in points_value if p is not None]

        if isinstance(points_value, (int, float)):
            return [float(points_value)]

        points_str = str(points_value).strip()
        if points_str.startswith('[') and points_str.endswith(']'):
            try:
                parsed = json.loads(points_str)
                return [float(p) for p in parsed if p is not None]
            except (json.JSONDecodeError, ValueError):
                pass

        try:
            return [float(points_str)]
        except ValueError:
            return [0.0]

    def _has_valid_marking(self, marking):
        """Check if marking contains valid scoring criteria"""
        if not marking:
            return False

        if not isinstance(marking, list):
            return False

        if len(marking) == 0:
            return False

        for item in marking:
            if item is None:
                continue

            if isinstance(item, list):
                if len(item) > 0:
                    return True
            elif isinstance(item, str):
                stripped = item.strip()
                if stripped and stripped.lower() not in ['', 'nan', 'none', 'null']:
                    return True
            else:
                return True

        return False

    def _has_multiple_marking_sets(self, marking):
        """Check for multiple marking criteria sets"""

        if not marking or len(marking) == 0:
            return False

        # Fixed logic: if length is 1 and first element is a list, this is a single criterion set (wrapped in list)
        if len(marking) == 1 and isinstance(marking[0], list):
            return False

        # Only when length > 1 and all elements are lists, it's truly multiple criterion sets
        result = len(marking) > 1 and all(isinstance(item, list) for item in marking)
        return result

    def _evaluate_multiple_marking_sets(self, prediction, marking_sets, points, judge_model, question):
        """Evaluate multiple marking criteria sets, take highest score"""

        best_score = 0.0
        best_detailed_scores = []

        for set_idx, marking_set in enumerate(marking_sets):
            score, detailed_scores = self._evaluate_single_marking_set(
                prediction, marking_set, points, judge_model, question
            )

            # Update best score
            if score > best_score:
                best_score = score
                best_detailed_scores = detailed_scores
                # Add marker to best detailed scores
                for detailed_score in best_detailed_scores:
                    detailed_score['best_marking_set'] = set_idx + 1

        return round(best_score, 2), best_detailed_scores

    def _evaluate_single_marking_set(self, prediction, marking, points, judge_model, question):
        """Evaluate single marking criteria set"""
        scoring_criteria = self._parse_marking_criteria(marking)
        max_possible_score = sum(points) if points else 0.0

        scores = []
        detailed_scores = []

        for criterion in scoring_criteria:
            score, response = self._evaluate_single_criterion(
                prediction, criterion, judge_model, question,
                max_total_score=max_possible_score,
                current_attempt=0
            )
            scores.append(score)

            # Save detailed scores for each marking
            detailed_scores.append({
                'marking_criterion': criterion['description'],
                'score': round(score, 2),
                'index': criterion['index'],
                'judge_response': response
            })

        total_score = sum(scores)

        # Scale down if exceeds maximum score
        if total_score > max_possible_score and max_possible_score > 0:
            scale_factor = max_possible_score / total_score
            total_score = max_possible_score
            for detailed_score in detailed_scores:
                detailed_score['original_score'] = detailed_score['score']
                detailed_score['score'] = round(detailed_score['score'] * scale_factor, 2)
                detailed_score['scaled'] = True

        return round(total_score, 2), detailed_scores

    def _parse_marking_criteria(self, marking_list):
        """Parse marking scoring criteria"""
        criteria = []
        if not marking_list:
            return criteria

        # Handle nested list cases
        flattened_marking = []
        for item in marking_list:
            if isinstance(item, list):
                flattened_marking.extend(item)
            else:
                flattened_marking.append(item)

        for i, marking_criterion in enumerate(flattened_marking):
            if marking_criterion and str(marking_criterion).strip():
                criteria.append({
                    'description': str(marking_criterion).strip(),
                    'index': i
                })

        return criteria

    def _extract_score_from_response(self, response):
        """Extract score from model response"""
        if not response:
            return 0.0

        response = response.strip()

        # Extract score using boxed format
        boxed_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'boxed\{([^}]+)\}',
        ]

        for pattern in boxed_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in reversed(matches):
                match = match.strip()
                if match:
                    try:
                        score = float(match)
                        return round(score, 2)
                    except ValueError:
                        nums = re.findall(r'\d+\.?\d*', match)
                        if nums:
                            try:
                                score = float(nums[-1])
                                return round(score, 2)
                            except ValueError:
                                continue

        # Find numbers
        all_numbers = re.findall(r'[0-9]*\.?[0-9]+', response)
        if all_numbers:
            try:
                score = float(all_numbers[-1])
                return round(score, 2)
            except ValueError:
                pass

        return 0.0

    def _build_result_item(self, row, index, result):
        """Build detailed result item"""
        # has_marking = (result['marking'] and len(result['marking']) > 0
        #                and self._has_valid_marking(result['marking']))
        earned_points = max(result['fine_grained_score'], result['coarse_grained_score'])

        return {
            "id": str(row.get('id', f"{self.dataset_name}_{index+1}")),
            "context": str(row.get('context', '')).strip(),
            "question": str(row.get('question', '')).strip(),
            "solution": str(row.get('solution', '')).strip(),
            "marking": result['marking'] if result['marking'] else [],
            "marking_detailed_scores": result['marking_detailed_scores'],
            "answer": [f"\\boxed{{{ans}}}" for ans in result['ground_truth']] if result['ground_truth'] else [''],
            "answer_type": result['answer_type'] if result['answer_type'] else ['Open-End'],
            "unit": result['unit'] if result['unit'] else [''],
            "points": result['points'] if result['points'] else [0.0],
            "modality": str(row.get('modality', 'text')).strip(),
            "field": str(row.get('field', '')).strip(),
            "source": self.dataset_name,
            "test_result": str(result['prediction']),
            "test_answer": ([f"\\boxed{{{ans.strip()}}}"
                             for ans in result['extracted_pred'].split(", ")
                             if ans.strip()]
                            if result['extracted_pred'] else ['']),
            "fine_grained_score": result['fine_grained_score'],
            "coarse_grained_score": result['coarse_grained_score'],
            "earned_points": earned_points
        }

    def _build_final_results(self, fine_total, coarse_total, final_total, max_score):
        """Build final results"""
        fine_rate = round((fine_total / max_score * 100), 2) if max_score > 0 else 0.0
        coarse_rate = round((coarse_total / max_score * 100), 2) if max_score > 0 else 0.0
        final_rate = round((final_total / max_score * 100), 2) if max_score > 0 else 0.0

        return {
            'fine_grained_total_score': fine_total,
            'fine_grained_score_rate': fine_rate,
            'coarse_grained_total_score': coarse_total,
            'coarse_grained_score_rate': coarse_rate,
            'max_possible_score': max_score,
            'total_score': final_total,  # Now uses the correct final score (max of fine and coarse)
            'score_rate': final_rate,
        }

    def _save_results(self, eval_file, results, detailed_results, data):
        """Save evaluation results"""
        score_file = eval_file.replace('.xlsx', '_score.json')
        detailed_file = eval_file.replace('.xlsx', '_detailed_results.json')
        detailed_xlsx_file = eval_file.replace('.xlsx', '_detailed.xlsx')

        dump(results, score_file)
        dump(detailed_results, detailed_file)

        eval_data_with_results = data.copy()
        eval_data_with_results['fine_grained_score'] = [r['fine_grained_score'] for r in detailed_results]
        eval_data_with_results['coarse_grained_score'] = [r['coarse_grained_score'] for r in detailed_results]
        eval_data_with_results['earned_points'] = [r['earned_points'] for r in detailed_results]
        eval_data_with_results['marking_detailed_scores'] = [
            json.dumps(r['marking_detailed_scores'], ensure_ascii=False) if r['marking_detailed_scores'] else '[]'
            for r in detailed_results
        ]
        dump(eval_data_with_results, detailed_xlsx_file)

    def _print_summary(self, results):
        """Print evaluation summary"""
        print(f"✅ {self.dataset_name} evaluation completed!")
        print(f"🏆 Overall score: {results['total_score']:.2f} / "
              f"{results['max_possible_score']:.2f} "
              f"({results['score_rate']:.2f}%)")
        print(f"📊 Fine-grained score: "
              f"{results['fine_grained_total_score']:.2f} "
              f"({results['fine_grained_score_rate']:.2f}%)")
        print(f"🎯 Coarse-grained score: "
              f"{results['coarse_grained_total_score']:.2f} "
              f"({results['coarse_grained_score_rate']:.2f}%)")
