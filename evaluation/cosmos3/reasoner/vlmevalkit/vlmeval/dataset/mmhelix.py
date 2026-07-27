import io
import os
import os.path as osp

import numpy as np
import pandas as pd
from tqdm import tqdm

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils.mmhelix.metrics import metrics
from vlmeval.dataset.utils.mmhelix.parser import parser
from vlmeval.smp import load
from vlmeval.smp.file import LMUDataRoot, file_size


class MMHELIX(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MM-HELIX': '',
        'MM-HELIX_lang': '',
    }
    GROUP_LIST = {
        'graph_problems': [
            'connectivity_test', 'eulerian_cycle', 'eulerian_path',
            'graph_isomorphism', 'hamiltonian_cycle', 'hamiltonian_path',
            'max_flow', 'shortest_distance_weighted', 'topological_sort',
        ],
        'puzzles': [
            'Calcudoku', 'Kukurasu', 'Skyscrapers', 'WordLadder', 'eulero',
            'numbrix', 'snake', 'aquarium', 'binairo', 'bridges', 'campsite',
            'futoshiki', 'hitori', 'kakuro', 'nonogram', 'shingoki', 'tapa',
            'wordsearch', 'sudoku'
        ],
        'algorithm_problems': [
            '24Points',
            'BestTimeToBuyAndSellStock',
            'ContainerWithMostWater', 'CountHillsAndValleys',
            'CryptoMath', 'HIndex', 'LargestRectangleInHistogram',
            'longest_increasing_subsequence',
            'trapping_rain_water',
        ],
        'games': [
            'sokoban', 'hanoi', 'maze', 'minesweeper', 'slidingpuzzle', 'nibbles',
        ],
    }

    def __init__(self, dataset='MM-HELIX', skip_noimg=False, use_verifier=True):
        super().__init__(dataset, skip_noimg)
        # import pdb; pdb.set_trace()
        self.language_only = 'lang' in dataset
        self.image_base_path = osp.join(LMUDataRoot(), 'images', 'MM-HELIX')

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), 'MM-HELIX.tsv')
        image_dir = osp.join(LMUDataRoot(), 'images', 'MM-HELIX')

        self._download_from_huggingface(data_path, image_dir)

        # Load processed local TSV file
        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path

        return load(data_path)

    def _download_from_huggingface(self, data_path, image_dir):
        """Download dataset from Hugging Face, save images locally, and update DataFrame before saving as TSV."""
        try:
            from datasets import load_dataset

            # 1. Load dataset from Hugging Face
            print("Loading dataset from tianhao2k/MM-HELIX...")
            # Use trust_remote_code=True to load features containing Image type
            hf_dataset = load_dataset("tianhao2k/MM-HELIX", split="test", token=os.environ.get('HF_TOKEN'))

            # 2. Convert to pandas DataFrame
            df = hf_dataset.to_pandas()

            # 3. Save images and update DataFrame
            print(f"Saving images to local directory: {image_dir}")
            df = self._save_images_and_update_df(df, image_dir)

            # 4. Save updated DataFrame as TSV file
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            # Save DataFrame without 'images' column
            df_to_save = df.drop(columns=['images'], errors='ignore')
            # Rename 'id' column to 'index'
            if 'id' in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={'id': 'index'})
            df_to_save.to_csv(data_path, sep='\t', index=False, encoding='utf-8')
            print(f"Dataset metadata saved to: {data_path}")

        except Exception as e:
            print(f"Error downloading and processing dataset from Hugging Face: {e}")
            raise

    def _save_images_and_update_df(self, df, base_image_dir):
        """Iterate through DataFrame, save images locally, and add 'image_path' column."""
        from PIL import Image as PIL_Image

        new_paths = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Saving images"):
            image_list = row.get('images', [])
            saved_paths_for_row = []

            # Check if image_list is empty
            is_empty = (image_list is None
                        or (isinstance(image_list, list) and len(image_list) == 0)
                        or (isinstance(image_list, np.ndarray) and len(image_list) == 0))

            if is_empty:
                new_paths.append(saved_paths_for_row)
                continue

            category = row.get('category', 'unknown_category')
            target_dir = osp.join(base_image_dir, category)

            for i, img_obj in enumerate(image_list):
                # Get filename
                if hasattr(img_obj, 'filename') and img_obj.filename:
                    filename = osp.basename(img_obj.filename)
                else:
                    filename = f"{row.get('index', index)}_{i}.png"

                local_image_path = osp.join(target_dir, filename)

                # Save image if not exists
                if not osp.exists(local_image_path):
                    os.makedirs(target_dir, exist_ok=True)
                    try:
                        # Get PIL Image
                        if hasattr(img_obj, 'save'):
                            pil_image = img_obj
                        elif isinstance(img_obj, dict) and img_obj.get('bytes'):
                            pil_image = PIL_Image.open(io.BytesIO(img_obj['bytes']))
                        else:
                            pil_image = None

                        if pil_image:
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            pil_image.save(local_image_path)
                    except Exception as e:
                        print(f"Warning: Unable to save image at index {index}: {e}")

                saved_paths_for_row.append(local_image_path)

            new_paths.append(saved_paths_for_row)

        df['image_path'] = new_paths
        return df

    def build_prompt(self, line):
        instruction_following = r'The final answer MUST BE enclosed within \boxed{}'
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        message = []
        if not self.language_only:
            question = line['question'] + '\n' + instruction_following
            message.append(dict(type='text', value=question))

            if (isinstance(line['image_path'], str)
                    and (line['image_path'].endswith('.png')
                         or line['image_path'].endswith('.jpg'))):
                message.append(dict(type='image', value=line['image_path']))
            elif isinstance(line['image_path'], list):
                for img_path in line['image_path']:
                    message.append(dict(type='image', value=img_path))
        else:
            if 'question_text' in line and pd.notna(line['question_text']):
                question = line['question_text'] + '\n' + instruction_following
                message.append(dict(type='text', value=question))
            else:
                print(f"WARNING: index {line.get('index', 'N/A')} has no question_text field.")
                message.append(dict(type='text', value="Hello"))

        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        import json
        import os.path as osp

        import pandas as pd

        try:
            eval_file = eval_file.replace('.xlsx', '.tsv')
            data = load(eval_file, fmt='tsv')
        except Exception:
            eval_file = eval_file.replace('.tsv', '.xlsx')
            data = load(eval_file, fmt='xlsx')

        results_by_category = {}

        def str_to_dict(s):
            try:
                if isinstance(s, dict):
                    return s
                import ast
                return ast.literal_eval(str(s))
            except Exception:
                print(f"Warning: Could not parse dictionary string: {s}")
                return {}

        def parse_answer_from_response(response, category):
            """
            Extract the content after "Answer:" or "Answer："
            Supports English colon and Chinese colon
            """
            if not response:
                return ""
            if category in parser:
                return parser[category](response)
            else:
                return parser['default'](response)

        def get_metric(category):
            if category in metrics:
                return metrics[category]
            else:
                return metrics['unsupported']

        def get_metric_info(row):
            try:
                # First try to parse using str_to_dict
                metric_info = str_to_dict(row['metric_info'])

                # If str_to_dict returns an empty dictionary, try using json.loads
                if not metric_info:
                    try:
                        metric_info = json.loads(row['metric_info'])
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        print(f"Question data: {row['metric_info']}")
                        # If JSON parsing fails, return the default evaluation function
                        return metrics['unsupported']

                return get_metric(metric_info.get('score_function'))
            except Exception as e:
                print(f"Error processing metric_info: {e}")
                print(f"Question data: {row['metric_info']}")
                return metrics['unsupported']

        def get_group_for_category(category):
            for group_name, categories in self.GROUP_LIST.items():
                if category in categories:
                    return group_name
            return 'other'

        total_records = 0
        total_score = 0

        data_judged = data.copy()

        # For graph_isomorphism tasks, we need to track pairs of questions
        graph_isomorphism_pairs = {}
        graph_isomorphism_results = {}

        for index, row in data.iterrows():
            # answer_prediction = row['answer']
            for field in row.index:
                if isinstance(row[field], str):
                    row[field] = row[field].replace('\\n', '\n')\
                        .replace('\\t', '\t')\
                        .replace('\\"', '"')\
                        .replace("\\'", "'")

            answer = row['answer']
            if pd.isna(row['prediction']) or row['prediction'] == '':
                print(f"Warning: index: {index} prediction is empty")
                response = ""
            else:
                try:
                    response = parse_answer_from_response(
                        row['prediction'], 'default'
                    )
                    # If parsing result is empty, use original prediction
                    if not response and row['prediction']:
                        print(f"Parser Warning: index: {index} prediction is empty, keep original prediction")
                        response = row['prediction']
                except Exception as e:
                    print(f"Error parsing answer: {e}")
                    print(f"index: {index}")
                    print(f"prediction type: {type(row['prediction'])}")
                    print(f"prediction length: {len(str(row['prediction'])) if row['prediction'] else 0}")
                    response = row['prediction'] if row['prediction'] else ""

            # Check if image name is in reverse_xxx format
            is_reverse = False
            if 'image_path' in row:
                image_path = row['image_path']
                if isinstance(image_path, list) and len(image_path) > 0:
                    # Check the first image path in the list
                    first_image = image_path[0]
                    is_reverse = 'reverse_' in first_image
                elif isinstance(image_path, str):
                    # Extract filename from full path
                    image_name = os.path.basename(image_path)
                    is_reverse = image_name.startswith('reverse_')

            initial_state = row['initial_state'] if 'initial_state' in row else None
            initial_state = self.str_to_dict(initial_state)

            if not isinstance(initial_state, dict):
                initial_state = row['initial_state'] if 'initial_state' in row else None

            score = get_metric_info(row).evaluate(response, answer, initial_state)

            # Special handling for graph_isomorphism tasks
            category = row['category']
            print(f"index: {index}")
            if category == 'graph_isomorphism':
                # Store individual scores for pair evaluation later
                graph_isomorphism_results[index] = {
                    'score': score,
                    'response': response,
                    'answer': answer,
                    'image_path': row.get('image_path', ''),
                    'is_reverse': is_reverse
                }

                # Use question_text for pairing instead of image path
                question_text = row.get('question_text', '')

                # Group by question_text - if question_text is the same, they belong to the same pair
                if question_text:
                    if question_text not in graph_isomorphism_pairs:
                        graph_isomorphism_pairs[question_text] = []
                    graph_isomorphism_pairs[question_text].append(index)

            # Save results to dictionary
            category = row['category']
            if category not in results_by_category:
                results_by_category[category] = {
                    'total_score': 0,
                    'count': 0,
                    'items': []
                }
            data_judged.at[index, 'judge'] = score
            self.language_only = False
            if not self.language_only:
                item_data = {
                    'index': index,
                    'question': row.get('question', ''),
                    'original_prediction': row['prediction'],
                    'answer': answer,
                    'prediction': response,
                    'score': score,
                }
            else:
                item_data = {
                    'index': index,
                    'question': row.get('question_text', ''),
                    'original_prediction': row['prediction'],
                    'answer': answer,
                    'prediction': response,
                    'score': score,
                }
            results_by_category[category]['items'].append(item_data)
            results_by_category[category]['total_score'] += score
            results_by_category[category]['count'] += 1

            # Accumulate total score
            total_score += score
            total_records += 1

        # Process graph_isomorphism pairs after all individual scores are calculated
        if graph_isomorphism_pairs:
            total_score_adjustment = MMHELIX._process_graph_isomorphism_pairs(
                graph_isomorphism_pairs,
                graph_isomorphism_results,
                data_judged,
                results_by_category
            )
            # Update total score to reflect pair evaluation adjustments
            total_score += total_score_adjustment

        # Save judged file results as xlsx
        data_judged_file = osp.splitext(eval_file)[0] + '_judged.xlsx'
        data_judged.to_excel(data_judged_file, index=False)

        # Calculate average score for each category
        for category in results_by_category:
            if results_by_category[category]['count'] > 0:
                results_by_category[category]['average_score'] = \
                    results_by_category[category]['total_score'] / results_by_category[category]['count']
            else:
                results_by_category[category]['average_score'] = 0

        category_results = []
        for category in results_by_category:
            category_results.append({
                'group': get_group_for_category(category),
                'category': category,
                'items': len(results_by_category[category]['items']),
                'average_score': results_by_category[category]['average_score'],
            })

        # Sort category_results by group
        category_results.sort(key=lambda x: x['group'])

        # Calculate average score for each group (average of task averages)
        group_results = {}
        for category_result in category_results:
            group = category_result['group']
            if group not in group_results:
                group_results[group] = {
                    'task_scores': [],
                    'total_items': 0,
                    'categories': []
                }

            group_results[group]['task_scores'].append(category_result['average_score'])
            group_results[group]['total_items'] += category_result['items']
            group_results[group]['categories'].append(category_result['category'])

        # Calculate average score for each group (average of task average scores)
        group_summary = []
        for group, data in group_results.items():
            # Average of task average scores, not weighted by number of items
            average_score = sum(data['task_scores']) / len(data['task_scores']) if len(data['task_scores']) > 0 else 0
            group_summary.append({
                'group': group,
                'categories_count': len(data['categories']),
                'total_items': data['total_items'],
                'average_score': average_score,
            })

        # Add overall statistics (average of all task averages)
        all_task_scores = [cat_result['average_score'] for cat_result in category_results]
        overall_average = sum(all_task_scores) / len(all_task_scores) if len(all_task_scores) > 0 else 0

        group_summary.append({
            'group': 'overall',
            'categories_count': '',  # Empty value
            'total_items': total_records,
            'average_score': overall_average,
        })

        # Add type field and merge two DataFrames
        # Add type field for category_results
        category_results_with_type = []
        for item in category_results:
            item_with_type = item.copy()
            # Add empty fields for category type to maintain consistency
            item_with_type['categories_count'] = ''
            item_with_type['total_items'] = ''
            category_results_with_type.append(item_with_type)

        # Add type field for group_summary
        group_results_with_type = []
        for item in group_summary:
            item_with_type = item.copy()
            # For overall, keep categories_count empty
            if item['group'] == 'overall':
                item_with_type['categories_count'] = ''
            group_results_with_type.append(item_with_type)

        # Merge two results
        combined_results = category_results_with_type + group_results_with_type

        # Add overall average score (average of task averages)
        _ = {
            'total_score': total_score,
            'total_count': total_records,
            'overall_average': overall_average,
            'category_results': category_results,
            'group_results': group_summary,
            'detailed_results': results_by_category
        }

        # Save merged results as TSV file
        combined_df = pd.DataFrame(combined_results)
        combined_tsv_file = osp.splitext(eval_file)[0] + '_results.tsv'
        combined_df.to_csv(combined_tsv_file, sep='\t', index=False, encoding='utf-8')
        print(f"combined results saved to: {combined_tsv_file}")

        # # Save TSV file with only score columns
        # scores_only_df = combined_df[['average_score']].copy()
        # scores_only_tsv_file = osp.splitext(eval_file)[0] + '_scores_only.tsv'
        # scores_only_df.to_csv(scores_only_tsv_file, sep='\t', index=False, encoding='utf-8')
        # print(f"scores only results saved to: {scores_only_tsv_file}")

        # Save detailed_results as JSON file
        detailed_json_file = osp.splitext(eval_file)[0] + '_detailed_results.json'
        with open(detailed_json_file, 'w', encoding='utf-8') as f:
            json.dump(results_by_category, f, ensure_ascii=False, indent=2)
        print(f"detailed results saved to: {detailed_json_file}")

    @staticmethod
    def _process_graph_isomorphism_pairs(pairs_dict, results_dict, data_judged, results_by_category):
        """
        Process graph_isomorphism pairs: both questions in a pair must be correct
        for the evaluation to count as correct.
        """
        print("\nProcessing graph_isomorphism pairs:")
        print(f"Found {len(pairs_dict)} unique questions: "
              f"{len(list(pairs_dict.keys()))} groups")
        for question_text, indices in pairs_dict.items():
            truncated_q = (question_text[:50] + '...'
                           if len(question_text) > 50 else question_text)
            print(f"  Question (truncated): {truncated_q}: indices {indices}")

        # Track total adjustments for category statistics
        total_score_adjustment = 0

        for question_text, indices in pairs_dict.items():
            if len(indices) == 2:  # We have a pair
                idx1, idx2 = indices
                result1 = results_dict[idx1]
                result2 = results_dict[idx2]
                question_short = (question_text[:50] + "..."
                                  if len(question_text) > 50 else question_text)
                print(f"Graph isomorphism pair '{question_short}': "
                      f"{result1['score']} and {result2['score']}")

                # Both must be correct for pair to be correct
                pair_correct = result1['score'] and result2['score']

                # Update individual scores based on pair result
                old_score1 = result1['score']
                old_score2 = result2['score']
                new_score = 1 if pair_correct else 0

                # Update data_judged
                data_judged.at[idx1, 'judge'] = new_score
                data_judged.at[idx2, 'judge'] = new_score

                # Update results_by_category for graph_isomorphism
                if 'graph_isomorphism' in results_by_category:
                    # Adjust total score for the category
                    score_change = (new_score * 2) - (old_score1 + old_score2)
                    results_by_category['graph_isomorphism']['total_score'] += score_change
                    total_score_adjustment += score_change

                    # Update individual items in results
                    for item in results_by_category['graph_isomorphism']['items']:
                        if item['index'] in [idx1, idx2]:
                            item['score'] = new_score
                            item['pair_evaluation'] = pair_correct

                print(f"Graph isomorphism pair '{question_short}': "
                      f"{old_score1 + old_score2}/2 -> {new_score * 2}/2 "
                      f"(pair_correct: {pair_correct})")

            elif len(indices) == 1:
                # Single question without pair - mark as incorrect per requirement
                idx = indices[0]
                old_score = results_dict[idx]['score']
                new_score = 0
                question_short = question_text[:50] + "..." if len(question_text) > 50 else question_text

                data_judged.at[idx, 'judge'] = new_score

                if 'graph_isomorphism' in results_by_category:
                    score_change = new_score - old_score
                    results_by_category['graph_isomorphism']['total_score'] += score_change
                    total_score_adjustment += score_change

                    for item in results_by_category['graph_isomorphism']['items']:
                        if item['index'] == idx:
                            item['score'] = new_score
                            item['pair_evaluation'] = False

                print(f"Graph isomorphism single question '{question_short}': "
                      f"{old_score} -> {new_score} (no pair found)")

        return total_score_adjustment

    def str_to_dict(input_str):
        if isinstance(input_str, str):
            try:
                import json
                input_str = json.loads(input_str)
            except (json.JSONDecodeError, TypeError):
                try:
                    import ast
                    input_str = ast.literal_eval(input_str)
                except (ValueError, SyntaxError):
                    pass
        return input_str
