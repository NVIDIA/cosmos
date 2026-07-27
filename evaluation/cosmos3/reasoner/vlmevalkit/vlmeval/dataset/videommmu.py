import io
import json
import os
import os.path as osp
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_intermediate_file_path, load, md5,
                         modelscope_flag_set)
from vlmeval.utils import track_progress_rich
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_options(options):
    # Define the option letters based on the number of options
    option_letters = [chr(ord("A") + i) for i in range(len(options))]

    # Check if the options are already appended with letters
    if all(
            option.startswith(f"{letter}.")
            for option, letter in zip(options, option_letters)):
        return "\n".join(options)

    # Otherwise, append option letters
    choices_str = "\n".join([
        f"{option_letter}. {option}"
        for option_letter, option in zip(option_letters, options)
    ])
    return choices_str


def check_is_number(string):
    """
    Check if the given string a number.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L65
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L76
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L100
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbersz
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L122
    """
    if response == "API Error" or response == "":
        return "API Error"

    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            # Common explanation or conclusion phrases
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
            "are ",
            "in total ",
            "total ",
            "identify ",
            "recognize ",
            "calculated as ",
            "counted as ",
            "measured as ",
            "observed as ",
            "concluded as ",
            "found to be ",
            "equals ",
            "determined to be ",
            "number of ",
            "value is ",
            "adds up to ",
            "have ",
            "has ",
        ]

        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            # the shortest response that may contain the answer (tail part of the response)
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(
                            indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(
                                shortest_key_response):
                            shortest_key_response = resp.split(
                                indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [
                        ":",
                        ",",
                        ".",
                        "!",
                        "?",
                        ";",
                        ":",
                        "'",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if response == "API Error" or response == "":
        return "API Error"

    # Step 1: Clean up punctuation from the response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match
    # print(response)

    index_ans = True
    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []

    # Step 2: If no candidates, look for choices with a period after (A. B. C. D.)
    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            # print(f"Found choice with period after: {choice}")
            candidates.append(choice)
            ans_with_period = True
    # Step 2.1: If no candidates, look for choices with a colon after (A: B: C: D:)
    for choice in all_choices:  # e.g., A: B: C: D:
        if f"{choice}:" in response:
            # print(f"Found choice with semicolon after: {choice}")
            candidates.append(choice)
            ans_with_colon = True

    # Step 3: Look for choices with parentheses e.g., (A) (B) (C) (D)
    if len(candidates) == 0:
        for choice in all_choices:  # e.g., (A) (B) (C) (D)
            if f"({choice})" in response:
                # print(f"Found choice with parentheses: {choice}")
                candidates.append(choice)
                ans_with_brack = True

    # Step 4: If no candidates, look for choices with a space after (A B C D)
    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                # print(f"Found choice without parentheses (space after): {choice}")
                candidates.append(choice)

    # Step 5: If no candidates and response has more than 5 tokens, try parsing based on content
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                # print(f"Found answer content match: {ans}")
                candidates.append(index)
                index_ans = False  # It's content answer, not an index

    # Step 6: If still no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = "No Answere Found"
        # print(f"No candidates found.")
    # Step 7: If multiple candidates found, use the one appearing last
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_period:
                for can in candidates:
                    index = response.rfind(f"{can}.")
                    # print(f"Checking position of choice: {can} at {index}")
                    start_indexes.append(index)
            elif ans_with_colon:
                for can in candidates:
                    index = response.rfind(f"{can}:")
                    # print(f"Checking position of choice: {can} at {index}")
                    start_indexes.append(index)
            elif ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    # print(f"Checking position of choice with parentheses: {can} at {index}")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    # print(f"Checking position of choice: {can} at {index}")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                # print(f"Checking position of content match: {can} at {index}")
                start_indexes.append(index)
        # Get the last one (max index)
        pred_index = candidates[np.argmax(start_indexes)]
        # print(f"Multiple candidates, selected based on last occurrence: {pred_index}")
    else:
        # If only one candidate, use it
        pred_index = candidates[0]
        # print(f"Only one candidate found, selected: {pred_index}")

    return pred_index


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def process_results(line):
    pred = line['prediction']
    pred = pred.rpartition('Answer:')[-1].strip()

    question_type = line.get("question_type", "None")
    if question_type == "multiple-choice":
        index2ans, all_choices = get_multi_choice_info(
            json.loads(line["options"]))
        parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    else:
        parsed_pred = parse_open_response(pred)

    return {"id": line["id"], "parsed_pred": parsed_pred}


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
    """
    # print(gold_i)
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred,
                      str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    # print(gold_i)
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def evaluate_mmmu(samples):
    """
    Batch evaluation for multiple choice and open questions.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L219
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        if sample["question_type"] == "multiple-choice":
            correct = eval_multi_choice(gold_i, pred_i)
        elif sample["question_type"] == "perception":
            correct = eval_multi_choice(gold_i, pred_i)
        else:  # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"total": 0, "hit": 0, "acc": 0}
    return judge_dict, {
        "total": len(samples),
        "hit": pred_correct,
        "acc": pred_correct / len(samples) * 100
    }


def aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)

    # Filter out results where parsed_pred is "API Error"
    valid_results = [
        result for result in results if result["parsed_pred"] != "API Error"
    ]

    for result in valid_results:
        subset_to_eval_samples[result["category"]].append(result)

    total = 0
    hit = 0
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_mmmu(sub_eval_samples)
        evaluation_result[subset] = metric_dict
        total += metric_dict['total']
        hit += metric_dict['hit']

    evaluation_result["Overall"] = {
        "total": total,
        'hit': hit,
        "acc": hit / total * 100,
    }
    df = pd.DataFrame(evaluation_result)
    return df


class VideoMMMU(VideoBaseDataset):

    MD5 = None

    PRE_PROMPT = "You should watch and learn the video content. Then apply what you learned to "
    PERCEPTION_AND_COMPREHENSION_PROMPT = "\nPlease ignore the Quiz question in last frame of the video."
    MCQ_PROMPT = "answer the following multi-choice question. The image for this question is at the end of the video.\n"
    OPEN_ENDED_PROMPT = "answer the following open-ended question. The image for this question is at the end of the video.\n"  # noqa: E501

    TYPE = 'VideoMMMU'

    def __init__(self,
                 dataset='VideoMMMU',
                 nframe=0,
                 fps=-1,
                 interleave=False):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset
        self.interleave = interleave

    @classmethod
    def supported_datasets(cls):
        return ['VideoMMMU']

    def prepare_dataset(self,
                        dataset_name='VideoMMMU',
                        repo_id='lmms-lab/VideoMMMU'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if self.MD5 and md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def unzip_hf_zip(pth):
                import zipfile
                base_dir = Path(pth)
                target_dir = base_dir / 'videos'
                target_dir.mkdir(exist_ok=True)
                zip_files = sorted(base_dir.glob('*.zip'))

                if not target_dir.exists():
                    for zip_file in zip_files:
                        with zipfile.ZipFile(str(zip_file), 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Check if the member is a file (not a directory)
                                if not member.endswith(
                                        '/') and not member.startswith('__'):
                                    # Extract the file to the specified directory
                                    source = zip_ref.open(member)
                                    target = target_dir / member
                                    target.parent.mkdir(exist_ok=True)
                                    if not target.exists():
                                        with source, open(target, 'wb'):
                                            target.write(source.read())
                    print(
                        'The video file has been restored and stored from the zip file.'
                    )
                else:
                    print('The video file already exists.')

            def generate_tsv(pth):

                data_file = Path(pth) / f'{dataset_name}.tsv'
                if data_file.exists() and (not self.MD5
                                           or md5(data_file) == self.MD5):
                    return

                sub_dfs = []
                for parquet_file in sorted(Path(pth).glob('**/*.parquet')):
                    df = pd.read_parquet(parquet_file)
                    rows = []
                    for _, row in df.iterrows():
                        videos = Path(pth).glob(f'videos/**/{row["id"]}.mp4')
                        video_pth = next(p for p in videos
                                         if 'question_only' not in str(p))
                        options = json.dumps(row['options'].tolist(),
                                             ensure_ascii=False)
                        row_data = {
                            'id': row['id'],
                            'question': row['question'],
                            'answer': row['answer'],
                            'options': options,
                            'question_type': row['question_type'],
                            'video': str(video_pth.relative_to(pth)),
                        }
                        if 'image' in df.columns:
                            image = Image.open(
                                io.BytesIO(row['image']['bytes']))
                            image_path = Path(
                                pth) / 'images' / row['image']['path']
                            image_path.parent.mkdir(exist_ok=True)
                            if not image_path.exists():
                                image.save(str(image_path))
                            row_data[
                                'image'] = f"images/{row['image']['path']}"
                        rows.append(row_data)
                    new_df = pd.DataFrame(rows)
                    new_df['category'] = parquet_file.parent.name
                    sub_dfs.append(new_df)
                cols = [
                    'id', 'category', 'question', 'options', 'answer',
                    'question_type', 'video', 'image'
                ]
                df = pd.concat(
                    sub_dfs,
                    ignore_index=True).reindex(columns=cols).reset_index()
                df.to_csv(data_file, sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id,
                                                 repo_type='dataset')
            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, id_, video_pth, video_llm=False):

        vid_path = osp.join(self.data_root, video_pth)
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(id_)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(id_, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + 'f.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(
            line['id'], line['video'], video_llm)

        message = []
        if video_llm:
            message.append(
                dict(type='video',
                     value=osp.join(self.data_root, line['video'])))
        else:
            message.extend(dict(type='image', value=im) for im in frames)

        if line['category'] == 'Adaptation':
            text_prompt = self.doc_to_text_adaptation(line)
            if self.interleave:
                pre, _, post = text_prompt.partition('<image 1>')
                post += '\nAdd `Answer: {Your final answer}` at the end of your reply.'
                if pre.strip():
                    message.append(dict(type='text', value=pre))
                message.append(
                    dict(type='image',
                         value=osp.join(self.data_root, line['image'])))
                if post.strip():
                    message.append(dict(type='text', value=post))
            else:
                message.append(
                    dict(type='image',
                         value=osp.join(self.data_root, line['image'])))
                text_prompt += '\nAdd `Answer: {Your final answer}` at the end of your reply.'
                message.append(dict(type='text', value=text_prompt))
        else:
            text_prompt = self.doc_to_text_perception_comprehension(line)
            text_prompt += '\nAdd `Answer: {Your final answer}` at the end of your reply.'
            message.append(dict(type='text', value=text_prompt))

        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        storage = get_intermediate_file_path(eval_file, '_score')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(line, ) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    process_results,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['id'] == v['id'] and ans[k][
                        'parsed_pred'] == v['parsed_pred']

            data['parsed_pred'] = [
                ans[idx]['parsed_pred'] for idx in data['index']
            ]
            dump(data, storage)

        score = aggregate_results([row for _, row in load(storage).iterrows()])
        score_pth = get_intermediate_file_path(storage, '_score', 'csv')
        dump(score, score_pth)
        return score

    def doc_to_text_perception_comprehension(self, doc):
        post_prompt = self.PERCEPTION_AND_COMPREHENSION_PROMPT
        question = doc["question"]
        parsed_options = parse_options(json.loads(doc["options"]))
        question += "\n" + parsed_options

        return f"{question}{post_prompt}"

    def doc_to_text_adaptation(self, doc):
        pre_prompt = self.PRE_PROMPT
        question = doc["question"]

        if doc["question_type"] == "multiple-choice":
            pre_prompt += self.MCQ_PROMPT
            parsed_options = parse_options(json.loads(doc["options"]))
            question += "\n" + parsed_options
        else:
            pre_prompt += self.OPEN_ENDED_PROMPT

        return f"{pre_prompt}{question}"
