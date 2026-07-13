# flake8: noqa
import copy
import os
import os.path as osp
import re
from collections import defaultdict

import pandas as pd

from vlmeval.smp import (decode_base64_to_image_file, dump, get_intermediate_file_path, listinstr,
                         load, read_ok, toliststr)
from ..utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

prompt_dict = {}
prompt_dict['LiveMMBench_Creation'] = {
    # Subjective Judge [GPT-4o reference]
    'subjective':
    """
Please act as an impartial judge and evaluate the quality of two responses provided by AI assistants to the user prompt.

Your task is to carefully assess two responses based on provided instructions and evaluation criteria. After evaluating both responses, determine which response features better quality and better meets the criteria. If both responses are similar or nearly identical in quality, you should indicate a tie. Avoid position bias toward the first or second response.

Suggested Steps for Evaluation:
1. Review both responses independently and then carefully compare their strengths and weaknesses. A good response should feature good language quality, follow the user instruction and meet as many criteria as possible.
2. After completing the first evaluation, swap the positions of response A and B and repeat Step 1 and get the 2nd evaluation outcome. This helps to mitigate the potential position bias.
3. After completing both evaluations (in the original and reversed order), combine your analysis and provide a final conclusion based on the overall assessment. If both responses are relatively similar, or the differences are minimal and hard to distinguish, your conclusion should indicate a tie ([[A=B]]).

Your **conclusion** should be one of the following options (A, B are of the original order):
1. [[A>>B]]: Response A is clearly better than Response B.
2. [[A>B]]: Response A is slightly better than Response B.
3. [[A=B]]: Response A is nearly identical to Response B.
4. [[B>A]]: Response B is slightly better than Response A.
5. [[B>>A]]: Response B is clearly better than Response A.

User Instruction:\n[INSTRUCTIONS]\n{instructions}\n[END INSTRUCTIONS]\n\n
Repsonse A:\n[RESPONSE A]\n{reference_answer_by_gpt4o}\n[END RESPONSE A]\n\n
Response B:\n[RESPONSE B]\n{prediction}\n[END RESPONSE B]\n\n
Evaluation Criteria:\n[CRITERIA]\n{criteria}\n[END CRITERIA]\n\n

Your output should include:
1. Conclusion: Your final conclusion based on the overall assessment.
2. Reasoning: Your reasoning process and analysis of the two responses.

Your output should follow the following format (CONCLUSION should be one of the five options: A>>B, A>B, A=B, B>A, B>>A):

Final Conclusion: [[CONCLUSION]]
Reasoning Process: [REASONING]\n
""",  # noqa: E501

    # Criteria Alignment w/o GT
    'objective_without_gt':
    """
Please act as an impartial judge and evaluate the **Criteria Alignment** of the two responses provided by AI assistants to the user prompt. The responses were generated based on the provided instructions and visual input from images.

Suggested Steps for Evaluation:
1. Evaluate **Criteria Alignment** of both responses based on the criteria.
    • If a criterion consist of **X aspects**, each aspect is worth **10 / X points**.
    • For each aspect, there may be multiple sub-criteria. If there are **Y sub-criteria for the aspect**, each sub-criterion worths **10 / (X * Y) points**.
2. Assign a total score out of 10 for each response.

User Instruction:\n[INSTRUCTIONS]\n{instructions}\n[END INSTRUCTIONS]\n\n
Repsonse A:\n[RESPONSE A]\n{reference_answer_by_gpt4o}\n[END RESPONSE A]\n\n
Response B:\n[RESPONSE B]\n{prediction}\n[END RESPONSE B]\n\n
Criteria:\n[CRITERIA]\n{criteria}\n[END CRITERIA]\n\n

Your output should evaluate alignment scores of each response and end with a conclusion in the following format (The full score is 10. X, Y are alignment scores for Response A and B):

Response A Alignment Score: X/10
Response B Alignment Score: Y/10\n
""",  # noqa: E501

    # Criteria Alignment w. GT
    'objective_with_gt':
    """
Please act as an impartial judge and evaluate the **Criteria Alignment** of the two responses provided by AI assistants to the user prompt. The responses were generated based on the provided instructions and visual input from images. There is also a ground truth corresponding to the instructions provided for reference.
Take this context into account when making your judgment.

Steps for Evaluation:
1. Evaluate **Criteria Alignment** of both responses based on the criteria and the ground truth.
    • If a criterion consist of **X aspects**, each aspect is worth **10 / X points**.
    • For each aspect, there may be multiple sub-criteria. If there are **Y sub-criteria for the aspect**, each sub-criterion worths **10 / (X * Y) points**.
2. Assign a total score out of 10 for each response.

User Instruction:\n[INSTRUCTIONS]\n{instructions}\n[END INSTRUCTIONS]\n\n
Ground Truth:\n[GROUND TRUTH]\n{groundtruth}\n[END GROUND TRUTH]\n\n
Repsonse A:\n[RESPONSE A]\n{reference_answer_by_gpt4o}\n[END RESPONSE A]\n\n
Response B:\n[RESPONSE B]\n{prediction}\n[END RESPONSE B]\n\n
Criteria:\n[CRITERIA]\n{criteria}\n[END CRITERIA]\n\n

Your output should evaluate alignment scores of each response and end with a conclusion in the following format (The full score is 10. X, Y are alignment scores for Response A and B):

Response A Alignment Score: X/10
Response B Alignment Score: Y/10\n
""",  # noqa: E501
}

prompt_dict['Creation_MMBench'] = {
    # Subjective Judge [GPT-4o reference, with image]
    'subjective':
    """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt below, considering both the provided criteria and the image.

Your task is to carefully assess each response based on how well it meets the evaluation criteria, incorporating the visual context from the image. The criteria should be the primary basis for your judgment, with the image serving to complement and inform your analysis.

Steps for Evaluation:
	1.	Review Both Responses Independently:
        Carefully analyze Assistant A’s and Assistant B’s responses with the criteria and the image. Do not assume any response is better just because it is listed first. Each response should be independently assessed based on the criteria and aided by images to help understand the context.

	2.	Compare the Strengths and Weaknesses:
        After evaluating each response independently, compare the two. Consider both the quality of the content and how closely it aligns with the criteria and image. Identify the strengths and weaknesses of each response, and highlight the key differences.

	3.	Ensure Fairness:
        To avoid positional bias, swap the positions of Assistant A and Assistant B after the first evaluation (i.e., make Assistant A become Assistant B and vice versa) and repeat the analysis and comparison. This ensures that each response is evaluated impartially under the same criteria.

	4.	Provide a Conclusion Based on Both Evaluations:
        After completing both evaluations (original and swapped positions), combine your analysis to provide a final verdict. If the responses are similar, with only minimal differences, your judgment should reflect that and indicate a tie.

Possible Verdict Options:

• If Assistant A is clearly better in both evaluations: [[A>>B]]
• If Assistant A is slightly better in both evaluations: [[A>B]]
• If both responses are nearly identical, showing minimal differences and no clear advantage: [[A=B]]
• If Assistant B is slightly better in both evaluations: [[B>A]]
• If Assistant B is clearly better in both evaluations: [[B>>A]]

Instructions to the AI Assistants:

[INSTRUCTIONS]
{instructions}
[END INSTRUCTIONS]

Assistant A Response:

[ASSISTANT A]
{reference_answer_by_gpt4o}
[END ASSISTANT A]

Evaluation Criteria:

[CRITERIA]
{criteria}
[END CRITERIA]

Assistant B Response:

[ASSISTANT B]
{prediction}
[END ASSISTANT B]

Output Format:

Your output should include:
	1.	Evaluation of Assistant A’s Response: Provide a detailed qualitative evaluation, focusing on how well Assistant A’s response aligns with the criteria and the image.
	2.	Evaluation of Assistant B’s Response: Provide a detailed qualitative evaluation, focusing on how well Assistant B’s response aligns with the criteria and the image.
	3.	Final Verdict: After considering both evaluations, select one of the following verdicts and justify it based on your analysis:

Your output format should end like this:
Assistant A Evaluation: [qualitative comment]
Assistant B Evaluation: [qualitative comment]
Final Verdict is: [[VERDICT]]
""",  # noqa: E501

    ##### For Visual Factuality
    'objective_without_gt':
    """
Please act as an impartial judge and evaluate the **Visual Factuality** of the responses provided by two AI assistants to the user prompt displayed below.

The responses were generated based on the provided instructions and visual input from images. Take this context into account when making your judgment.

Steps for Evaluation:
1. Evaluate visual factuality for both responses based on the visual factuality criteria.
    • If the visual factuality criteria consist of **X aspects**, each aspect is worth **10/X points**.
    • For each aspect, there may be multiple small criteria. If there are **Y small criteria in one aspect**, each small criterion is worth **10/X/Y points**.
2. Assign a total score out of 10 for each response.

Instructions to the AI assistants:
[INSTRUCTIONS]
{instructions}
[END INSTRUCTIONS]

Assistant A response:
[ASSISTANT A]
{reference_answer_by_gpt4o}
[END ASSISTANT A]

Visual Factuality Criteria:
[VISUAL FACTUALITY CRITERIA]
{criteria}
[END CRITERIA]

Assistant B response:
[ASSISTANT B]
{prediction}
[END ASSISTANT B]

Your output should evaluate visual factuality scores for each assistant and end like this:

Response A Visual Factuality Score: X/10
Response B Visual Factuality Score: Y/10
""",  # noqa: E501
    'objective_with_gt':
    """
Please act as an impartial judge and evaluate the **Visual Factuality** of the responses provided by two AI assistants to the user prompt displayed below.

The responses were generated based on the provided instructions and visual input from images.
There is a provided ground truth for the instructions, but the ground truth was not given to the AI assistants when generating their responses.
Take this context into account when making your judgment.

Steps for Evaluation:
1. Evaluate visual factuality for both responses based on the provided ground truth and visual factuality criteria.
    • If the visual factuality criteria consist of **X aspects**, each aspect is worth **10/X points**.
    • For each aspect, there may be multiple small criteria. If there are **Y small criteria in one aspect**, each small criterion is worth **10/X/Y points**.
2. Assign a total score out of 10 for each response.

Instructions to the AI assistants:
[INSTRUCTIONS]
{instructions}
[END INSTRUCTIONS]

Assistant A response:
[ASSISTANT A]
{reference_answer_by_gpt4o}
[END ASSISTANT A]

Visual Factuality Criteria:
[VISUAL FACTUALITY CRITERIA]
{criteria}
[END CRITERIA]

Assistant B response:
[ASSISTANT B]
{prediction}
[END ASSISTANT B]

Ground truth:
[GROUND TRUTH]
{groundtruth}
[END GROUND TRUTH]

Your output should evaluate visual factuality scores for each assistant and end like this:

Response A Visual Factuality Score: X/10
Response B Visual Factuality Score: Y/10
""",  # noqa: E501
}

creation_mmbench_category_dict = {
    'CATEGORY_Literary_Writing': [
        'story_continue',
        'landscape_to_poem',
        'historical_story_creation',
        'story_novel_creation',
        'prose_writing_scenery',
        'art_inspired_prose',
        'daily_conversation_creation',
        'children_book_illustration_dialogue_creation'
    ],
    'CATEGORY_Common_Functionality_Writing':[
        'ins_simple_daily_copywriter',
        'travel_journal',
        'short_video_scripts_for_social_media',
        'social_media_travel_content',
        'daily_achievement_show_off',
        'scientific_research_simple_promotion',
        'twitter_comment_on_daily_news',
        'personal_event_summaries',
        'daily_affairs_inquiries',
        'business_collaborative_email_writing',
        'daily_emotional_email_writing',
        'letter_of_complaint',
        'daily_invitation_email_writing',
        'holiday_card_writing',
        'letter_of_application',
        'product_usage_experience_review',
        'store_experience_review',
        'public_welfare_activity_participation_initiative'
    ],
    'CATEGORY_Professional_Functionality_Writing': [
        'museum_guide_word_creation',
        'recipe_infer_and_guide',
        'landscape_introduction',
        'drafting_announcements_for_public_spaces',
        'floor_plan_renovation_design',
        'teaching_plan',
        'nutritional_formulation_of_recipe',
        'clothing_match_design',
        'software_engineering_diagram_explanation',
        'event_planning_and_venue_arrangement',
        'ui_design_analysis_and_optimization',
        'attraction_promotional_words',
        'product_marketing_strategy',
        'script_writing_for_product_advertisement_promotional_video',
        'residence_reasoning',
        'scientific_diagram_understanding',
        'pulitzer_prize_judge',
        'architecture_appreciation',
        'company_team_amuse_broadcast'
    ],
    'CATEGORY_Creative_Multimodal_Understanding': [
        'travel_itinerary_planning_and_recommendations',
        'photography_appreciation',
        'meme_explanation',
        'advertisement_explanation',
        'document_understanding',
        'snapshot_analysis'
    ]

}


def is_criteria_valid(criteria):
    import re
    for value in criteria.values():
        if value == '\\' or value == '' or not re.search('[a-zA-Z]', value):
            return False
    return True


key_mapping = {
    "sub_parse_ok": "preference_parse_ok",
    "sub_dist": "preference_dist",
    "win_rate": "win_rate",
    "sub_reward": "reward",
    "obj_parse_ok": "visual_factuality_parse_ok",
    "obj_score": "visual_factuality_score",
    "obj_ref_score": "visual_factuality_ref_score"
}


def rename_keys(data, key_mapping):
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            new_key = key_mapping.get(key, key)
            new_data[new_key] = rename_keys(value, key_mapping)
        return new_data
    elif isinstance(data, list):
        return [rename_keys(item, key_mapping) for item in data]
    else:
        return data


def build_prompt(line, dataset_name):
    try:
        criteria = eval(line['criteria'])
    except Exception:
        criteria = line['criteria']

    if isinstance(criteria, dict):
        new_criteria = {}
        for k in criteria:
            if 'subjective' in k.lower():
                new_criteria['subjective'] = criteria[k]
            else:
                new_criteria['objective'] = criteria[k]
    else:
        assert isinstance(criteria, str)
        new_criteria = {'subjective': criteria}
    criteria = new_criteria
    assert 'subjective' in criteria, 'No subjective criteria found in the criteria dict'

    prompts = {}
    if listinstr(['Creation_MMBench'], dataset_name):
        dataset_name = 'Creation_MMBench'
    prompts['subjective'] = prompt_dict[dataset_name]['subjective'].format(
        instructions=line['question'],
        criteria=criteria['subjective'],
        reference_answer_by_gpt4o=line['reference_answer_by_gpt4o'],
        prediction=line['prediction'])
    if 'objective' in criteria:
        if 'ground_truth' in line and (not pd.isna(
                line['ground_truth'])) and line['ground_truth'] != '':
            prompts['objective'] = prompt_dict[dataset_name]['objective_with_gt'].format(
                instructions=line['question'],
                criteria=criteria['objective'],
                groundtruth=line['ground_truth'],
                reference_answer_by_gpt4o=line['reference_answer_by_gpt4o'],
                prediction=line['prediction'])
        else:
            prompts['objective'] = prompt_dict[dataset_name]['objective_without_gt'].format(
                instructions=line['question'],
                criteria=criteria['objective'],
                reference_answer_by_gpt4o=line['reference_answer_by_gpt4o'],
                prediction=line['prediction'])
    return prompts


def Generate_Creation_MMBench_judge(model, image_list, prompt):
    assert isinstance(prompt, dict)
    response = {}
    for key in prompt.keys():
        if image_list and key == 'subjective':
            input_msg = []
            for img_path in image_list:
                if read_ok(img_path):
                    input_msg.append({'type': 'image', 'value': img_path})
                else:
                    raise ValueError(f"Image not found: {img_path}")
            input_msg.append({'type': 'text', 'value': prompt[key]})
            # print(f'using image {image_list} and text')
            response[key] = model.generate(input_msg)
        else:
            response[key] = model.generate(prompt[key])
    return response


def extract_subjective(inp, dataset_name):
    mapping_dict = {
        'LiveMMBench_Creation': 'FINAL CONCLUSION:',
        'Creation_MMBench': 'FINAL VERDICT IS:'
    }
    cands = {'A>>B', 'A>B', 'A=B', 'B>A', 'B>>A', 'B<<A', 'B<A', 'B=A', 'A<B', 'A<<B'}

    lines = inp.split('\n')
    for line in lines:
        line_upper = line.upper()
        if mapping_dict[dataset_name] in line_upper:

            match = re.search(r'\[\[\s*(.*?)\s*\]\]', line_upper)
            if match:
                rem = match.group(1).replace(' ', '')
                if rem in cands:
                    return rem
    return None


def extract_objective(inp, dataset_name):
    # Response A Alignment Score: X/10
    mapping_dict = {
        'LiveMMBench_Creation': {
            'A': 'RESPONSE A ALIGNMENT SCORE:',
            'B': 'RESPONSE B ALIGNMENT SCORE:'
        },
        'Creation_MMBench': {
            'A': 'RESPONSE A VISUAL FACTUALITY SCORE:',
            'B': 'RESPONSE B VISUAL FACTUALITY SCORE:'
        },
    }
    if pd.isna(inp) or inp is None or inp == '':
        return 'NO_OBJECTIVE'
    lines = inp.split('\n')
    a_score, b_score = None, None
    for line in lines:
        line = line.upper()
        line = re.sub(r"[“”*]", "", line)
        if line.startswith(mapping_dict[dataset_name]['A']):
            rem = line.split(mapping_dict[dataset_name]['A'])[1].strip()
            rem = rem.split('/')[0].strip()
            try:
                a_score = float(rem)
            except Exception:
                continue
        elif line.startswith(mapping_dict[dataset_name]['B']):
            rem = line.split(mapping_dict[dataset_name]['B'])[1].strip()
            rem = rem.split('/')[0].strip()
            try:
                b_score = float(rem)
            except Exception:
                continue
    if (a_score is not None and b_score is not None and (0 <= a_score <= 10)
            and (0 <= b_score <= 10)):
        return f'{a_score}|{b_score}'
    else:
        return None


def Creation_MMBench_extract(judge_response_pkl, org_data, dataset_name):
    data = copy.deepcopy(org_data)
    data['subjective_judge'] = [judge_response_pkl[idx]['subjective'] for idx in data['index']]
    data['objective_judge'] = [
        judge_response_pkl[idx].get('objective', None) for idx in data['index']
    ]
    data['subjective_score'] = [
        extract_subjective(x, dataset_name) for x in data['subjective_judge']
    ]
    data['objective_score'] = [extract_objective(x, dataset_name) for x in data['objective_judge']]
    return data


def get_dimension_rating(score_file_name, rev=False):

    def get_pw_score(text):
        if 'A<<B' in text or 'B>>A' in text:
            return 2
        elif 'A<B' in text or 'B>A' in text:
            return 1
        elif 'A=B' in text or 'B=A' in text:
            return 0
        elif 'A>B' in text or 'B<A' in text:
            return -1
        elif 'A>>B' in text or 'B<<A' in text:
            return -2
        else:
            return None

    score_file = load(score_file_name)
    base_dict = {
        'sub_valid': 0,
        'sub_missing': 0,
        'sub_score': [],
        'obj_valid': 0,
        'obj_missing': 0,
        'obj_ref_score': [],
        'obj_score': []
    }
    return_dict = {'overall': copy.deepcopy(base_dict)}

    for idx, item in score_file.iterrows():
        task_name = item['task_name']
        if task_name not in return_dict.keys():
            return_dict[task_name] = copy.deepcopy(base_dict)

        if not pd.isna(item['subjective_score']):
            for k in ['overall', task_name]:
                return_dict[k]['sub_valid'] += 1
                return_dict[k]['sub_score'].append(get_pw_score(item['subjective_score']))
        else:
            return_dict['overall']['sub_missing'] += 1
            return_dict[task_name]['sub_missing'] += 1

        if item['objective_score'] == 'NO_OBJECTIVE':
            continue
        elif not pd.isna(item['objective_score']):
            score = item['objective_score']
            assert '|' in score
            ref_score, score = [float(x) for x in score.split('|')]
            for k in ['overall', task_name]:
                return_dict[k]['obj_valid'] += 1
                return_dict[k]['obj_score'].append(score)
                return_dict[k]['obj_ref_score'].append(ref_score)
        else:
            return_dict['overall']['obj_missing'] += 1
            return_dict[task_name]['obj_missing'] += 1

    final_res = {}

    for k, v in return_dict.items():
        res = {}
        res['sub_parse_ok'] = v['sub_valid'] / (v['sub_valid'] + v['sub_missing'])
        dist = defaultdict(lambda: 0)
        for x in v['sub_score']:
            dist[x] += 1
        assert len(dist) <= 5 and sum(list(dist.values())) == v['sub_valid']
        if v['sub_valid']:
            res['sub_dist'] = {k: dist[k] / v['sub_valid'] for k in [-2, -1, 0, 1, 2]}
            res['sub_reward'] = (-100 * dist[-2] - 50 * dist[-1] + 50 * dist[1] +
                                 100 * dist[2]) / v['sub_valid']

        if v['obj_valid'] + v['obj_missing']:
            res['obj_parse_ok'] = v['obj_valid'] / (v['obj_valid'] + v['obj_missing'])
            if v['obj_valid']:
                res['obj_score'] = sum(v['obj_score']) / v['obj_valid']
                res['obj_ref_score'] = sum(v['obj_ref_score']) / v['obj_valid']
        final_res[k] = res

    final_res['raw'] = return_dict
    return final_res


def merge_dual(raw, raw_dual, dataset_name):
    final_res = {}
    category_raw = {}
    for k, v in raw.items():
        # merge dual: {'sub_valid': 0, 'sub_missing': 0, 'sub_score': [], 'obj_valid': 0, 'obj_missing': 0, 'obj_ref_score': [], 'obj_score': []}
        dual_v = raw_dual[k]
        v['sub_valid'] += dual_v['sub_valid']
        v['sub_missing'] += dual_v['sub_missing']
        v['sub_score'].extend([-x for x in dual_v['sub_score']])
        v['obj_valid'] += dual_v['obj_valid']
        v['obj_missing'] += dual_v['obj_missing']
        v['obj_score'].extend(dual_v['obj_ref_score'])
        v['obj_ref_score'].extend(dual_v['obj_score'])
        raw[k] = v

        res = {}
        res['sub_parse_ok'] = v['sub_valid'] / (v['sub_valid'] + v['sub_missing'])
        dist = defaultdict(lambda: 0)
        for x in v['sub_score']:
            dist[x] += 1
        assert len(dist) <= 5 and sum(list(dist.values())) == v['sub_valid']
        res['sub_dist'] = {k: dist[k] / v['sub_valid'] for k in [-2, -1, 0, 1, 2]}
        res['win_rate'] = (dist[2] + dist[1]) / v['sub_valid'] * 100
        res['sub_reward'] = (-100 * dist[-2] - 50 * dist[-1] + 50 * dist[1] +
                             100 * dist[2]) / v['sub_valid']

        if v['obj_valid'] + v['obj_missing']:
            res['obj_parse_ok'] = v['obj_valid'] / (v['obj_valid'] + v['obj_missing'])
            if v['obj_valid']:
                res['obj_score'] = sum(v['obj_score']) / v['obj_valid']
                res['obj_ref_score'] = sum(v['obj_ref_score']) / v['obj_valid']
        final_res[k] = res

        if listinstr(['Creation_MMBench'], dataset_name):
            pass_flag = False
            for main_category_name, category_list in creation_mmbench_category_dict.items():
                if k in creation_mmbench_category_dict.keys() or k == 'overall':
                    pass_flag = True
                    break
                if k in category_list:
                    if main_category_name not in category_raw.keys():
                        category_raw[main_category_name] = {
                            'sub_valid': 0,
                            'sub_missing': 0,
                            'sub_score': [],
                            'obj_valid': 0,
                            'obj_missing': 0,
                            'obj_ref_score': [],
                            'obj_score': []
                        }
                    category_raw[main_category_name]['sub_valid'] += v['sub_valid']
                    category_raw[main_category_name]['sub_missing'] += v['sub_missing']
                    category_raw[main_category_name]['sub_score'].extend(v['sub_score'])
                    category_raw[main_category_name]['obj_valid'] += v['obj_valid']
                    category_raw[main_category_name]['obj_missing'] += v['obj_missing']
                    category_raw[main_category_name]['obj_score'].extend(v['obj_score'])
                    category_raw[main_category_name]['obj_ref_score'].extend(v['obj_ref_score'])
                    pass_flag = True
                    break
            if not pass_flag:
                raise Exception(f"Error: {k} not found in type_dict")

    for k, v in category_raw.items():
        res = {}
        res['sub_parse_ok'] = v['sub_valid'] / (v['sub_valid'] + v['sub_missing'])
        dist = defaultdict(lambda: 0)
        for x in v['sub_score']:
            dist[x] += 1
        assert len(dist) <= 5 and sum(list(dist.values())) == v['sub_valid']
        res['sub_dist'] = {k: dist[k] / v['sub_valid'] for k in [-2, -1, 0, 1, 2]}
        res['win_rate'] = (dist[2] + dist[1]) / v['sub_valid'] * 100
        res['sub_reward'] = (-100 * dist[-2] - 50 * dist[-1] + 50 * dist[1] +
                             100 * dist[2]) / v['sub_valid']

        if v['obj_valid'] + v['obj_missing']:
            res['obj_parse_ok'] = v['obj_valid'] / (v['obj_valid'] + v['obj_missing'])
            if v['obj_valid']:
                res['obj_score'] = sum(v['obj_score']) / v['obj_valid']
                res['obj_ref_score'] = sum(v['obj_ref_score']) / v['obj_valid']
        final_res[k] = res

    final_res['raw'] = raw
    final_res['category_raw'] = category_raw
    if listinstr(['Creation_MMBench'], dataset_name):
        final_res = rename_keys(final_res, key_mapping)
    return final_res


class CreationMMBenchDataset(ImageBaseDataset):

    TYPE = 'CreationVQA'
    DATASET_URL = {
        'LiveMMBench_Creation': '',
        'Creation_MMBench': 'https://opencompass.openxlab.space/utils/VLMEval/Creation_MMBench.tsv'
    }
    DATASET_MD5 = {'Creation_MMBench': '870c0332a9c6a169d0ac9b8574c245fe'}

    # It returns a dictionary
    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                if 'image_path' in line:
                    assert isinstance(line['image_path'],
                                      str) or (isinstance(line['image_path'], list)
                                               and len(line['image_path']) == 1)
                    if isinstance(line['image_path'], list):
                        line['image_path'] = line['image_path'][0]
                    tgt_path = osp.join(self.img_root, line['image_path'])
                else:
                    tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def evaluate(self, eval_file, **judge_kwargs):
        rating_rev = None
        dual_eval = judge_kwargs.pop('dual_eval', True)
        if dual_eval:
            print('Dual Evaluation Strategy is enabled.')
            src = load(eval_file)
            tgt = load(eval_file)
            tgt['reference_answer_by_gpt4o'] = src['prediction']
            tgt['prediction'] = src['reference_answer_by_gpt4o']
            tgt_file_name = get_intermediate_file_path(eval_file, '_rev')
            dump(tgt, tgt_file_name)
            judge_kwargs['dual_eval'] = False
            rating_rev = self.evaluate(tgt_file_name, **judge_kwargs)
        judge_kwargs.pop('dual_eval', None)

        score_file = get_intermediate_file_path(eval_file, '_score')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')

        model = judge_kwargs.pop('model', 'gpt-4o-0806')
        model_name = model.split('/')[-1] if '/' in model else model
        tmp_file = get_intermediate_file_path(eval_file, f'_{model_name}', 'pkl')

        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            data = load(eval_file)
            lt = len(data)
            lines = [data.iloc[i] for i in range(len(data))]
            judge_kwargs['max_tokens'] = 4096

            model = build_judge(model=model, **judge_kwargs)
            assert model.working(), ('CreationMMBench evaluation requires a working OPENAI API\n' +
                                     DEBUG_MESSAGE)

            prompts = [build_prompt(line, self.dataset_name) for line in lines]

            indices = [line['index'] for line in lines]

            if listinstr(['Creation_MMBench'], self.dataset_name):
                no_relative_image_list = [
                    self.dump_image(line) for idx, line in self.data.iterrows()
                ]
                assert len(no_relative_image_list) == len(lines)
                image_list = []
                for subimage_list in no_relative_image_list:
                    sublist = []
                    for image_path in subimage_list:
                        image_path = osp.join(self.img_root, image_path)
                        assert osp.exists(image_path), f"Image not found: {image_path}"
                        sublist.append(image_path)
                    image_list.append(sublist)
            else:
                image_list = [[] for _ in range(len(lines))]
            tups = [(model, image, prompt) for prompt, image in zip(prompts, image_list)]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            ans = {k: v for k, v in ans.items() if model.fail_msg not in str(v)}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    Generate_Creation_MMBench_judge,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            data = Creation_MMBench_extract(ans, data, self.dataset_name)
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)

        if dual_eval:
            raw = rating['raw']
            rev_tgt_file = tgt_file.replace('rating.json', 'rev_rating.json')
            rev_raw = load(rev_tgt_file)['raw']
            merged_rating = merge_dual(raw, rev_raw, self.dataset_name)
            dump(merged_rating, tgt_file.replace('rating.json', 'merged_rating.json'))
            print(f"Rating:\n{rating['overall']}\n\nDual Rating:\n{merged_rating['overall']}")
            return merged_rating['overall']
        else:
            return rating['overall']
