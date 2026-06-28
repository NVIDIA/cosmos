from collections import defaultdict

import numpy as np
import pandas as pd

from ..smp.log import get_logger

logger = get_logger(__name__)

rule_dict = {
    'llava_bench_conv': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_detail': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_complex': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'}  # noqa: E501
}

rule_dict_ko = {
    'llava_bench_conv': {'role': '어시스턴트', 'prompt': '두 인공지능 어시스턴트의 성능을 [질문]에 대한 응답에 기반하여 평가하세요. 해당 [질문]은 특정 이미지를 보고 생성되었습니다. `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력`을 기준으로 응답을 평가하세요. 각각의 어시스턴트에게 1에서 10까지의 전반적인 점수를 부여하며, 높은 점수일수록 더 나은 전반적인 성능을 나타냅니다.\n\n# 단계\n1. 제공된 이미지 [설명]을 검토하세요.\n2. 각 어시스턴트의 응답을 다음 기준으로 분석하세요:\n   - `유용성`: 응답이 사용자의 질문을 얼마나 잘 해결하는가?\n   - `관련성`: 응답이 사용자의 질문에 얼마나 적절한가?\n   - `정확성`: 응답에서 제공한 정보가 얼마나 정확한가?\n   - `세부 수준`: 응답이 과하지 않게 충분히 자세한가?\n   - `한국어 생성능력`: 생성된 한국어 문장이 자연스럽고 문법적으로 올바른가?\n3. 분석에 기반하여 각 어시스턴트에게 1에서 10까지의 점수를 부여하세요.\n4. 두 점수를 공백으로 구분하여 한 줄로 제공하세요.\n5. 점수에 대한 이유를 강조하면서 포괄적인 평가를 제공하고, 편견을 피하며 응답의 순서가 판단에 영향을 미치지 않도록 하세요.\n\n# 출력 형식\n- 첫 번째 줄: `어시스턴트1_점수 어시스턴트2_점수` (예: `8 9`)\n- 두 번째 줄: `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력` 기준으로 점수를 설명하는 자세한 문단을 제공합니다.\n\n# 주의사항\n- 평가 시 잠재적 편견을 방지하여 객관성을 확보하세요.\n- 분석과 설명에서 일관성과 명확성을 유지하세요.'},  # noqa: E501
    'llava_bench_detail': {'role': '어시스턴트', 'prompt': '두 인공지능 어시스턴트의 성능을 [질문]에 대한 응답에 기반하여 평가하세요. 해당 [질문]은 특정 이미지를 보고 생성되었습니다. `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력`을 기준으로 응답을 평가하세요. 각각의 어시스턴트에게 1에서 10까지의 전반적인 점수를 부여하며, 높은 점수일수록 더 나은 전반적인 성능을 나타냅니다.\n\n# 단계\n1. 제공된 이미지 [설명]을 검토하세요.\n2. 각 어시스턴트의 응답을 다음 기준으로 분석하세요:\n   - `유용성`: 응답이 사용자의 질문을 얼마나 잘 해결하는가?\n   - `관련성`: 응답이 사용자의 질문에 얼마나 적절한가?\n   - `정확성`: 응답에서 제공한 정보가 얼마나 정확한가?\n   - `세부 수준`: 응답이 과하지 않게 충분히 자세한가?\n   - `한국어 생성능력`: 생성된 한국어 문장이 자연스럽고 문법적으로 올바른가?\n3. 분석에 기반하여 각 어시스턴트에게 1에서 10까지의 점수를 부여하세요.\n4. 두 점수를 공백으로 구분하여 한 줄로 제공하세요.\n5. 점수에 대한 이유를 강조하면서 포괄적인 평가를 제공하고, 편견을 피하며 응답의 순서가 판단에 영향을 미치지 않도록 하세요.\n\n# 출력 형식\n- 첫 번째 줄: `어시스턴트1_점수 어시스턴트2_점수` (예: `8 9`)\n- 두 번째 줄: `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력` 기준으로 점수를 설명하는 자세한 문단을 제공합니다.\n\n# 주의사항\n- 평가 시 잠재적 편견을 방지하여 객관성을 확보하세요.\n- 분석과 설명에서 일관성과 명확성을 유지하세요.'},  # noqa: E501
    'llava_bench_complex': {'role': '어시스턴트', 'prompt': '두 인공지능 어시스턴트의 성능을 [질문]에 대한 응답에 기반하여 평가하세요. 해당 [질문]은 특정 이미지를 보고 생성되었습니다. `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력`을 기준으로 응답을 평가하세요. 각각의 어시스턴트에게 1에서 10까지의 전반적인 점수를 부여하며, 높은 점수일수록 더 나은 전반적인 성능을 나타냅니다.\n\n# 단계\n1. 제공된 이미지 [설명]을 검토하세요.\n2. 각 어시스턴트의 응답을 다음 기준으로 분석하세요:\n   - `유용성`: 응답이 사용자의 질문을 얼마나 잘 해결하는가?\n   - `관련성`: 응답이 사용자의 질문에 얼마나 적절한가?\n   - `정확성`: 응답에서 제공한 정보가 얼마나 정확한가?\n   - `세부 수준`: 응답이 과하지 않게 충분히 자세한가?\n   - `한국어 생성능력`: 생성된 한국어 문장이 자연스럽고 문법적으로 올바른가?\n3. 분석에 기반하여 각 어시스턴트에게 1에서 10까지의 점수를 부여하세요.\n4. 두 점수를 공백으로 구분하여 한 줄로 제공하세요.\n5. 점수에 대한 이유를 강조하면서 포괄적인 평가를 제공하고, 편견을 피하며 응답의 순서가 판단에 영향을 미치지 않도록 하세요.\n\n# 출력 형식\n- 첫 번째 줄: `어시스턴트1_점수 어시스턴트2_점수` (예: `8 9`)\n- 두 번째 줄: `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력` 기준으로 점수를 설명하는 자세한 문단을 제공합니다.\n\n# 주의사항\n- 평가 시 잠재적 편견을 방지하여 객관성을 확보하세요.\n- 분석과 설명에서 일관성과 명확성을 유지하세요.'}  # noqa: E501
}


def get_eval(judge, content):
    return judge.generate(content)


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            logger.error('error', review)
            return [-1, -1]
    except Exception as e:
        logger.error(e, 'error', review)
        return [-1, -1]


def build_prompt(line):
    cap_str = line['caption']
    question = line['question']
    ans1 = line['gpt4_ans']
    ans2 = line['prediction']
    category = 'llava_bench_' + line['category']
    rule = rule_dict[category]
    role, prompt = rule['role'], rule['prompt']

    content = (f'[Context]\n{cap_str}\n\n'
               f'[Question]\n{question}\n\n'
               f'[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n'
               f'[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n'
               f'[System]\n{prompt}\n\n')
    return content


def build_prompt_ko(line):
    cap_str = line['caption']
    question = line['question']
    ans1 = line['gpt4_ans']
    ans2 = line['prediction']
    category = 'llava_bench_' + line['category']
    rule = rule_dict_ko[category]
    role, prompt = rule['role'], rule['prompt']

    content = (f'[설명]\n{cap_str}\n\n'
               f'[질문]\n{question}\n\n'
               f'[{role} 1]\n{ans1}\n\n[{role} 1 끝]\n\n'
               f'[{role} 2]\n{ans2}\n\n[{role} 2 끝]\n\n'
               f'[System]\n{prompt}\n\n')
    return content


def LLaVABench_atomeval(model, prompt):
    review = get_eval(model, prompt)
    scores = parse_score(review)
    return scores


def LLaVABench_score(data):
    cates = ['overall'] + list(set(data['category']))
    ret = defaultdict(list)

    for c in cates:
        ret['split'].append(c)
        sub = data[data['category'] == c] if c != 'overall' else data
        ret['Relative Score (main)'].append(np.mean(sub['score']) / np.mean(sub['gpt4_score']) * 100)
        ret['VLM Score'].append(np.mean(sub['score']) * 10)
        ret['GPT4 Score'].append(np.mean(sub['gpt4_score']) * 10)
    return pd.DataFrame(ret)
