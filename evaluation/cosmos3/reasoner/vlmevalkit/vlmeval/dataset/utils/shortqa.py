# flake8: noqa
from vlmeval.smp import cn_string

EVAL_TMPL = """
You are an AI assistant tasked with evaluating whether a model's response correctly answers a given visual-language question.

You will be provided with:
1. The question (text only)
2. The model's response
3. The ground truth answer

Your task is to determine whether the model's response conveys the same meaning as the ground truth. The response is considered **correct** if:
- It has the same meaning as the ground truth, even if phrased differently.
- It provides additional relevant details without altering the original meaning.

The response is considered **wrong** if:
- It contradicts the ground-truth
- It misses essential information or include additional incorrect information.

Your evaluation should include the following fields:
- **Correctness**: Either `"yes"` (if correct) or `"no"` (if incorrect).
- **Reason**: A brief explanation of your judgment.

{requirement}

Here are some examples:
{examples}

Now please complete the following task:

[Begin Question]{question}[End Question]
[Begin Ground-Truth]{ground_truth}[End Ground-Truth]
[Begin Response]{response}[End Response]
"""

EVAL_TMPL_CN = """
你是一名 AI 助理，负责评估模型的回答是否正确回答了给定的视觉语言问题。

你将被提供以下信息：
1. 问题（仅包含文本）
2. 模型的回答
3. 标准答案（Ground Truth）

你的任务是判断模型的回答是否与标准答案表达相同的含义。若满足以下条件，则认为回答是**正确的**：
- 回答的含义与标准答案相同，即使措辞不同。
- 回答提供了额外的相关细节，但没有改变原本的含义。

若满足以下条件，则认为回答是**错误的**：
- 回答与标准答案相矛盾。
- 回答遗漏了关键信息，或包含额外的错误信息。

你的评估应包含以下字段：
- **正确性（Correctness）**：值应为 `"yes"`（正确）或 `"no"`（错误）。
- **原因（Reason）**：对你的判断进行简要解释。

{requirement}

以下是一些示例：
{examples}

现在，请完成以下任务：
[Begin Question]{question}[End Question]
[Begin Ground-Truth]{ground_truth}[End Ground-Truth]
[Begin Response]{response}[End Response]
"""

en_single_ICEs = [
    {
        'question': 'Please tell me the name of the man in this image, in the format of "[First Name] [Given Name]".',
        'answer': 'Franklin D. Roosevelt',
        'prediction': 'Franklin Roosevelt',
        'correctness': "yes",
        'reason': 'The model response basically align with the ground-truth answer, just omitted the middle name. Thus the response is correct.'
    },
    {
        'question': 'Please tell me the name of the man in this image, in the format of "[First Name] [Given Name]".',
        'answer': 'Usain Bolt',
        'prediction': 'Bolt',
        'correctness': 'no',
        'reason': 'The question asks the model to output the '
    },
    {
        'question': 'Where did the text in this image originate from',
        'answer': 'Ancient Egypt',
        'prediction': 'egypt',
        'correctness': 'yes',
        'reason': 'The model response basically align with the ground-truth answer (egypt vs. Ancient Egypt). Thus the response is correct. '
    },
    {
        'question': 'Name this building',
        'answer': "St. Peter's Basilica Church",
        'prediction': 'st peters basilica',
        'correctness': 'yes',
        'reason': "The model response basically align with the ground-truth answer. Thus the response is correct."
    },
    {
        'question': 'Extract the text on the umbrella in the image',
        'answer': 'keter',
        'prediction': 'ketter',
        'correctness': 'no',
        'reason': 'The question requires the model to exactly extract the text on the umbrella. The model response does not contain the exact text on the umbrella (keter). Thus the response is incorrect. '
    }
]

en_multiple_ICEs = [
    {
        'question': 'Please tell me the name of the man in this image, in the format of "[First Name] [Given Name]".',
        'answer': 'Franklin D. Roosevelt',
        'prediction': 'Franklin Roosevelt',
        'correctness': "yes",
        'reason': 'The model response basically align with the ground-truth answer, just omitted the middle name. Thus the response is correct.'
    },
    {
        'question': 'Please tell me the name of the man in this image, in the format of "[First Name] [Given Name]".',
        'answer': 'Usain Bolt',
        'prediction': 'Bolt',
        'correctness': 'no',
        'reason': 'The question asks the model to output the '
    },
    {
        'question': 'Name all countries besides this lake',
        'answer': "['Jordan', 'Israel', 'Palestine']",
        'prediction': "Israel, Jordan",
        'correctness': 'no',
        'reason': 'The model response does not contain all the countries besides this lake (missing Palestine). Thus the response is incorrect.'
    },
    {
        'question': 'Name this building, as well as the country that the building located in.',
        'answer': "['La Tour Eiffel (or Eiffel Tower)', 'France']",
        'prediction': "Eiffel Tower, France",
        'correctness': 'yes',
        'reason': 'The model response basically align with the ground-truth answer. Thus the response is correct.'
    },
    {
        'question': 'Name this attraction, as well as the country that this attraction located in. ',
        'answer': "['Notre Dame de Paris', 'France']",
        'prediction': "Notre-Dame Cathedral, France",
        'correctness': 'yes',
        'reason': 'The model response basically align with the ground-truth answer. Thus the response is correct.'
    },
    {
        'question': "Who are the top three players in terms of goals scored for these teams' Top Team Scorers?",
        'answer': "['Mohamed Salah','Erling Haaland','Alexander Isak']",
        'prediction': "The top three players in terms of goals scored for these teams' Top Team Scorers are: 1. Mohamed Salah - 19 goals (Liverpool) 2. Erling Haaland - 18 goals (Manchester City) 3. Alexander Isak - 17 goals (Newcastle Utd)",
        'correctness': 'yes',
        'reason': 'The model response align with the ground-truth answer, and include some additional information including the team and number of goals of the player. Thus the response is correct.'
    }
]

cn_single_ICEs = [
    {
        'question': '请直接告诉我图中右侧人物的名字',
        'answer': '姚明',
        'prediction': 'Yao Ming',
        'correctness': 'yes',
        'reason': '回答中用英语提及了人物的名称 Yao Ming，因此回答正确。'
    },
    {
        'question': '图中右起第几位女士的身高最高？',
        'answer': '右起第一位',
        'prediction': '1',
        'correctness': 'yes',
        'reason': '回答中提及了第一位 (依据提问，默认为右起)，因此回答正确。'
    },
    {
        'question': '图中文物的具体名称是什么？',
        'answer': '司母戊鼎',
        'prediction': '鼎',
        'correctness': 'no',
        'reason': '问题询问文物的具体名称，回答仅提及了鼎，但未提及司母戊鼎，因此回答错误。'
    },
    {
        'question': '图中的景点位于哪个省份？',
        'answer': '四川',
        'prediction': '图中的景点是九寨沟，位于中国四川省阿坝藏族羌族自治州。',
        'correctness': 'yes',
        'reason': '回答提及景点位于四川省，与标准答案一致，且更加详尽，因此回答正确'
    }
]

cn_multiple_ICEs = [
    {
        'question': '图中文物的具体名称是什么？',
        'answer': '司母戊鼎',
        'prediction': '鼎',
        'correctness': 'no',
        'reason': '问题询问文物的具体名称，回答仅提及了鼎，但未提及司母戊鼎，因此回答错误。'
    },
    {
        'question': 'JiaQi在2025年与哪些战队交过手？',
        'answer': "['⁠Ninjas in Pyjamas','Invictus Gaming']",
        'prediction': "根据提供的图片信息，JiaQi在2025年与以下战队交过手：\n- TES Challenger\n- ⁠Ninjas in Pyjamas\n- Invictus Gaming",
        'correctness': 'no',
        'reason': "依据问题与标准答案，JiaQi 在 2025 年应仅与 Ninjas in Pyjamas 与 Invictus Gaming 两支战队交过手。模型回复中，TES Challenger 未在标准答案中提及，因此回答错误。"
    },
    {
        'question': '根据图中内容，请列出全部在 2022年7月份到10月份间的开源模型？',
        'answer': "['CodeGeeX','GLM','Flan-T5','NLLB']",
        'prediction': "根据图中内容，2022年7月份到10月份间的开源模型有：Tk-Instruct, Luminous, GLM, NLLB",
        'correctness': 'no',
        'reason': "根据问题与标准答案，2022年7月份到10月份间的开源模型有 CodeGeeX, GLM, Flan-T5, NLLB。模型回复中，Tk-Instruct 与 Luminous 未在标准答案中提及，且遗漏了 cODEGeeX 与 Flan-T5，因此回答错误。"
    },
    {
        'question': '图中的景点是什么，位于哪个城市',
        'answer': "['少林寺', '河南登封市']",
        'prediction': "The scenic spot in the picture is Shaolin Temple, located in Dengfeng City, Henan Province.",
        'correctness': 'yes',
        'reason': "答案中提及了少林寺及河南省登封市，因此回答正确"
    },
    {
        'question': '图中中央的物品是什么，它最流行于中国的南方还是北方？',
        'answer': "['铜火锅', '北方']",
        'prediction': '图中中央的物品是火锅，它最流行于中国的北方。',
        'correctness': "yes",
        "reason": "回答中提及了火锅及北方，因此回答正确。"
    },
    {
        'question': '请直接告诉我图中右侧人物的名字',
        'answer': "['姚明', '易建联']",
        'prediction': 'Yao Ming & Jianlian Yi',
        'correctness': 'yes',
        'reason': '回答中用英语提及了姚明与易建联的名字，与标准答案一致，因此回答正确。'
    },
]


def ICE_builder(ICEs):
    res = ''
    for i, eg in enumerate(ICEs):
        res += f"Example {i + 1}:\n"
        res += "[Begin Question]" + eg['question'] + "[End Question]\n"
        res += "[Begin Ground-Truth]" + eg['answer'] + "[End Ground-Truth]\n"
        res += "[Begin Response]" + eg['prediction'] + "[End Response]\n"
        res += "[Begin Correctness]" + eg['correctness'] + "[End Correctness]\n"
        res += "[Begin Reason]" + eg['reason'] + "[End Reason]\n"
        res += '\n'
    return res


def ShortQA_prompt(line):
    question = line['question']
    is_cn = cn_string(question)
    answer = line['answer']
    answer_type = line.get('answer_type', 'listofstr')
    if answer[0] == '[' and answer[-1] == ']' and answer_type not in ('exactMatch', 'multipleChoice'):
        answer = eval(answer)
    else:
        answer = [answer]

    requirements = {
        'en_multi': "The provided ground-truth is a list. The answer is correct if the model response contains and only contains all contents in the list (no other answer included)",
        'cn_multi': "题目提供的标准答案是一个列表。如果模型回答包含且仅包含列表中的所有内容，则回答正确",
    }

    examples = ''
    if is_cn:
        examples = ICE_builder(cn_single_ICEs if len(answer) == 1 else cn_multiple_ICEs)
    else:
        examples = ICE_builder(en_single_ICEs if len(answer) == 1 else en_multiple_ICEs)

    if len(answer) > 1:
        requirement = requirements['en_multi'] if not is_cn else requirements['cn_multi']
    else:
        requirement = ''
        answer = answer[0]

    tmpl = EVAL_TMPL_CN if is_cn else EVAL_TMPL
    prompt = tmpl.format(
        question=question,
        examples=examples,
        requirement=requirement,
        ground_truth=answer,
        response=line['prediction']
    )
    return prompt
