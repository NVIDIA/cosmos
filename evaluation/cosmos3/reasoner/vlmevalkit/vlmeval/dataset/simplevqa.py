import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils.simplevqa import SimpleVQAEval
from vlmeval.smp import file, misc
from vlmeval.smp.file import get_intermediate_file_path

NUM_WORKERS = 16

COMPARE_ANSWER_PROMPT = """
            请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。
            首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
            ## 以下是【正确】的答复示例：
            ```
            问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
            标准答案：玛丽亚·奥巴马和萨莎·奥巴马
            模型预测1：Malia Obama and Sasha Obama
            模型预测2：玛丽亚和萨沙
            模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
            模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
            ```
            这些答复均为【正确】，因为：
            - 完整地包含了标准答案中的重要信息。
            - 不包含任何与标准答案矛盾的信息。
            - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
            - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

            ## 以下是【错误】的答复示例：
            ```
            问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
            标准答案：玛丽亚·奥巴马和萨莎·奥巴马
            模型预测1：玛丽亚
            模型预测2：玛丽亚、萨莎和苏珊
            模型预测3：巴拉克·奥巴马没有孩子
            模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
            模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
            模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
            ```
            这些答复均为【错误】，因为：
            - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

            ## 以下是【未尝试】的答复示例：
            ```
            问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
            标准答案：玛丽亚·奥巴马和萨莎·奥巴马
            模型预测1：我不知道。
            模型预测2：我需要更多关于您所指奥巴马的上下文。
            模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
            模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
            模型预测5：我无法识别图中的人物。
            模型预测6：N/A。
            ```
            这些答复均为【未尝试】，因为：
            - 没有包含标准答案中的重要信息。
            - 回复中没有与标准答案矛盾的陈述。
            只返回字母”A”、”B”或”C”，无须添加其他文本。

            另外注意以下几点：
            - 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”：
            - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
            - 预测答案“3520”和“3600”均为【错误】。
            - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
            - 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
            - 例如，考虑问题“菱镁矿的主要化学成分是什么？”,标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
            - 如果从问题中明显可以推断出预测答案省略的信息，那么算作【正确】。
            - 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。
            - 如果能明显看出名字翻译版本不同但是是同一个人也认为【正确】。
            - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均【正确】。
            - 预测答案和标准答案对应的是同一事物，但是称呼不同，如“天坛”和“祈年殿”，那么算作【正确】

            ## 下面是一个新的问题示例。对每一个预测答案，请只回复"正确"、"错误"、"未尝试"之一，不要道歉或纠正自己的错误，只需要评估该回答。
            ```
            问题: {question}
            正确答案: {answer}
            预测答案: {candidates}
            ```

            请严格按照以下格式回复，以JSON格式返回一个字典,而且字典第一层key不要替换为具体答案。不要返回其他任何内容。
            [回复格式]:
            ```json
            {{
                "预测答案0": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
                "预测答案1": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
                ...
            }}
            ```
            """.strip()


class SimpleVQA(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "SimpleVQA": "https://opencompass.openxlab.space/utils/VLMEval/SimpleVQA.tsv",
    }
    DATASET_MD5 = {
        "SimpleVQA": "e3ce3c11df59a2a15d37489a8b245a87",
    }

    def build_prompt(self, line: Union[int, pd.Series], qa_type: str = 'Direct') -> List[Dict[str, str]]:
        """
        Build a prompt for the model from a data line.

        Args:
            line: Either an index into the dataset or a pandas Series
            qa_type: Choose from ['Direct', 'CoT', 'PoT']

        Returns:
            List of message dictionaries containing the image and question
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = misc.toliststr(line["image"])
        else:
            tgt_path = self.dump_image(line)

        # load data line elements
        question = line['question']
        assert len(tgt_path) == 1

        # form messages
        msgs = []
        msgs = [dict(type='image', value=tgt_path[0])]
        msgs.append(dict(type='text', value=question))

        return msgs

    def get_scores(self, result_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        data = file.load(result_file)
        model_keys = ['model_response']
        fout = open(str(Path(result_file).parent) + '/gpt_eval.json', 'w', encoding='utf-8')

        gpt4_key = os.environ.get('OPENAI_API_KEY', None)
        base_url = os.environ.get('OPENAI_API_BASE', None)
        nproc = judge_kwargs.get('nproc', 16)

        client = OpenAI(
            api_key=gpt4_key,
            base_url=base_url
        )

        def process_one(idx):
            question = data['question'][idx]
            answer = data['answer'][idx]
            model_response = data['prediction'][idx]
            line = {'question': question, 'answer': answer, 'model_response': model_response}
            res_json = line.copy()
            candidates = "\n[预测答案0]：{}".format(model_response)
            prompt = COMPARE_ANSWER_PROMPT.format(question=question, answer=answer, candidates=candidates)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1
                )
                res = response.choices[0].message.content
                res = res.replace("```json", "").replace("```python", "").replace("```", "").strip()
                if res[-1] != "}":
                    res += "}"
                res = json.loads(res)
            except Exception:
                res = {"预测答案0": {"conclusion": "答案解析失败"}}
            new_res = {}
            for i, key in enumerate(res.keys()):
                if i >= len(model_keys):
                    break
                new_res[model_keys[i]] = res[key]
            res_json["judge_res"] = new_res
            return idx, res_json

        results = [None] * len(data)
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            futures = {executor.submit(process_one, i): i for i in range(len(data))}

            for future in tqdm(as_completed(futures), total=len(data)):
                idx, res_json = future.result()
                results[idx] = res_json

        json.dump(results, fout, ensure_ascii=False, indent=4)
        fout.close()
        scores = SimpleVQAEval(results)
        return pd.DataFrame(list(scores.items()))

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        """
        Evaluate model predictions on the ChartQAPro dataset.

        Args:
            eval_file: Path to the file containing model predictions
            **judge_kwargs: Additional arguments for the judge model

        Returns:
            DataFrame with evaluation scores by category
        """
        score = self.get_scores(eval_file, **judge_kwargs)
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        file.dump(score, score_file)
        return score
