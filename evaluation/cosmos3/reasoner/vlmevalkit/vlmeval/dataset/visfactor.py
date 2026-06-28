import re

from vlmeval.smp import d2df, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset


class VisFactor(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'VisFactor': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor.tsv',
        'VisFactor_CoT': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor.tsv',
        'VisFactor_GE': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor_GE.tsv',
        'VisFactor_GE_CoT': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor_GE.tsv',
        'VisFactor_GN': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor_GN.tsv',
        'VisFactor_GN_CoT': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor_GN.tsv',
        'VisFactor_GH': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor_GH.tsv',
        'VisFactor_GH_CoT': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor_GH.tsv',
    }

    DATASET_MD5 = {
        'VisFactor': 'a069bcd8eb529e1d8c66e4cd7e279d13',
        'VisFactor_CoT': 'a069bcd8eb529e1d8c66e4cd7e279d13',
        'VisFactor_GE': '7e1377b46faff392409a7d4e56688dba',
        'VisFactor_GE_CoT': '7e1377b46faff392409a7d4e56688dba',
        'VisFactor_GN': '34bec098b47f87a8a6239a68f19b9ec8',
        'VisFactor_GN_CoT': '34bec098b47f87a8a6239a68f19b9ec8',
        'VisFactor_GH': '03c902ea44da8469814a8c1933baa923',
        'VisFactor_GH_CoT': '03c902ea44da8469814a8c1933baa923',
    }

    def replace_additional_tags(self, text, additional):
        def replacer(match):
            index = int(match.group(1))
            if 0 <= index < len(additional):
                return additional[index]
            else:
                return match.group(0)
        return re.sub(r"<ADDITIONAL_(\d+)>", replacer, text)

    def split_image_tags(self, text):
        parts = re.split(r'(<IMAGE_\d+>)', text)
        return [part for part in parts if part != '']

    def extract_last_json_answer(self, s):
        pattern = r'\{\s*"answer"\s*:\s*(.+?)\s*\}'
        matches = list(re.finditer(pattern, s))
        if not matches:
            return ''

        raw_value = matches[-1].group(1).strip()

        if (raw_value.startswith('"') and raw_value.endswith('"')) or \
           (raw_value.startswith("'") and raw_value.endswith("'")):
            raw_value = raw_value[1:-1]
        return raw_value

    def extract_last_numbers(self, s):
        return [num for num in re.findall(r'\d+', s)]

    def extract_last_uppercase_letter(self, s):
        for char in reversed(s):
            if char.isupper():
                return char
        return None

    def build_prompt(self, line):
        msgs = line['question'].replace('<br>', '\n')
        image_paths = self.dump_image(line)

        if str(line['additional']) != 'nan':
            additional = str(line['additional']).replace('<br>', '\n').split(';')
            msgs = self.replace_additional_tags(msgs, additional)

        if 'cot' in self.dataset_name.lower():
            msgs = msgs.replace(
                'Output: Respond',
                'Output: Solve the problem step-by-step. First, reason through the problem clearly. At the end, respond'
            )
        msgs = self.split_image_tags(msgs)

        for i in range(len(msgs)):
            if msgs[i][0] != '<':
                msgs[i] = dict(type="text", value=msgs[i])
            else:
                msgs[i] = dict(type="image", value=image_paths[int(msgs[i][-2])])

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        subtests = sorted(set([str(i) for i in data['category_id']]))
        accuracy = {key: {} for key in subtests}

        for index, row in data.iterrows():
            cid = str(row['category_id'])
            prediction = self.extract_last_json_answer(str(row['prediction']))
            answer = str(row['answer'])
            additional = str(row['additional'])

            if (cid in ['CF1', 'CF2', 'MV1', 'MV2', 'MV3', 'P3', 'RL2', 'S1', 'S2', 'SS2', 'VZ1', 'VZ2']) or \
               (cid == 'VZ3' and not additional.isdigit()):
                if prediction.lower() in ['t', 'y', '1', 'true', 'yes']:
                    data.at[index, 'pred'] = 'T'
                elif prediction.lower() in ['f', 'n', '0', 'false', 'no']:
                    data.at[index, 'pred'] = 'F'
                else:
                    data.at[index, 'pred'] = ''
                data.at[index, 'correct'] = (data.at[index, 'pred'] == answer)
            elif cid in ['CS1', 'CS2', 'CS3']:
                data.at[index, 'pred'] = prediction
                data.at[index, 'correct'] = (data.at[index, 'pred'].lower() in [i.lower() for i in answer.split(',')])
            elif cid in ['CF3', 'I3', 'MA1', 'SS3', 'VZ3']:
                if prediction == '':
                    prediction = str(row['prediction'])
                if cid == 'VZ3':
                    prediction = self.extract_last_uppercase_letter(prediction)
                elif cid == 'CF3':
                    prediction = self.extract_last_numbers(prediction)
                    prediction = '' if len(prediction) < 2 else f'({prediction[-2]}, {prediction[-1]})'
                else:
                    prediction = self.extract_last_numbers(prediction)
                    prediction = '' if len(prediction) < 1 else prediction[-1]
                data.at[index, 'pred'] = prediction
                data.at[index, 'correct'] = (data.at[index, 'pred'] == answer)

        for index, row in data.iterrows():
            cid = str(row['category_id'])
            eid = str(row['eval_index'])
            if eid not in accuracy[cid]:
                accuracy[cid][eid] = [row['correct']]
            else:
                accuracy[cid][eid].append(row['correct'])

        for subtest in accuracy:
            for eval_index in accuracy[subtest]:
                accuracy[subtest][eval_index] = 1 if all(accuracy[subtest][eval_index]) else 0

        for subtest in accuracy:
            results = [accuracy[subtest][eval_index] for eval_index in accuracy[subtest]]
            accuracy[subtest] = sum(results) / len(results)

        accuracy['ALL'] = sum([accuracy[s] for s in accuracy]) / len([accuracy[s] for s in accuracy])

        verbose_file = get_intermediate_file_path(eval_file, '_verbose')
        dump(data, verbose_file)

        score_df = d2df(accuracy)
        score_file = get_intermediate_file_path(eval_file, '_acc')
        dump(score_df, score_file)

        return accuracy
