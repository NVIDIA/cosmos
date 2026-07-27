import os.path as osp
import warnings

from vlmeval.smp import dump, get_intermediate_file_path, listinstr, load
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge


class ImageYORNDataset(ImageBaseDataset):

    TYPE = 'Y/N'

    DATASET_URL = {
        'MME': 'https://opencompass.openxlab.space/utils/VLMEval/MME.tsv',
        'HallusionBench': 'https://opencompass.openxlab.space/utils/VLMEval/HallusionBench.tsv',
        'POPE': 'https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv',
        'AMBER': 'https://huggingface.co/datasets/yifanzhang114/AMBER_base64/resolve/main/AMBER.tsv',
        'VSR-zeroshot': (
            "https://huggingface.co/datasets/ignoreandfly/"
            "vsr_zeroshot_tsv/resolve/main/vsr_zeroshot_dataset_yn_strict.tsv"), }

    DATASET_MD5 = {
        'MME': 'b36b43c3f09801f5d368627fb92187c3',
        'HallusionBench': '0c23ac0dc9ef46832d7a24504f2a0c7c',
        'POPE': 'c12f5acb142f2ef1f85a26ba2fbe41d5',
        'AMBER': '970d94c0410916166e0a76ba75da7934',
        'VSR-zeroshot': '5ff5e49908ac1cfad35c60b92b001aeb',
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        if self.dataset_name == 'AMBER':
            assert sum([x['type'] == 'text' for x in msgs]) == 1
            for item in msgs:
                if item['type'] == 'text':
                    item['value'] += '\nPlease answer yes or no.'
        return msgs

    # It returns a dataframe
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.yorn import (AMBER_rating, Hallusion_rating, MME_rating, POPE_rating,
                                 VSR_rating, YOrN_auxeval, YOrN_Extraction, default_rating)

        dataset = self.dataset_name
        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        storage = get_intermediate_file_path(eval_file, '_auxmatch')
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            ans_map = {k: YOrN_Extraction(v) for k, v in zip(data['index'], data['prediction'])}
            if osp.exists(tmp_file):
                tmp = load(tmp_file)
                for k in tmp:
                    if ans_map[k] == 'Unknown' and tmp[k] != 'Unknown':
                        ans_map[k] = tmp[k]

            data['extracted'] = [ans_map[x] for x in data['index']]
            unknown = data[data['extracted'].isin(['Unknown', 'JudgeFailed'])]

            model = judge_kwargs.get('model', 'exact_matching')
            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None

            if model is not None:
                lt = len(unknown)
                lines = [unknown.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = list(unknown['index'])
                if len(tups):
                    res = track_progress_rich(
                        YOrN_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            data['extracted'] = [ans_map[x] for x in data['index']]
            dump(data, storage)

        data = load(storage)
        if listinstr(['AMBER'], dataset):
            data['score'] = (data['answer'].str.lower() == data['extracted'].str.lower())
        else:
            data['score'] = (data['answer'] == data['extracted'])
        dump(data, storage)

        if dataset is not None and listinstr(['MME'], dataset):
            score = MME_rating(storage)
        elif dataset is not None and listinstr(['Hallusion'], dataset):
            score = Hallusion_rating(storage)
        elif dataset is not None and listinstr(['POPE'], dataset):
            score = POPE_rating(storage)
        elif dataset is not None and listinstr(['AMBER'], dataset):
            score = AMBER_rating(storage)
        elif dataset is not None and listinstr(['VSR-zeroshot'], dataset):
            score = VSR_rating(storage)
        else:
            score = default_rating(storage)

        score_tgt = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_tgt)
        return score
