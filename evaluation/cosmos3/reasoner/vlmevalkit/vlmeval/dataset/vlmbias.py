import os.path as osp

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich
from .image_vqa import ImageVQADataset
from .utils.multiple_choice import report_acc
from .utils.omni_verifier import OmniVerifier


def VLMBias_auxeval(verifier, pred, gt):
    return verifier.verify(pred, gt)


class VLMBias(ImageVQADataset):

    DATASET_URL = {'VLMBias': 'https://opencompass.openxlab.space/utils/VLMEval/VLMBias.tsv'}
    DATASET_MD5 = {'VLMBias': '23d0119c89243954e81f41a11a2ef347'}

    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs.pop('model', 'gpt-4o')
        storage = get_intermediate_file_path(eval_file, f'_{model}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}_tmp')
        nproc = judge_kwargs.pop('nproc', 16)

        if not osp.exists(storage):
            data = load(eval_file)
            assert 'answer' in data and 'prediction' in data
            verifier = OmniVerifier(tmpl='brace', judge=model, retry=3, timeout=60, lower_case=True, rule_only=False)
            if not verifier.rule_only:
                assert verifier.judge.working(), 'OmniVerifier should have a working API. '
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(verifier, line['prediction'], line['answer']) for line in lines]
            indices = [line['index'] for line in lines]
            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]
            if len(indices):
                new_results = track_progress_rich(
                    VLMBias_auxeval, tups,
                    nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k][0] == v[0] and ans[k][1] == v[1]
            data['hit'] = [ans[idx][0] for idx in data['index']]
            data['log'] = [ans[idx][1] for idx in data['index']]
            dump(data, storage)

        data = load(storage)
        acc = report_acc(data)
        score_file = get_intermediate_file_path(eval_file, '_acc')
        dump(acc, score_file)
        return acc
