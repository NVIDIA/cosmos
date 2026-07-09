import ast
import os

from huggingface_hub import snapshot_download
from tqdm import tqdm

from vlmeval.smp import get_cache_path, load, modelscope_flag_set, toliststr
from .image_mcq import ImageMCQDataset


class MindCubeBench(ImageMCQDataset):
    """
    MindCube.

    Reference:
      Spatial Mental Modeling from Limited Views
      https://arxiv.org/abs/2506.21458
    """

    TYPE = 'MCQ'

    DATASET_URL = {
        'MindCubeBench_tiny_raw_qa': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_tiny_raw_qa.tsv',  # noqa: E501
        'MindCubeBench_raw_qa': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_raw_qa.tsv'  # noqa: E501
    }
    DATASET_MD5 = {
        'MindCubeBench_tiny_raw_qa': '35f69fc30d7c2d2880417ce0769f5347',
        'MindCubeBench_raw_qa': '6a53cd353bc93d8e3a87098249c806ad'
    }

    def _task_category(self):
        return ['rotation', 'among', 'around']

    def prepare_tsv(self, url, file_md5=None, repo_id='MLL-Lab/MindCube'):
        data = super().prepare_tsv(url, file_md5)

        SENTINEL_NAME = '.mindcubebench_extracted'
        cache_path = get_cache_path(repo_id)

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text='ok'):
                tmp = sentinel_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            def unzip_hf_zip(pth):
                import zipfile

                base_dir = pth
                zip_files = [
                    os.path.join(base_dir, f) for f in os.listdir(base_dir)
                    if f.endswith('.zip')
                ]
                zip_files.sort()

                for zip_file in tqdm(zip_files, desc='Unpacking Origin Data...'):
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue

                            rel = os.path.normpath(info.filename).lstrip('/\\')
                            dst = os.path.join(pth, rel)

                            absp = os.path.abspath(pth)
                            absd = os.path.abspath(dst)
                            if not absd.startswith(absp + os.sep):
                                raise RuntimeError(f'Unsafe path in zip: {info.filename}')

                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with zf.open(info, 'r') as src, open(dst, 'wb') as out:
                                out.write(src.read())

                sentinel_path = os.path.join(pth, SENTINEL_NAME)
                _write_sentinel(sentinel_path, text='done')
                print('MindCube data extracted to current directory with original layout.')

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)

        # === Transfer rel path to abs path ===
        if 'image_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, s.lstrip(r'\/')))

            def to_abs(p):
                if isinstance(p, list):
                    return [fix_one(xx) for xx in p]
                if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return [fix_one(xx) for xx in lst]
                    except Exception:
                        pass
                return fix_one(p)

            data['image_path'] = data['image_path'].map(to_abs)

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        # # Raw QA prompt format use in paper
        prompt = line['input_prompt']

        msgs = self.build_msgs(tgt_path, prompt)
        return msgs

    @staticmethod
    def build_msgs(tgt_path, prompt):
        """
        Interlaced text and pictures
        """
        images = tgt_path if isinstance(tgt_path, list) else [tgt_path]

        parts = prompt.split('<image>')
        segs = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                segs.append(dict(type='text', value=part))
            if (i != len(parts) - 1) and (i < len(images)):
                segs.append(dict(type='image', value=images[i]))

        return [s for s in segs if s['value']]

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='category',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'MindCubeBench'),
        )
