import os
import os.path as osp
import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd

from vlmeval.smp import (LMUDataRoot, decode_base64_to_image_file, download_file, file_size,
                         istype, load, md5, mmqa_display, read_ok, toliststr)


def img_root_map(dataset):
    if 'MM_NIAH' in dataset:
        return 'MMNIAH'
    if 'CRPE' in dataset:
        return 'CRPE'
    if 'OCRVQA' in dataset:
        return 'OCRVQA'
    if 'COCO_VAL' == dataset:
        return 'COCO'
    if "QSpatial" in dataset:
        return "QSpatial"

    mmbench_root_map = {
        'MMBench_DEV_EN': 'MMBench', 'MMBench_TEST_EN': 'MMBench',
        'MMBench_DEV_CN': 'MMBench', 'MMBench_TEST_CN': 'MMBench',
        'MMBench_DEV_KO': 'MMBench',
        'MMBench': 'MMBench', 'MMBench_CN': 'MMBench',
        'MMBench_DEV_EN_V11': 'MMBench_V11', 'MMBench_TEST_EN_V11': 'MMBench_V11',
        'MMBench_DEV_CN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_V11',
        'MMBench_V11': 'MMBench', 'MMBench_CN_V11': 'MMBench',
    }
    if dataset in mmbench_root_map:
        return mmbench_root_map[dataset]
    return dataset


class ImageBaseDataset:

    MODALITY = 'IMAGE'
    DATASET_URL = {}
    DATASET_MD5 = {}

    def __init__(self, dataset='MMBench', skip_noimg=True, **kwargs):
        # **kwargs absorbs profile-level fields (e.g. model_family) that
        # subclasses didn't consume in their own __init__; harmless here.
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, 'images', img_root_map(dataset))

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]

        data['index'] = [str(x) for x in data['index']]

        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name_legacy = url.split('/')[-1]
        file_name = f"{self.dataset_name}.tsv"
        data_path_legacy = osp.join(data_root, file_name_legacy)
        data_path = osp.join(data_root, file_name)

        self.data_path = data_path
        if osp.exists(data_path):
            if file_md5 is None or md5(data_path) == file_md5:
                pass
            else:
                warnings.warn(f'The tsv file is in {data_root}, but the md5 does not match, will re-download')
                download_file(url, data_path)
                update_flag = True
        else:
            if osp.exists(data_path_legacy) and (file_md5 is None or md5(data_path_legacy) == file_md5):
                warnings.warn(
                    'Due to a modification in #1055, the local target file name has changed. '
                    f'We detected the tsv file with legacy name {data_path_legacy} exists and will do the rename. '
                )
                import shutil
                shutil.move(data_path_legacy, data_path)
            else:
                download_file(url, data_path)
                update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                if 'image_path' in line:
                    image_path = line['image_path']
                else:
                    index = line['index']
                    image_path = [f'{index}_{i}.png' for i in range(len(line['image']))]
                for img, im_name in zip(line['image'], image_path):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)

            elif isinstance(line['image'], str) and 'image_path' in line:
                assert isinstance(line['image_path'], str)
                tgt_path = osp.join(self.img_root, line['image_path'])
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.png")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])
            read_ok_flag = [read_ok(x) for x in tgt_path]
            # Might be the Relative Path
            if not all(read_ok_flag):
                tgt_path_abs = [osp.join(self.img_root, x) for x in tgt_path]
                read_ok_flag = [read_ok(x) for x in tgt_path_abs]
                assert read_ok_flag, f"Field `image` is missing and we could not find {tgt_path} both as absolute or relative paths. "  # noqa
                tgt_path = tgt_path_abs

        return tgt_path

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        assert isinstance(line, pd.Series) or isinstance(line, dict)
        mmqa_display(line)

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        url = self.DATASET_URL.get(dataset, None)
        if url is None or url == '':
            url = dataset + '.tsv'
        file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
        return self.prepare_tsv(url, file_md5)

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass
