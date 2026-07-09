import csv
import hashlib
import json
import mimetypes
import multiprocessing as mp
import os
import os.path as osp
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import validators

from .misc import toliststr
from .vlm import decode_base64_to_image_file


def decode_img_omni(tup):
    root, im, p = tup
    images = toliststr(im)
    paths = toliststr(p)
    if len(images) > 1 and len(paths) == 1:
        paths = [osp.splitext(p)[0] + f'_{i}' + osp.splitext(p)[1] for i in range(len(images))]

    assert len(images) == len(paths)
    paths = [osp.join(root, p) for p in paths]
    for p, im in zip(paths, images):
        if osp.exists(p):
            continue
        if isinstance(im, str) and len(im) > 64:
            decode_base64_to_image_file(im, p)
    return paths


def localize_df(data, dname, nproc=32):
    assert 'image' in data
    indices = list(data['index'])
    indices_str = [str(x) for x in indices]
    images = list(data['image'])
    image_map = {x: y for x, y in zip(indices_str, images)}

    root = LMUDataRoot()
    root = osp.join(root, 'images', dname)
    os.makedirs(root, exist_ok=True)

    if 'image_path' in data:
        img_paths = list(data['image_path'])
    else:
        img_paths = []
        for i in indices_str:
            if len(image_map[i]) <= 64 and isinstance(image_map[i], str):
                idx = image_map[i]
                assert idx in image_map and len(image_map[idx]) > 64
                img_paths.append(f'{idx}.jpg')
            else:
                img_paths.append(f'{i}.jpg')

    tups = [(root, im, p) for p, im in zip(img_paths, images)]

    pool = mp.Pool(32)
    ret = pool.map(decode_img_omni, tups)
    pool.close()
    data.pop('image')
    if 'image_path' not in data:
        data['image_path'] = [x[0] if len(x) == 1 else x for x in ret]
    return data


def LMUDataRoot():
    if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
        return os.environ['LMUData']
    home = osp.expanduser('~')
    root = osp.join(home, 'LMUData')
    os.makedirs(root, exist_ok=True)
    return root


def HFCacheRoot():
    cache_list = ['HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_HOME']
    for cache_name in cache_list:
        if cache_name in os.environ and osp.exists(os.environ[cache_name]):
            if os.environ[cache_name].split('/')[-1] == 'hub':
                return os.environ[cache_name]
            else:
                return osp.join(os.environ[cache_name], 'hub')
    home = osp.expanduser('~')
    root = osp.join(home, '.cache', 'huggingface', 'hub')
    os.makedirs(root, exist_ok=True)
    return root


def MMBenchOfficialServer(dataset_name):
    root = LMUDataRoot()

    if dataset_name in ['MMBench', 'MMBench_V11', 'MMBench_CN', 'MMBench_CN_V11']:
        ans_file = f'{root}/{dataset_name}.tsv'
        if osp.exists(ans_file):
            data = load(ans_file)
            if 'answer' in data and sum([pd.isna(x) for x in data['answer']]) == 0:
                return True

    if dataset_name in ['MMBench_TEST_EN', 'MMBench_TEST_CN', 'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11']:
        ans_file1 = f'{root}/{dataset_name}.tsv'
        mapp = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_CN': 'MMBench_CN',
            'MMBench_TEST_EN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11',
        }
        ans_file2 = f'{root}/{mapp[dataset_name]}.tsv'
        for f in [ans_file1, ans_file2]:
            if osp.exists(f):
                data = load(f)
                if 'answer' in data and sum([pd.isna(x) for x in data['answer']]) == 0:
                    return True
    return False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,
                      (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        # 处理 DataFrame 对象
        if isinstance(data, pd.DataFrame):
            # 转换为 records 格式（列表格式）
            data = data.to_dict('records')
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    try:
        return handlers[suffix](data, f, **kwargs)
    except Exception:
        # if dump failed, fallback to pkl format
        pkl_file = f.rsplit('.', 1)[0] + '.pkl'
        warnings.warn(f'Failed to dump to {suffix} format, falling back to pkl: {pkl_file}')
        return dump_pkl(data, pkl_file, **kwargs)


def get_pred_file_format():
    pred_format = os.getenv('PRED_FORMAT', '').lower()
    if pred_format == '':
        return 'tsv'  # default format (xlsx has 32767 char cell limit that silently truncates long predictions)
    else:
        assert pred_format in ['tsv', 'xlsx', 'json'], f'Unsupported PRED_FORMAT {pred_format}'
        return pred_format


def get_eval_file_format():
    eval_format = os.getenv('EVAL_FORMAT', '').lower()
    if eval_format == '':
        return 'csv'  # default format
    else:
        assert eval_format in ['csv', 'json'], f'Unsupported EVAL_FORMAT {eval_format}'
        return eval_format


def get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True):
    if use_env_format:
        file_format = get_pred_file_format()
        if file_format == 'xlsx':
            return osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')
        elif file_format == 'tsv':
            return osp.join(work_dir, f'{model_name}_{dataset_name}.tsv')
        elif file_format == 'json':
            return osp.join(work_dir, f'{model_name}_{dataset_name}.json')
    else:
        # default
        return osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')


def get_eval_file_path(eval_file, judge_model, use_env_format=True):
    suffix = eval_file.split('.')[-1]
    if use_env_format:
        file_format = get_eval_file_format()
        if file_format == 'csv':
            return eval_file.replace(f'.{suffix}', f'_{judge_model}.csv')
        elif file_format == 'json':
            return eval_file.replace(f'.{suffix}', f'_{judge_model}.json')
    else:
        # default
        return eval_file.replace(f'.{suffix}', f'_{judge_model}.xlsx')


def _should_convert_to_dataframe(data):
    if not isinstance(data, dict):
        return False
    if not data:
        return False
    if 'columns' in data and 'data' in data:
        return True
    values = list(data.values())
    if all(not isinstance(v, (list, dict)) for v in values):
        return False
    if any(isinstance(v, list) for v in values):
        lists = [v for v in values if isinstance(v, list)]
        if lists and all(len(lst) == len(lists[0]) for lst in lists):
            return True

    return False


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    import validators
    if validators.url(f):
        tgt = osp.join(LMUDataRoot(), 'files', osp.basename(f))
        if not osp.exists(tgt):
            download_file(f, tgt)
        f = tgt

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def download_file(url, filename=None):
    import urllib.request

    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if filename is None:
        filename = url.split('/')[-1]

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    except Exception as e:
        import logging
        logging.warning(f'{type(e)}: {e}')
        # Handle Failed Downloads from huggingface.co
        if 'huggingface.co' in url:
            url_new = url.replace('huggingface.co', 'hf-mirror.com')
            try:
                download_file(url_new, filename)
                return filename
            except Exception as e:
                logging.warning(f'{type(e)}: {e}')
                raise Exception(f'Failed to download {url}')
        else:
            raise Exception(f'Failed to download {url}')

    return filename


def ls(dirname='.', match=[], mode='all', level=1):
    if isinstance(level, str):
        assert '+' in level
        level = int(level[:-1])
        res = []
        for i in range(1, level + 1):
            res.extend(ls(dirname, match=match, mode='file', level=i))
        return res

    if dirname == '.':
        ans = os.listdir(dirname)
    else:
        ans = [osp.join(dirname, x) for x in os.listdir(dirname)]
    assert mode in ['all', 'dir', 'file']
    assert level >= 1 and isinstance(level, int)
    if level == 1:
        if isinstance(match, str):
            match = [match]
        for m in match:
            if len(m) == 0:
                continue
            if m[0] != '!':
                ans = [x for x in ans if m in x]
            else:
                ans = [x for x in ans if m[1:] not in x]
        if mode == 'dir':
            ans = [x for x in ans if osp.isdir(x)]
        elif mode == 'file':
            ans = [x for x in ans if not osp.isdir(x)]
        return ans
    else:
        dirs = [x for x in ans if osp.isdir(x)]
        res = []
        for d in dirs:
            res.extend(ls(d, match=match, mode=mode, level=level - 1))
        return res


def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f


def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))


def md5(s):
    hash = hashlib.new('md5')
    if osp.exists(s):
        with open(s, 'rb') as f:
            for chunk in iter(lambda: f.read(2**20), b''):
                hash.update(chunk)
    else:
        hash.update(s.encode('utf-8'))
    return str(hash.hexdigest())


def last_modified(pth):
    stamp = osp.getmtime(pth)
    m_ti = time.ctime(stamp)
    t_obj = time.strptime(m_ti)
    t = time.strftime('%Y%m%d%H%M%S', t_obj)[2:]
    return t


def parse_file(s):
    if osp.exists(s) and s != '.':
        assert osp.isfile(s)
        suffix = osp.splitext(s)[1].lower()
        # 添加对webp的支持
        if suffix == '.webp':
            return ('image/webp', s)
        mime = mimetypes.types_map.get(suffix, 'unknown')
        return (mime, s)
    elif s.startswith('data:image/'):
        # To be compatible with OPENAI base64 format
        content = s[11:]
        mime = content.split(';')[0]
        content = ';'.join(content.split(';')[1:])
        dname = osp.join(LMUDataRoot(), 'files')
        assert content.startswith('base64,')
        b64 = content[7:]
        os.makedirs(dname, exist_ok=True)
        tgt = osp.join(dname, md5(b64) + '.png')
        decode_base64_to_image_file(b64, tgt)
        return parse_file(tgt)
    elif validators.url(s):
        suffix = osp.splitext(s)[1].lower()
        # 添加对webp的支持
        if suffix == '.webp':
            mime = 'image/webp'
        elif suffix in mimetypes.types_map:
            mime = mimetypes.types_map[suffix]
            dname = osp.join(LMUDataRoot(), 'files')
            os.makedirs(dname, exist_ok=True)
            tgt = osp.join(dname, md5(s) + suffix)
            download_file(s, tgt)
            return (mime, tgt)
        else:
            return ('url', s)

    else:
        return (None, s)


def file_size(f, unit='GB'):
    stats = os.stat(f)
    div_map = {
        'GB': 2 ** 30,
        'MB': 2 ** 20,
        'KB': 2 ** 10,
    }
    return stats.st_size / div_map[unit]


def parquet_to_tsv(file_path):
    data = pd.read_parquet(file_path)
    pth = '/'.join(file_path.split('/')[:-1])
    data_name = file_path.split('/')[-1].split('.')[0]
    data.to_csv(osp.join(pth, f'{data_name}.tsv'), sep='\t', index=False)


def fetch_aux_files(eval_file):
    file_root = osp.dirname(eval_file)
    file_name = osp.basename(eval_file)

    eval_id = osp.basename(file_root)
    if eval_id[:3] == 'T20' and eval_id[9:11] == '_G':
        model_name = osp.basename(osp.dirname(file_root))
    else:
        model_name = eval_id

    dataset_name = osp.splitext(file_name)[0][len(model_name) + 1:]
    from vlmeval.dataset import SUPPORTED_DATASETS
    to_handle = []
    for d in SUPPORTED_DATASETS:
        if d.startswith(dataset_name) and d != dataset_name:
            to_handle.append(d)
    fs = ls(file_root, match=f'{model_name}_{dataset_name}')
    if len(to_handle):
        for d in to_handle:
            fs = [x for x in fs if d not in x]
    return fs


def get_file_extension(file_path):
    return file_path.split('.')[-1]


def get_intermediate_file_path(eval_file, suffix, target_format=None):
    original_ext = get_file_extension(eval_file)

    def ends_with_list(s, lst):
        for item in lst:
            if s.endswith(item):
                return True
        return False

    if target_format is None:
        if ends_with_list(suffix, ['_tmp', '_response', '_processed']):
            target_format = 'pkl'
        elif ends_with_list(suffix, ['_rating', '_config', '_meta']):
            target_format = 'json'
        elif ends_with_list(suffix, ['_acc', '_fine', '_metrics']):
            target_format = get_eval_file_format()
        else:
            target_format = get_pred_file_format()

    return eval_file.replace(f'.{original_ext}', f'{suffix}.{target_format}')


def prepare_reuse_files(pred_root_meta, eval_id, model_name, dataset_name, reuse, reuse_aux):
    import shutil

    from .misc import timestr
    work_dir = osp.join(pred_root_meta, eval_id)
    os.makedirs(work_dir, exist_ok=True)
    if not reuse:
        files = ls(work_dir, match=f'{model_name}_{dataset_name}')
        if len(files):
            t_str = timestr('second')
            bak_dir = osp.join(work_dir, f'bak_{t_str}_{dataset_name}')
            os.makedirs(bak_dir, exist_ok=True)
            for f in files:
                shutil.move(f, bak_dir)
            warnings.warn(
                f'--reuse flag not set but history records detected in {work_dir}. '
                f'Those files are moved to {bak_dir} for backup. '
            )
            return
    # reuse flag is set
    prev_pred_roots = ls(pred_root_meta, mode='dir')
    prev_pred_roots.sort()
    prev_pred_roots.remove(work_dir)

    files = ls(work_dir, match=f'{model_name}_{dataset_name}.')
    prev_file = None
    prev_aux_files = None
    if len(files):
        pass
    else:
        for root in prev_pred_roots[::-1]:
            fs = ls(root, match=f'{model_name}_{dataset_name}.')
            if len(fs):
                if len(fs) > 1:
                    warnings.warn(f'Multiple candidates in {root}: {fs}. Will use {fs[0]}')
                prev_file = fs[0]
                prev_aux_files = fetch_aux_files(prev_file)
                break
        if prev_file is not None:
            warnings.warn(f'--reuse is set, will reuse prediction file {prev_file}')
            os.system(f'cp {prev_file} {work_dir}')

    if not reuse_aux:
        warnings.warn(f'--reuse-aux is not set, all auxiliary files in {work_dir} are removed. ')
        os.system(f'rm -rf {osp.join(work_dir, f"{model_name}_{dataset_name}_*openai*")}')
        os.system(f'rm -rf {osp.join(work_dir, f"{model_name}_{dataset_name}_*csv")}')
        os.system(f'rm -rf {osp.join(work_dir, f"{model_name}_{dataset_name}_*json")}')
        os.system(f'rm -rf {osp.join(work_dir, f"{model_name}_{dataset_name}_*pkl")}')
        os.system(f'rm -rf {osp.join(work_dir, f"{model_name}_{dataset_name}_*gpt*")}')
    elif prev_aux_files is not None:
        for f in prev_aux_files:
            os.system(f'cp {f} {work_dir}')
            warnings.warn(f'--reuse-aux is set, will reuse auxiliary file {f}')
    return
