import os
import os.path as osp

from vlmeval.smp import load
from vlmeval.smp.file import LMUDataRoot
from .image_base import ImageBaseDataset

MAX_LEN = 300
MMSIVIDEO_INSTRUCTION = """
# MMSI-video-Bench Evaluation on VLMEvalkit

Download the images of MMSI-video-Bench dataset (frames.zip, ref_images.zip) from
https://huggingface.co/datasets/rbler/MMSI-Video-Bench and put them under `LMUDATA`.
Below shows the expected folder structure.

```
LMUDATA/
├──MMSIVideo.tsv
├──images/
├────MMSIVideo/
├──────frames/
├──────ref_images/
├──────...
```

"""


# NOTE: This file is deprecated. For the latest MMSI-Video implementation,
# see `vlmeval/dataset/mmsibench.py` (class `MMSIVideoBench`).


class MMSIVideoBench(ImageBaseDataset):
    TYPE = 'MCQ'
    Settings = ['_U50', '_SC']
    DATASET_URL = {}
    DATASET_MD5 = {}
    for Setting in Settings:
        DATASET_NAME = 'MMSIVideo' + Setting
        DATASET_URL[
            DATASET_NAME] = f'https://opencompass.openxlab.space/utils/VLMEval/{DATASET_NAME}.tsv'
        DATASET_MD5[DATASET_NAME] = 'cd0f9174a5517a191d7656df84fdd6f7'
    main_categories = [
        '(Cross-Video) Memoery Update',
        '(Cross-Video) Multi-View Integration',
        'Planning',
        'Prediction',
        '(Motion Understanding) Camera Motion',
        '(Motion Understanding) Instance Motion',
        '(Motion Understanding) Interactive Motion',
        '(Spatial Construction) Instance-Instance Spatial Relationship',
        '(Spatial Construction) Instance-Scene Spatial Relationship',
        '(Spatial Construction) Scene-Scene Spatial Relationship',
        '(Spatial Construction) Instance/Scene Attribute',
        '(Spatial Construction) Camera-Instance Spatial Relationship',
        '(Spatial Construction) Camera-Scene Spatial Relationship']
    difficulty_categories = [
        '(Easy2hard)level_1',
        '(Easy2hard)level_2',
        '(Easy2hard)level_3']
    indoor_categories = [
        '(Indoor_Scene_Perception)Avg',
        '(Indoor_Scene_Perception)Static Scene (Instance-Centric)',
        '(Indoor_Scene_Perception)Static Scene (Camera-Centric)',
        '(Indoor_Scene_Perception)Dynamic Scene']
    robot_categories = [
        '(Robot)Avg',
        '(Robot)Navigation',
        '(Robot)Manipulation']
    grounding_categories = ['(Grounding)Avg', '(Grounding)Time Localization',
                            '(Grounding)Target Grounding']

    all_categories = {'Main Bench': main_categories,
                      'Three Difficulty Levels': difficulty_categories,
                      'Indoor Scene Percpetion Sub-Bench': indoor_categories,
                      'Robot Sub-Bench': robot_categories,
                      'Grounding Sub-Bench': grounding_categories}

    def __init__(self, dataset='MMSIVideo_SC'):
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different
        # directory
        self.dataset_name = dataset
        self.frames_dir = osp.join(ROOT, 'images', 'MMSIVideo', 'frames')
        self.ref_images_dir = osp.join(
            ROOT, 'images', 'MMSIVideo', 'ref_images')
        assert osp.exists(self.frames_dir), MMSIVIDEO_INSTRUCTION
        assert osp.exists(self.ref_images_dir), MMSIVIDEO_INSTRUCTION

        data = self.load_data(self.dataset_name)

        data['index'] = [int(qid.split('_')[1]) for qid in data['id']]
        if 'SC' in dataset:
            self.max_frame = MAX_LEN
        elif 'U50' in dataset:
            self.max_frame = 50
        self.data = data

    def build_prompt(self, line):

        if isinstance(line, int):
            sample = self.data.iloc[line]
        else:
            sample = line

        def interval_sampling_list(a, b):
            step = (a - 1) / (b - 1) if b > 1 else 0
            indices = [int(i * step) for i in range(b)]
            return indices

        def proportional_sample_from_lists(num_list, k):
            total_indices = interval_sampling_list(sum(num_list), k)

            sum_list = [sum(num_list[:i + 1]) for i in range(len(num_list))]
            sum_list = [0] + sum_list
            indices_list = []
            for i in range(len(sum_list) - 1):
                raw_indices = [k - sum_list[i]
                               for k in total_indices if k >= sum_list[i] and k < sum_list[i + 1]]
                indices_list.append(raw_indices)
            return indices_list

        def split_txt_based_on_images(text, image_paths):
            ret = []
            assert text.count('<image>') == len(image_paths)
            split_text_list = text.split('<image>')
            for index_ in range(len(image_paths)):
                ret.append(split_text_list[index_])
                ret.append(image_paths[index_])
            ret.append(split_text_list[-1])
            ret = [item for item in ret if len(item) > 0]
            return ret

        split_text = sample['system_prompt'] + '\n' + sample['task_prompt'] + \
            '\n' + sample['user_prompt'] + sample['format_prompt']
        split_images = []

        ori_frames_list = eval(sample['frames_list'])
        ref_images_list = eval(sample['ref_images'])

        total_frames = sum([len(ori_frames) for ori_frames in ori_frames_list])
        sampled_frames_list = []
        if total_frames > self.max_frame:
            indices_list = proportional_sample_from_lists(
                [len(ori_frames) for ori_frames in ori_frames_list],
                self.max_frame)
            for i in range(len(indices_list)):
                if len(indices_list[i]) < 1:
                    indices_list[i] = [0]
                sampled_frames_list.append(
                    [ori_frames_list[i][j] for j in indices_list[i]])
        else:
            sampled_frames_list = [ori_frames_list[i]
                                   for i in range(len(ori_frames_list))]
        assert split_text.count('<video>') == len(sampled_frames_list)
        assert split_text.count('<image>') == len(ref_images_list)
        for frames in sampled_frames_list:
            split_text = split_text.replace(
                '<video>', '<image>' * len(frames), 1)
            split_images.extend(frames)

        split_images = [
            os.path.join(
                self.frames_dir,
                image_path) for image_path in split_images]
        split_images.extend([os.path.join(self.ref_images_dir, img_name)
                            for img_name in ref_images_list])
        info_list = split_txt_based_on_images(split_text, split_images)
        content = []

        for info_item in info_list:
            if os.path.exists(info_item):
                content.append(dict(type='image', value=info_item))
            else:
                content.append(dict(type='text', value=info_item))
        return content

    def evaluate(self, eval_file, **judge_kwargs):
        def clear_words(text):
            return text.replace(
                ' ',
                '').replace(
                '\"',
                '').replace(
                "\'",
                '').replace(
                '\n',
                '').replace(
                    ':',
                '')

        def is_nan_or_none(value):
            if value is None:
                return True
            try:
                if isinstance(value, str) and value.lower() in [
                        'nan', 'null', 'none', '']:
                    return True
                if isinstance(value, float) and value != value:  # nan != nan
                    return True
            except BaseException:
                pass
            return False

        def extract_answer(response):
            response = response.replace(
                '<answer>', '').replace(
                '</answer>', '')
            if response is None or 'no answer' in response:
                return 'O'
            if 'boxed{' in response:
                split_text = response.split('boxed{')[1].split('}')[0]
                split_text = clear_words(split_text)
                return split_text
            words = [
                '\"answer\":',
                'answer is',
                'answer:',
                '\"Answer\":',
                'Answer is',
                'Answer:']
            for word in words:
                if word in response:
                    split_text = response.split(word)[-1]
                    split_text = split_text.split(',')[0].split('.')[0]
                    split_text = clear_words(split_text)
                    return split_text

            if clear_words(response.split('.')[0]) in [
                    'A', 'B', 'C', 'D', 'E', 'F']:
                return clear_words(response.split('.')[0])
            else:
                return 'O'
        df = load(eval_file)
        data = df.to_dict(orient="list")
        score_dict = {'Overall': []}
        SUB_BENCHS = [
            'Easy2hard',
            'Grounding',
            'Indoor_Scene_Perception',
            'Robot']
        error_list = []
        for index in range(len(data['id'])):
            question_type, gt, response = data['type'][index], data['ground_truth'][index], data['prediction'][index]
            try:
                pred = extract_answer(response)
                assert pred in ['A', 'B', 'C', 'D', 'E', 'F']
                if question_type not in score_dict:
                    score_dict[question_type] = []
                score_dict[question_type].append(float(pred == gt))
                score_dict['Overall'].append(float(pred == gt))
                for BENCH in SUB_BENCHS:
                    if is_nan_or_none(data[BENCH][index]):
                        continue
                    beglongs_ = [
                        f'({BENCH})Avg',
                        f'({BENCH}){data[BENCH][index]}']
                    for key in beglongs_:
                        if key not in score_dict:
                            score_dict[key] = []
                        score_dict[key].append(float(pred == gt))
            except BaseException:
                if question_type not in score_dict:
                    score_dict[question_type] = []
                score_dict[question_type].append(0.0)
                score_dict['Overall'].append(0.0)
                for BENCH in SUB_BENCHS:
                    if is_nan_or_none(data[BENCH][index]):
                        continue
                    beglongs_ = [
                        f'({BENCH})Avg',
                        f'({BENCH}){data[BENCH][index]}']
                    for key in beglongs_:
                        if key not in score_dict:
                            score_dict[key] = []
                        score_dict[key].append(0.0)
                error_list.append(index)
        static_results = {}
        print(score_dict.keys())
        print('-' * 50)
        print('Overall: ',
              sum(score_dict['Overall']) / len(score_dict['Overall']),
              len(score_dict['Overall']))
        static_results['Overall'] = sum(
            score_dict['Overall']) / len(score_dict['Overall'])
        static_results['Failure'] = len(error_list)
        print(
            f'Failure Count/ Total Count: {len(error_list)} / {len(data["prediction"])}')
        for bench in self.all_categories:
            print('-' * 50)
            print('Evaluating on ', bench)
            for key in self.all_categories[bench]:
                print(key,
                      ': ',
                      sum(score_dict[key]) / len(score_dict[key]),
                      len(score_dict[key]))
                static_results[key] = sum(
                    score_dict[key]) / len(score_dict[key])
        return static_results
