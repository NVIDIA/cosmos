import json
import math
import numbers
import random
import re

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageOps

from vlmeval.smp import load
from ..utils.multiple_choice import extract_answer_from_item


def get_dimension_rating(data_path, category_type='subtask_type'):
    data = load(data_path)
    result_board = {}
    for idx, item in data.iterrows():
        if item[category_type] not in result_board:
            result_board[item[category_type]] = [0, 0]
        result_board[item[category_type]][1] += 1
        if item['score']:
            result_board[item[category_type]][0] += 1

    correct = 0
    total = 0
    for key, value in result_board.items():
        correct += value[0]
        total += value[1]
        result_board[key].append(f'{value[0] / value[1] * 100:.2f}%')

    result_board['overall'] = [correct, total, f'{correct / total * 100:.2f}%']

    return result_board


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]


def extract_option(model, input_item, dataset_name):
    options = input_item['question'].split('\n')[1:]
    for id, option in enumerate(options):
        option_id = chr(ord('A') + id) + '.'
        if option.find(option_id) >= 0:
            input_item[chr(ord('A') + id)] = option[option.find(option_id) + len(option_id):].strip('. \n')
    return extract_answer_from_item(model, input_item, dataset_name)['opt']


def process_results(score_file, model_name):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score)
    data = pd.read_excel(score_file)

    # Create the prediction column based on the Score and Answer columns
    data['prediction'] = data.apply(
        lambda row: row['answer'] if row['score'] == 1 else ('Yes' if row['answer'] == 'No' else 'No'), axis=1
    )

    # Recompute metrics for tamper types including 'original' in the calculations but exclude 'original' from the output
    grouped_metrics_with_original_excluding_original = {}

    original_group = data[data['tamper_type'] == 'original']

    for tamper_type, group in data[data['tamper_type'] != 'original'].groupby('tamper_type'):
        # Combine the current group with the 'original' group
        combined_group = pd.concat([group, original_group])

        # Extract ground truth and predictions for the combined group
        y_true_group = combined_group['answer'].map({'Yes': 1, 'No': 0})
        y_pred_group = combined_group['prediction'].map({'Yes': 1, 'No': 0})

        # Calculate metrics for the combined group
        accuracy = accuracy_score(y_true_group, y_pred_group)
        precision = precision_score(y_true_group, y_pred_group, zero_division=0)
        recall = recall_score(y_true_group, y_pred_group, zero_division=0)
        f1 = f1_score(y_true_group, y_pred_group, zero_division=0)
        conf_matrix = confusion_matrix(y_true_group, y_pred_group)

        # Store metrics for the tamper_type
        grouped_metrics_with_original_excluding_original[tamper_type] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": conf_matrix.tolist()  # Convert to list for JSON compatibility
        }

        # Add the Macro Average row to the Dictionary
        # grouped_metrics_with_original_excluding_original["overall"] = macro_averages

    # Display the metrics in a dataframe for clarity
    df_grouped_metrics_with_original_excluding_original = pd.DataFrame.from_dict(
        grouped_metrics_with_original_excluding_original, orient='index'
    )

    # Compute Macro Averages for Accuracy, Precision, Recall, and F1 Score
    macro_averages = {
        "Accuracy": df_grouped_metrics_with_original_excluding_original["Accuracy"].mean(),
        "Precision": df_grouped_metrics_with_original_excluding_original["Precision"].mean(),
        "Recall": df_grouped_metrics_with_original_excluding_original["Recall"].mean(),
        "F1 Score": df_grouped_metrics_with_original_excluding_original["F1 Score"].mean(),
        "Confusion Matrix": "N/A"  # Macro average doesn't have a meaningful confusion matrix
    }

    # # Add the Macro Average row to the DataFrame
    df_grouped_metrics_with_original_excluding_original.loc["overall"] = macro_averages

    # df_grouped_metrics_with_original_excluding_original
    metrics_dict = json.loads(df_grouped_metrics_with_original_excluding_original.T.to_json())
    # Process Model Level Metrics
    formatted_data = []
    for task, task_metrics in metrics_dict.items():
        task_metrics['Model'] = model_name
        task_metrics['Task'] = task
        formatted_data.append(task_metrics)

    df_metrics = pd.DataFrame(formatted_data)

    # Reorder columns to make 'Model' and 'Task' appear first
    columns_order = ['Model', 'Task'] + [col for col in df_metrics.columns if col not in ['Model', 'Task']]
    df_metrics = df_metrics[columns_order]

    return df_metrics


def aggregate_metrics_with_macro_average(score_file):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score)

    # Load data
    data = pd.read_excel(score_file)

    # Create the prediction column based on the Score and Answer columns
    data['prediction'] = data.apply(
        lambda row: row['answer'] if row['score'] == 1 else ('Yes' if row['answer'] == 'No' else 'No'), axis=1
    )

    # Initialize a dictionary to store metrics
    task_type_metrics = {}

    # Process each task_type separately
    for task_type, task_group in data.groupby('task_type'):
        # Separate the 'original' group for the current task_type
        original_group = task_group[task_group['tamper_type'] == 'original']

        # Skip if there is no 'original' data for this task_type
        if original_group.empty:
            continue

        # Process each tamper type for the current task_type (excluding 'original')
        tamper_metrics = {}
        for tamper_type, tamper_group in task_group[task_group['tamper_type'] != 'original'].groupby('tamper_type'):

            # Combine the tamper group with the original group of the current task_type
            combined_group = pd.concat([tamper_group, original_group])

            # Map answers and predictions to binary values
            y_true = combined_group['answer'].map({'Yes': 1, 'No': 0})
            y_pred = combined_group['prediction'].map({'Yes': 1, 'No': 0})

            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            conf_matrix = confusion_matrix(y_true, y_pred)

            # Store metrics for the tamper_type
            tamper_metrics[tamper_type] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Confusion Matrix": conf_matrix.tolist()  # Convert to list for JSON compatibility
            }

        # Compute Macro Averages for the current task_type
        metrics_df = pd.DataFrame(tamper_metrics).T
        macro_average = {
            "Accuracy": metrics_df["Accuracy"].mean(),
            "Precision": metrics_df["Precision"].mean(),
            "Recall": metrics_df["Recall"].mean(),
            "F1 Score": metrics_df["F1 Score"].mean(),
            "Confusion Matrix": "N/A"  # Macro average doesn't have a meaningful confusion matrix
        }

        # Add the macro average as "overall" for the task_type
        tamper_metrics["overall"] = macro_average

        # Add tamper metrics for the current task_type to the main dictionary
        task_type_metrics[task_type] = tamper_metrics

    # Transform the nested dictionary into a DataFrame
    dataframes = []
    for task_type, metrics in task_type_metrics.items():
        task_df = pd.DataFrame.from_dict(metrics, orient='index')
        task_df['task_type'] = task_type  # Add the task_type as a column
        dataframes.append(task_df)

    # Combine all task-specific DataFrames into a single DataFrame
    result_df = pd.concat(dataframes).reset_index().rename(columns={'index': 'tamper_type'})
    # Reorder the columns to place task_type first, then tamper_type
    result_df = result_df[['task_type', 'tamper_type', 'Accuracy', 'Precision', 'Recall',
                           'F1 Score', 'Confusion Matrix']]

    # Select only numeric columns for aggregation
    numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Group by task_type and tamper_type, and calculate the mean for numeric columns
    average_metrics = result_df.groupby(['task_type', 'tamper_type'])[numeric_columns].mean().reset_index()

    return average_metrics


def check_ans(pred, gt):
    """
    Checks if the predicted answer matches the ground truth.

    Args:
        pred (str): The predicted answer.
        gt (str): The ground truth answer.

    Returns:
        bool: True if the predicted answer matches the ground truth, False otherwise.
    """
    # Convert both predictions and ground truths to lowercase and split them into options and contents
    flag = False

    # Split prediction into option and content
    pred_list = pred.lower().strip().split(' ')
    pred_option, _ = pred_list[0], ' '.join(pred_list[1:])

    # Split ground truth into option and content
    gt_list = gt.lower().strip().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])

    # Remove trailing period from ground truth content if present
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]

    # Check for matching conditions
    # Condition 1: If the predicted option is a substring of the ground truth option
    if pred_option.replace('.', '') in gt_option:
        flag = True
    # Condition 2: If the ground truth option is a substring of the predicted option
    elif gt_option in pred_option:
        flag = True
    # Condition 3: If the ground truth is a substring of the predicted answer
    elif gt in pred:
        flag = True

    return flag


def check_ans_with_model(pred, gt, model, item, dataset_name='MVBench'):
    """
    Checks if the predicted answer matches the ground truth using a given model.

    Args:
        pred (str): The predicted answer.
        gt (str): The ground truth answer.
        model: A machine learning model used for additional verification.
        item (dict): An item containing information about the question or task.
        dataset_name (str, optional): Name of the dataset being used. Defaults to 'MVBench'.

    Returns:
        bool: True if the predicted answer matches the ground truth, False otherwise.
    """
    # Initialize flag to track match status
    flag = False

    # Preprocess prediction and ground truth by converting to lowercase and splitting into options and contents
    pred_list = pred.lower().strip().split(' ')
    pred_option, _ = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().strip().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])

    # Remove trailing period from ground truth content if presen
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]

    # Check for matching conditions
    # Condition 1: If the predicted option is a substring of the ground truth option
    if pred_option.replace('.', '') in gt_option:
        flag = True
    # Condition 2: If the ground truth option is a substring of the predicted option
    elif gt_option in pred_option:
        flag = True
    # Condition 3: Use the provided model to verify the answer
    elif extract_answer_from_item(model, item, dataset_name)['opt'] == item['answer']:
        flag = True

    return flag


def check_ans_advanced(pred, gt):
    number_table = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine',
    }
    flag = False

    pred_list = pred.lower().split(' ')
    pred_option, _ = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]

    try:
        gt_content = number_table[int(gt_content.strip('. \n'))]
        print(gt_content)
    except Exception:
        pass

    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
    elif gt_content.lower().strip('. \n') in pred.lower().strip('. \n'):
        flag = True

    return flag


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class MultiGroupRandomCrop(object):
    def __init__(self, size, groups=1):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.groups = groups

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        for i in range(self.groups):
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            for img in img_group:
                assert (img.size[0] == w and img.size[1] == h)
                if w == tw and h == th:
                    out_images.append(img)
                else:
                    out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)
        return oversample_group


class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1,
                 fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop(
                (offset_w,
                 offset_h,
                 offset_w + crop_w,
                 offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(
                x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [
            self.input_size[0] if abs(
                x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))
                out_group.append(
                    img.resize(
                        (self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class ConvertDataFormat(object):
    def __init__(self, model_type):
        self.model_type = model_type

    def __call__(self, images):
        if self.model_type == '2D':
            return images
        tc, h, w = images.size()
        t = tc // 3
        images = images.view(t, 3, h, w)
        images = images.permute(1, 0, 2, 3)
        return images


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2)
                                   for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1]
                                       for x in img_group], axis=2)
            else:
                # print(np.concatenate(img_group, axis=2).shape)
                # print(img_group[0].shape)
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(
                    pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data
