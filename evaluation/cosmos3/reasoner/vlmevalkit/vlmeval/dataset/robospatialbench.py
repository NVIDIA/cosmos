import ast
import os

import numpy as np
import pandas as pd
from PIL import Image

from vlmeval.smp import dump, load, toliststr
from .image_vqa import ImageVQADataset
from .utils.spatial_bench.tools.utils import Point2DParser


class RoboSpatialBench(ImageVQADataset):
    """
    RoboSpatial-Home.

    Reference:
      RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics
      https://arxiv.org/abs/2411.16537
    """

    DATASET_URL = {
        'RoboSpatialHome': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/RoboSpatialHome.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'RoboSpatialHome': 'e2d01075ce470bf8e06d3a16c8fa10bc'
    }

    def _task_category(self):
        return ['compatibility', 'configuration', 'context']

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        raw_q = str(line['question'])
        category = line['category']

        if category == 'context':
            main_q = raw_q.split('Your answer should')[0].strip()
            # Align prompt format with Qwen3-VL Technical Report (https://arxiv.org/pdf/2511.21631)
            prompt = (
                main_q + ' Output the point coordinates in JSON format.\n'
                'For example:\n'
                '[\n'
                '  {"point_2d": [x, y], "label": "point_1"}\n'
                ']\n'
            )
        else:
            prompt = raw_q

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([{'type': 'image', 'value': p} for p in tgt_path])
        else:
            msgs = [{'type': 'image', 'value': tgt_path}]
        msgs.append({'type': 'text', 'value': prompt})

        return msgs

    @staticmethod
    def point_in_polygon(x, y, poly):
        """
        Check if the point (x, y) lies within the polygon defined by a list of (x, y) tuples.
        Uses the ray-casting algorithm.
        """
        num = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, num + 1):
            p2x, p2y = poly[i % num]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def parse_prediction(self, pred_text: str, width: int, height: int) -> np.ndarray:
        """
        Parse raw model output into normalized coordinates of shape (N, 2),
        where each (x, y) is in [0, 1].

        Default: use Point2DParser (JSON/Python literal with 'point_2d'/'point'
        or fallback '(x, y)' / '(x0, y0, x1, y1)'), and return [0, 1]-normalized
        coordinates.

        Override this method if your model uses a different format.
        Always return normalized coordinates in [0, 1].
        """
        Point2DParser.log_hint(task_name=getattr(self, 'dataset_name', 'RoboSpatialHome'))
        return Point2DParser.parse(pred_text, width, height, output='norm')

    @staticmethod
    def evaluate_answer(ground_truth, generated_answer, img_width=None, img_height=None, parse_fn=None):
        """
        Evaluate a single answer.

        Returns:
            (is_correct, is_binary, parsed_answer, is_parsable)
        """
        gen_answer_raw = (generated_answer or '').strip()
        gen_answer_lower = gen_answer_raw.lower()
        gt_str = (ground_truth or '').strip()
        gt_lower = gt_str.lower()

        # 1) binary yes/no
        if gt_lower in ('yes', 'no'):
            is_binary = True
            is_gt_yes = (gt_lower == 'yes')
            is_parsable = len(gen_answer_raw) > 0
            if is_gt_yes:
                correct = gen_answer_lower.startswith('yes')
            else:
                correct = gen_answer_lower.startswith('no')
            return correct, is_binary, gen_answer_raw, is_parsable

        # 2) polygon case (GT is in [0, 1] normalized coords)
        is_binary = False
        parsed_answer = None
        is_parsable = False

        # 2.1 parse GT polygon in [0, 1]
        try:
            raw_poly = ast.literal_eval(gt_str)
        except Exception as e:
            print(f'[WARN] failed to parse ground_truth polygon: {ground_truth} ({e})')
            return False, is_binary, parsed_answer, is_parsable

        if not isinstance(raw_poly, (list, tuple)) or len(raw_poly) < 3:
            return False, is_binary, parsed_answer, is_parsable

        gt_polygon = []
        for pt in raw_poly:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                continue
            gx, gy = float(pt[0]), float(pt[1])
            # dataset convention: GT already in [0, 1]
            gt_polygon.append((gx, gy))

        if len(gt_polygon) < 3:
            return False, is_binary, parsed_answer, is_parsable

        # 2.2 parse predicted point as [0, 1]
        try:
            w = img_width if img_width is not None else 1
            h = img_height if img_height is not None else 1
            if parse_fn is None:
                pts = Point2DParser.parse(gen_answer_raw, int(w), int(h), output='norm')
            else:
                pts = parse_fn(gen_answer_raw, int(w), int(h))
        except Exception as e:
            print(f'[WARN] failed to parse prediction: {generated_answer} ({e})')
            return False, is_binary, parsed_answer, is_parsable

        if pts is None or len(pts) == 0:
            return False, is_binary, parsed_answer, is_parsable

        x, y = float(pts[0, 0]), float(pts[0, 1])
        parsed_answer = (x, y)
        is_parsable = True
        correct = RoboSpatialBench.point_in_polygon(x, y, gt_polygon)
        return correct, is_binary, parsed_answer, is_parsable

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.tools.files import build_eval_paths

        result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag='Point2D')

        data = load(eval_file)
        if 'index' in data.columns:
            data = data.sort_values(by='index')

        num_total = 0
        num_correct = 0
        illformed_responses = 0
        category_stats = {}

        is_correct_list = []
        is_binary_list = []
        parsed_answer_list = []
        is_parsable_list = []

        # main loop
        for _, row in data.iterrows():
            idx = int(row['index'])
            gt = row['answer']
            pred = row['prediction']
            category = row['category'] or 'unknown'

            assert category in self._task_category(), (
                f'Except RoboSpatial category to be one of {self._task_category()}, '
                f'but got {category}.'
            )

            if category not in category_stats:
                category_stats[category] = {'num_correct': 0, 'num_total': 0}
            category_stats[category]['num_total'] += 1
            num_total += 1

            img_path = os.path.join(self.img_root, f'{idx}.png')
            with Image.open(img_path) as img:
                img_w, img_h = img.size  # width, height

            # use parse_prediction as the default point parser
            correct, is_binary, parsed_answer, is_parsable = RoboSpatialBench.evaluate_answer(
                gt, pred, img_w, img_h, parse_fn=self.parse_prediction
            )

            if not is_parsable:
                illformed_responses += 1
            if correct:
                num_correct += 1
                category_stats[category]['num_correct'] += 1

            is_correct_list.append(bool(correct))
            is_binary_list.append(bool(is_binary))
            parsed_answer_list.append(None if parsed_answer is None else str(parsed_answer))
            is_parsable_list.append(bool(is_parsable))

        # attach evaluation fields
        data['is_correct'] = is_correct_list
        data['is_binary'] = is_binary_list
        data['parsed_answer'] = parsed_answer_list
        data['is_parsable'] = is_parsable_list

        accuracy = 100.0 * num_correct / num_total if num_total > 0 else 0.0

        # save detailed results
        dump(data, result_file)

        # optional xlsx export
        try:
            data.to_excel(xlsx_path, index=False)
        except Exception as e:
            print(f'[WARN] failed to save xlsx to {xlsx_path}: {e}')

        # aggregate accuracy to TSV
        rows = []

        rows.append(
            dict(
                dataset='RoboSpatial',
                category='ALL',
                accuracy=accuracy,
                num_correct=num_correct,
                num_total=num_total,
            )
        )

        for cat, stat in category_stats.items():
            cat_total = stat['num_total']
            cat_acc = 100.0 * stat['num_correct'] / cat_total if cat_total > 0 else 0.0
            rows.append(
                dict(
                    dataset='RoboSpatial',
                    category=cat,
                    accuracy=cat_acc,
                    num_correct=stat['num_correct'],
                    num_total=cat_total,
                )
            )

        acc_df = pd.DataFrame(rows)
        acc_df.to_csv(
            acc_tsv_path,
            sep='\t',
            index=False,
            float_format='%.2f',
        )

        print(
            f'[RoboSpatial] accuracy = {accuracy:.2f} '
            f'(num_correct={num_correct}, num_total={num_total}, '
            f'illformed_resp={illformed_responses})'
        )

        return {
            'accuracy': accuracy,
            'num_correct': num_correct,
            'num_total': num_total,
            'illformed_responses': illformed_responses,
            'category_stats': category_stats,
        }
