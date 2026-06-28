import json
import math
import re
import signal
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vlmeval.smp import load


def rating_func(data_path):
    df = load(data_path)

    task_mode_fields = {
        "long_acc": ["acc", "oboa", "mae", "rmse"],
        "ref_acc": ["acc", "oboa", "mae", "rmse"],
        "clue_acc": ["wcs", "ifa"],
    }

    rating = {}

    for task_mode, fields in task_mode_fields.items():
        sub_df = df[df["task_mode"] == task_mode]
        for field in fields:
            values = sub_df[field]
            if field == "rmse":
                # RMSE: sqrt(mean(x^2))
                rmse_val = np.sqrt(values.mean())
                rating[f"{task_mode}/rmse"] = round(rmse_val, 4)
            else:
                rating[f"{task_mode}/{field}"] = round(values.mean(), 4)

    return rating


def get_timestampes(frame_indices, fps):
    seconds = list(map(lambda x: str(round(x / fps, 4)), frame_indices))
    timestamps = ", ".join(seconds)
    return "A total of {frame_num} frames are sampled. Their corresponding timestamps are:\n\n{timestamps}\n\n".format(
        frame_num=len(frame_indices), timestamps=timestamps
    )


def time_str_to_seconds(time_str: str) -> float:
    time_str = time_str.strip()
    if '.' in time_str:
        time_main, milliseconds = time_str.split('.')
        milliseconds = float(f"0.{milliseconds}")
    else:
        time_main = time_str
        milliseconds = 0.0

    parts = list(map(int, time_main.split(":")))

    if len(parts) == 2:
        minutes, seconds = parts
        total_seconds = minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    return total_seconds + milliseconds


def extract_outer_json(text):
    stack = []
    start_idx = None
    opening = {'{': '}', '[': ']'}
    closing = {'}': '{', ']': '['}

    for i, char in enumerate(text):
        if char in opening:
            if not stack:
                start_idx = i  # 最外层起点
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == closing[char]:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx:i + 1]
                    try:
                        return json.dumps(json.loads(candidate))
                    except json.JSONDecodeError:
                        continue  # 尝试下一个 JSON 块
    return None


def compute_tiou(t1, t2):
    """Temporal IoU"""
    inter_start = max(t1[0], t2[0])
    inter_end = min(t1[1], t2[1])
    inter = max(0.0, inter_end - inter_start)
    union = max(t1[1], t2[1]) - min(t1[0], t2[0])
    return inter / union if union > 0 else 0.0


def compute_sIoU(box1, box2):
    """
    Complete IoU (sIoU) between two bounding boxes.
    Args:
        box1 (list or np.array): [x1, y1, x2, y2] of ground truth box
        box2 (list or np.array): [x1, y1, x2, y2] of predicted box

    Returns:
        IoU (float): The IoU score between the two boxes.
    """

    # Ensure the coordinates are ordered: [min_x, min_y, max_x, max_y]
    box1 = np.array([min(box1[0], box1[2]), min(box1[1], box1[3]),
                     max(box1[0], box1[2]), max(box1[1], box1[3])])
    box2 = np.array([min(box2[0], box2[2]), min(box2[1], box2[3]),
                     max(box2[0], box2[2]), max(box2[1], box2[3])])

    # Compute the intersection area
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute areas of the individual boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union = area1 + area2 - inter_area
    iou = inter_area / union if union > 0 else 0.0

    return iou


def greedy_matching(gt_instances, pred_instances, iou_func):
    """Greedy matching based on maximum IoU"""
    unmatched_gt = set(range(len(gt_instances)))
    unmatched_pred = set(range(len(pred_instances)))
    matches = []

    while unmatched_gt and unmatched_pred:
        max_iou = -1
        best_match = None
        for gt_idx in unmatched_gt:
            for pred_idx in unmatched_pred:
                iou = iou_func(gt_instances[gt_idx], pred_instances[pred_idx])
                if iou > max_iou:
                    max_iou = iou
                    best_match = (gt_idx, pred_idx)

        if best_match:
            gt_idx, pred_idx = best_match
            matches.append((gt_idx, pred_idx))
            unmatched_gt.remove(gt_idx)
            unmatched_pred.remove(pred_idx)

    return matches


def compute_cluster_pair_wcs(gt, pred, iou_type):
    if iou_type == 'tIoU':
        loc_sum = 0.0
        for g in gt:
            loc_sum += max([compute_tiou(g, p) for p in pred] or [0.0])
        loc_acc = loc_sum / len(gt) if gt else 0.0
        count_penalty = 1.0 - abs(len(pred) - len(gt)) / max(len(gt), 1)
        # count_penalty = 1.0
        return math.sqrt(loc_acc * max(0, count_penalty))

    elif iou_type == 'sIoU':
        # group by frame index
        from collections import defaultdict
        gt_by_f = defaultdict(list)
        pred_by_f = defaultdict(list)
        for f, box in gt:
            gt_by_f[f].append(box)
        for f, box in pred:
            pred_by_f[f].append(box)

        all_f = set(gt_by_f) | set(pred_by_f)
        wcs = 0.0
        for f in all_f:
            gt_f = gt_by_f.get(f, [])
            pred_f = pred_by_f.get(f, [])
            matches = greedy_matching(gt_f, pred_f, compute_sIoU)
            loc_sum = sum([compute_sIoU(gt_f[i], pred_f[j]) for i, j in matches])
            loc_acc = loc_sum / len(gt_f) if gt_f else 0.0
            count_penalty = 1.0 - abs(len(pred_f) - len(gt_f)) / max(len(gt_f), 1)
            # count_penalty = 1.0
            wcs += math.sqrt(loc_acc * max(0, count_penalty))
        return wcs / max(len(all_f), 1)

    else:
        raise ValueError("Unsupported iou_type")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function execution exceeded the time limit.")


def compute_wcs_unlabeled(gt_clusters, pred_clusters, iou_type='tIoU',
                          timeout=10):  # 主要是给attribute用的，但是object和event视作一个cluster也能用
    from scipy.optimize import linear_sum_assignment

    # Set the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set the alarm to go off in 'timeout' seconds

    try:
        # Original function logic
        K = len(gt_clusters)
        M = len(pred_clusters)

        # Build cost matrix (we want max score → min cost)
        score_matrix = np.zeros((K, M))
        for i in range(K):
            for j in range(M):
                score_matrix[i, j] = compute_cluster_pair_wcs(gt_clusters[i], pred_clusters[j], iou_type)

        cost_matrix = -score_matrix  # maximize score → minimize cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_scores = [score_matrix[i, j] for i, j in zip(row_ind, col_ind)]

        # WCS = average over gt clusters (including unmatched = 0)
        total_wcs = sum(matched_scores)
        return total_wcs / K

    except TimeoutException:
        print(gt_clusters, pred_clusters)
        print("Function execution exceeded the time limit.")
        return None  # or you can return some default value to indicate timeout

    finally:
        signal.alarm(0)  # Cancel the alarm after the function completes or times out


def post_process(response, right_answer, task_mode, category):
    from word2number import w2n
    if task_mode in ["long_acc", "ref_acc"]:
        result = {"acc": 0, "oboa": 0, "mae": 0, "rmse": 0}
        if response:
            try:
                pred = w2n.word_to_num(response)
            except Exception:
                pred = 0
            if abs(float(right_answer) - float(pred)) <= 1e-5:
                result["acc"] = 1

            if abs(float(right_answer) - float(pred)) <= 1:
                result["oboa"] = 1

            if abs(float(right_answer) - float(pred)) <= max(2 * float(right_answer), 100):
                result["mae"] = abs(float(right_answer) - float(pred))
                result["rmse"] = abs(float(right_answer) - float(pred)) ** 2
            else:
                result["mae"] = abs(float(right_answer) * 2)
                result["rmse"] = abs(float(right_answer) * 2) ** 2
    elif task_mode == "clue_acc":
        result = {"wcs": 0, "ifa": 0}
        if response:
            clues = json.loads(right_answer)
            content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else response.strip()
            j = None
            try:
                try:
                    j = json.loads(student_answer)
                except Exception:
                    j = json.loads(extract_outer_json(student_answer))
            except Exception:
                pass
            if j is not None:
                try:
                    if category == "event":
                        pred = []
                        for e in j:

                            if isinstance(e[0], str) and isinstance(e[1], str) and ":" in e[0] and ":" in e[1]:
                                pred.append([time_str_to_seconds(e[0]), time_str_to_seconds(e[1])])
                            else:
                                pred.append([float(e[0].split(" ")[0]) if isinstance(e[0], str) else e[0],
                                             float(e[1].split(" ")[0]) if isinstance(e[1], str) else e[1]])
                        gt = []
                        for e in clues:
                            gt.append([float(e['start']), float(e['end'])])

                        result["wcs"] = compute_wcs_unlabeled([gt], [pred], "tIoU")
                        result["ifa"] = 1
                    elif category == "object":
                        gt = []
                        clue_timestamp_list = []
                        for clue in clues:
                            if clue["timestamp"] not in clue_timestamp_list:
                                clue_timestamp_list.append(clue["timestamp"])
                        for clue in clues:
                            gt.append((clue_timestamp_list.index(clue["timestamp"]), clue['bbox']))
                        pred = []
                        for key in j.keys():
                            if "Frame" not in key:
                                continue
                            idx = int(key.replace("Frame", "")) - 1
                            if len(j[key]) == 0:
                                continue
                            if isinstance(j[key][0], list) and len(j[key][0]) == 4:
                                for e in j[key]:
                                    if isinstance(e, list) and len(e) == 4:
                                        pred.append((idx, e))
                            elif isinstance(j[key][0], list) and len(j[key][0]) == 2:
                                for ii in range(int(len(j[key]) // 2)):
                                    if isinstance(j[key][ii * 2], list) and len(j[key][ii * 2]) == 2 and isinstance(
                                            j[key][ii * 2 + 1], list) and len(j[key][ii * 2 + 1]) == 2:
                                        pred.append((idx, [j[key][ii * 2][0], j[key][ii * 2][1], j[key][ii * 2 + 1][0],
                                                           j[key][ii * 2 + 1][1]]))
                        result["wcs"] = compute_wcs_unlabeled([gt], [pred], "sIoU")
                        result["ifa"] = 1
                    elif category == "attribute":
                        gt = []
                        clue_timestamp_list = []
                        for clue_ in clues:
                            for clue in clue_:
                                if clue["timestamp"] not in clue_timestamp_list:
                                    clue_timestamp_list.append(clue["timestamp"])
                        for clue_ in clues:
                            gt_ = []
                            for clue in clue_:
                                gt_.append((clue_timestamp_list.index(clue["timestamp"]), clue['bbox']))
                            gt.append(gt_)
                        pred = {}
                        for key in j.keys():
                            if "Frame" not in key:
                                continue
                            idx = int(key.replace("Frame", "")) - 1
                            for e in j[key]:
                                if e['label'] not in pred.keys():
                                    pred[e['label']] = []
                                if 'bbox' in e:
                                    if isinstance(e['bbox'], list) and len(e['bbox']) == 4:
                                        pred[e['label']].append((idx, e['bbox']))
                                if 'bbox_2d' in e:
                                    if isinstance(e['bbox_2d'], list) and len(e['bbox_2d']) == 4:
                                        pred[e['label']].append((idx, e['bbox_2d']))
                        pred_list = [pred[key] for key in pred]
                        result["wcs"] = compute_wcs_unlabeled(gt, pred_list, "sIoU")
                        result["ifa"] = 1
                except Exception:
                    pass

    return result


def get_chunk_number(filename):
    try:
        num = filename.split("chunk_")[1].split(".zip")[0]
        return int(num)
    except Exception:
        return float('inf')


def auto_merge_and_unzip_parts(target_dir, extract_dir, zip_prefix=None):
    target_dir = Path(target_dir)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 匹配 zip 分卷：例如 video_chunk_001.zip.part000
    part_files = sorted(target_dir.glob("*.zip.part*"))
    groups = {}

    # 分组：根据前缀提取 group 名（即 zip 文件名）
    for part_file in part_files:
        match = re.match(r"(.*\.zip)\.part\d+$", part_file.name)
        if match:
            zip_name = match.group(1)
            if zip_prefix is None or Path(zip_name).stem.startswith(zip_prefix):
                groups.setdefault(zip_name, []).append(part_file)

    if not groups:
        print(f"No matching zip parts found with prefix: {zip_prefix}")
        return

    # 合并每一组分卷 -> 解压
    for zip_name, parts in tqdm(groups.items(), desc="Merging and unzipping"):
        parts = sorted(parts, key=lambda p: int(p.name.split("part")[-1]))
        zip_path = target_dir / zip_name

        # 合并分卷
        with open(zip_path, 'wb') as outfile:
            for part in parts:
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())

        # 解压合并后的 zip 文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 删除合并后的 zip 文件（可注释）
        zip_path.unlink()


def unzip_hf_zip(target_dir):
    target_dir = Path(target_dir)

    videos_dir = target_dir / "cg_videos_720p"
    ref_videos_dir = target_dir / "ref_videos"

    if videos_dir.exists() and ref_videos_dir.exists():
        print("all target dirs exist, skip.")
        return

    videos_dir.mkdir(parents=True, exist_ok=True)

    auto_merge_and_unzip_parts(target_dir, ref_videos_dir, zip_prefix="ref_videos")
    auto_merge_and_unzip_parts(target_dir, videos_dir, zip_prefix="videos")

    print("sucessfully unzip all files.")
