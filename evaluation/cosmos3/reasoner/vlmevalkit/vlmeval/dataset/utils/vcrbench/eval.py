import ast
import copy
import json


def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def save_json(data, file_path, indent=4):
    with open(file_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding="utf-8") as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")


def calculate_time_iou(interval1, interval2):

    start1, end1 = interval1
    start2, end2 = interval2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start

    if union == 0:
        return 0
    iou = intersection / union
    return iou


def is_valid_time_interval(interval_str):

    try:
        interval = ast.literal_eval(interval_str)
        if isinstance(interval, list) and len(interval) == 2:
            if all(isinstance(x, (int, float)) for x in interval):
                return True
        return False
    except (ValueError, SyntaxError):
        return False


def is_valid_space_interval(s):
    if not (s.startswith('[') and s.endswith(']')):
        return False
    content = s[1:-1]
    parts = content.split(',')
    if len(parts) != 4:
        return False
    for part in parts:
        try:
            int(part.strip())
        except ValueError:
            return False
    return True


def string_to_list(s):
    content = s[1:-1]
    return [int(part.strip()) for part in content.split(',')]


def extract_json_between_backticks(s):
    # pattern = r'```json\n(.*?)```'
    # match = re.search(pattern, s, re.DOTALL)
    # if not match:
    #     raise ValueError("No JSON content wrapped by ``` was found.")
    # json_str = match.group(1).strip()
    json_str = s

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        raise ValueError(f"Extracted content is not valid JSON: {e}")


def calculate_recall(json_object):

    stats = {
        "Video Description Steps": {"Matched": 0, "Unmatched": 0},
        "Logical Inference Steps": {"Matched": 0, "Unmatched": 0},
        "Background Review Steps": {"Matched": 0, "Unmatched": 0}
    }
    for item in json_object:
        step_type = item["step_type"]
        judgement = item["judgment"]
        stats[step_type][judgement] += 1

    return stats


def calculate_space_iou(box1, box2):

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area
    return iou


def calculate_precision(json_object):
    stats = {
        "Video Description Steps": {"Matched": 0, "Wrong": 0, "Redundant": 0},
        "Logical Inference Steps": {"Matched": 0, "Wrong": 0, "Redundant": 0},
        "Background Review Steps": {"Matched": 0, "Wrong": 0, "Redundant": 0}
    }
    for item in json_object:
        step_type = item["step_type"]
        judgement = item["judgment"]
        stats[step_type][judgement] += 1

    return stats


def recall(item):
    processed_item = copy.deepcopy(item)

    json_object = json.loads(extract_json_between_backticks(processed_item['recall_eval']))
    stats = calculate_recall(json_object)

    Video_recall = "" if (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Unmatched']) == 0 else stats['Video Description Steps']['Matched'] / (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Unmatched'])  # noqa: E501

    logic_recall = "" if (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Unmatched']) == 0 else stats['Logical Inference Steps']['Matched'] / (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Unmatched'])  # noqa: E501

    background_recall = "" if (stats['Background Review Steps']['Matched'] + stats['Background Review Steps']['Unmatched']) == 0 else stats['Background Review Steps']['Matched'] / (stats['Background Review Steps']['Matched'] + stats['Background Review Steps']['Unmatched'])  # noqa: E501

    processed_item['Video_recall'] = Video_recall
    processed_item['logic_recall'] = logic_recall
    processed_item['background_recall'] = background_recall

    total_matched = (
        stats['Video Description Steps']['Matched']
        + stats['Logical Inference Steps']['Matched']
    )

    total_steps = (
        (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Unmatched'])
        + (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Unmatched'])
    )

    if total_steps == 0:
        overall_recall = ""
    else:
        overall_recall = total_matched / total_steps

    processed_item['overall_recall'] = overall_recall

    return processed_item


def precision(item):
    processed_item = copy.deepcopy(item)

    json_object = json.loads(extract_json_between_backticks(processed_item['precision_eval']))
    stats = calculate_precision(json_object)

    Video_precision = "" if (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Wrong']) == 0 else stats['Video Description Steps']['Matched'] / (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Wrong'])  # noqa: E501

    logic_precision = "" if (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Wrong']) == 0 else stats['Logical Inference Steps']['Matched'] / (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Wrong'])  # noqa: E501

    background_precision = "" if (stats['Background Review Steps']['Matched'] + stats['Background Review Steps']['Wrong']) == 0 else stats['Background Review Steps']['Matched'] / (stats['Background Review Steps']['Matched'] + stats['Background Review Steps']['Wrong'])  # noqa: E501

    processed_item['Video_precision'] = Video_precision
    processed_item['logic_precision'] = logic_precision
    processed_item['background_precision'] = background_precision

    total_matched = (
        stats['Video Description Steps']['Matched']
        + stats['Logical Inference Steps']['Matched']
    )

    total_wrong = (
        stats['Video Description Steps']['Wrong']
        + stats['Logical Inference Steps']['Wrong']
    )

    if (total_matched + total_wrong) == 0:
        overall_precision = ""
    else:
        overall_precision = total_matched / (total_matched + total_wrong)

    processed_item['overall_precision'] = overall_precision

    total_step_num = 0
    for counts in stats.values():
        total_step_num += sum(counts.values())

    redundant_num = 0
    for counts in stats.values():
        redundant_num += counts['Redundant']

    efficiency = (total_step_num - redundant_num) / total_step_num
    processed_item['efficiency'] = efficiency
    return processed_item
