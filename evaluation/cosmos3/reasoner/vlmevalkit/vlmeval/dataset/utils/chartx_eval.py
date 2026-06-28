
import json
import re

import Levenshtein
import numpy as np

# --- SCRM Logic ---


def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False


def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def extract_csv(text):
    text = text.strip()
    if '```csv' in text:
        text = text.split('```csv')[1].split('```')[0]
    elif '```' in text:
        text = text.split('```')[1].split('```')[0]
    return text.strip()


def csv2triples(csv, separator='\t', delimiter='\n'):
    csv = extract_csv(csv)
    lines = csv.strip().split(delimiter)
    if not lines:
        return []
    header = lines[0].split(separator)
    triples = []
    for line in lines[1:]:
        if not line:
            continue
        values = line.split(separator)
        entity = values[0]
        for i in range(1, len(values)):
            if i >= len(header):
                break
            temp = sorted([entity.strip(), header[i].strip()])
            value = values[i].strip()
            value = value.replace("%", "")
            value = value.replace("$", "")
            triples.append((temp[0].strip(), temp[1].strip(), value))
    return triples


def process_triplets(triplets):
    new_triplets = []
    for triplet in triplets:
        if len(triplet) > 2:
            if is_int(triplet[2]) or is_float(triplet[2]):
                triplet_temp = (
                    triplet[0].lower(),
                    triplet[1].lower(),
                    float(
                        triplet[2]))
            else:
                triplet_temp = (
                    triplet[0].lower(),
                    triplet[1].lower(),
                    triplet[2].lower())
        else:
            triplet_temp = (
                triplet[0].lower(),
                triplet[1].lower(),
                "no meaning")
        new_triplets.append(triplet_temp)
    return new_triplets


def intersection_with_tolerance(a, b, tol_word, tol_num):
    a = set(a)
    b = set(b)
    c = set()
    for elem1 in a:
        for elem2 in b:
            if is_float(elem1[-1]) and is_float(elem2[-1]):
                if (Levenshtein.distance(''.join(elem1[:-1]), ''.join(elem2[:-1])) <= tol_word) and (
                        abs(elem1[-1] - elem2[-1]) / (elem2[-1] + 0.000001) <= tol_num):
                    c.add(elem1)
            else:
                if (Levenshtein.distance(''.join([str(i) for i in elem1]), ''.join(
                        [str(j) for j in elem2])) <= tol_word):
                    c.add(elem1)
    return list(c)


def union_with_tolerance(a, b, tol_word, tol_num):
    c = set(a) | set(b)
    d = set(a) & set(b)
    e = intersection_with_tolerance(a, b, tol_word, tol_num)
    f = set(e)
    g = c - (f - d)
    return list(g)


def get_eval_list(
        pred_csv,
        label_csv,
        separator='\\t',
        delimiter='\\n',
        tol_word=3,
        tol_num=0.05):
    pred_triple_list = []
    for it in pred_csv:
        pred_triple_temp = csv2triples(
            it, separator=separator, delimiter=delimiter)
        pred_triple_pre = process_triplets(pred_triple_temp)
        pred_triple_list.append(pred_triple_pre)

    label_triple_list = []
    for it in label_csv:
        label_triple_temp = csv2triples(it, separator='\\t', delimiter='\\n')
        label_triple_pre = process_triplets(label_triple_temp)
        label_triple_list.append(label_triple_pre)

    sim_list = []
    for pred, label in zip(pred_triple_list, label_triple_list):
        intersection = intersection_with_tolerance(
            pred, label, tol_word=tol_word, tol_num=tol_num)
        union = union_with_tolerance(
            pred, label, tol_word=tol_word, tol_num=tol_num)
        if len(union) == 0:
            sim = 0
        else:
            sim = len(intersection) / len(union)
        sim_list.append(sim)
    return sim_list


def get_ap(
        predictions,
        labels,
        sim_threhold,
        tolerance,
        separator='\\t',
        delimiter='\\n',
        easy=1):
    if tolerance == 'strict':
        tol_word = 0
        if easy == 1:
            tol_num = 0
        else:
            tol_num = 0.1
    elif tolerance == 'slight':
        tol_word = 2
        if easy == 1:
            tol_num = 0.05
        else:
            tol_num = 0.3
    elif tolerance == 'high':
        tol_word = 5
        if easy == 1:
            tol_num = 0.1
        else:
            tol_num = 0.5
    sim_list = get_eval_list(
        predictions,
        labels,
        separator=separator,
        delimiter=delimiter,
        tol_word=tol_word,
        tol_num=tol_num)
    if len(sim_list) == 0:
        return 0
    ap = len([num for num in sim_list if num >= sim_threhold]) / len(sim_list)
    return ap


def chartx_scrm_eval(predictions, references, easy=1):
    s = "\\t"
    d = "\\n"

    map_strict = 0
    for sim_threhold in np.arange(0.5, 1, 0.05):
        map_temp_strict = get_ap(
            predictions,
            references,
            sim_threhold=sim_threhold,
            tolerance='strict',
            separator=s,
            delimiter=d,
            easy=easy)
        map_strict += map_temp_strict / 10

    ap_50_strict = get_ap(
        predictions,
        references,
        sim_threhold=0.5,
        tolerance='strict',
        separator=s,
        delimiter=d,
        easy=easy)

    return {
        'SCRM': map_strict,
        'AP50_Strict': ap_50_strict
    }

# --- GPT Evaluation Prompts & Logic ---


QA_EXAMPLES = [
    {
        "query": "<question> What was the incremental increase in revenue from 2020 to 2021? "
        "<groundtruth answer> 5 million $ <answer> 20\n</s>",
        "answer": "Answer: False"
    },
    {
        "query": "<question> What percentage of government spending was allocated to infrastructure in 2020? "
        "<groundtruth answer> 10% <answer> 14-4=10\n</s>",
        "answer": "Answer: True"
    },
    {
        "query": "<question> What is the total production of Wind Energy in the four months from January to "
        "April 2021? <groundtruth answer> 2300 MW <answer> The total production of Wind Energy in the four "
        "months from January to April 2021 is 2450 MW.",
        "answer": "Answer: True"
    },
    {
        "query": "<question> What is the total of manufactured goods for UK and Germany combined? "
        "<groundtruth answer> 5 <answer> Five",
        "answer": "Answer: True"
    },
]

QA_PREFIX = (
    "Given multiple question-answer pairs and the corresponding predictions, evaluate the correctness of "
    "predictions. The output should be only \"True\" or \"False\". Note that if the groundtruth answer is a "
    "numeric value with/without the unit, impose 5% error tolerance to the answer, e.g., the answer of 95 is "
    "marked as correct when groundtruth value is 100 million.\n"
    "Please provide your evaluation result at the very end in the format: 'Answer: True' or 'Answer: False'."
)
QA_SUFFIX = """
    User: {query}
    AI: """

CRITERIA = {  # noqa: E501
    'description': """
        You're an expert evaluating a model's description of a chart, based on its alignment with the
        ground truth and raw data. Score the model from 0 to 5 based on these criteria:
        0 points: Description irrelevant or shows no understanding of the chart or data.
        1 point: Refers to the chart but with largely incorrect details; minimal understanding.
        2 points: Some correct details, but key elements are wrong or missing; basic understanding with
        significant errors.
        3 points: Most details are correct; good understanding but with minor errors/omissions.
        4 points: All details are correct; very good understanding, minor improvements possible.
        5 points: Comprehensive, accurate description; excellent understanding with no errors; clear and
        detailed, perfect as a standalone explanation.
        Score the model's description on this scale, providing a single value without providing any reasons.
        Please provide your evaluation score at the very end in the format: 'Score: <score>'.
        """,
    'summary': """
        You're an expert evaluating a model's summarization of a chart, based on its alignment with the
        ground truth and raw data. Score the model from 0 to 5 based on these criteria:
        0 points: The summary is irrelevant or shows no understanding of the original text, failing to
        address the core content or theme.
        1 point: While referencing the original text, the summary contains predominantly incorrect details
        or interpretations, showing minimal understanding and significant inaccuracies.
        2 points: The summary captures some correct details, indicating a basic understanding. However, it
        misses key elements or includes major inaccuracies, leading to a flawed interpretation of the text.
        3 points: Most details in the summary are accurate, reflecting a good understanding of the original
        text. Minor errors or omissions are present but don't significantly impact the overall accuracy or
        comprehension.
        4 points: The summary accurately represents all main ideas and important details of the original text.
        It shows a very good understanding, with minor room for improvement in clarity, conciseness, or structure.
        5 points: This represents a comprehensive and accurate summary, perfectly encapsulating all essential
        aspects of the original text. It demonstrates excellent understanding, is error-free, clear, concise,
        well-structured, and serves as an excellent standalone representation of the original content.
        Score the model's summarization on this scale, providing a single value without providing any reasons.
        Please provide your evaluation score at the very end in the format: 'Score: <score>'.
        """,
    'redrawing': """
        You're an expert evaluating a redrawing code of a chart, based on its alignment with the ground truth
        and raw data. Score the code from 0 to 5 based on these criteria:
        0 points: Completely Irrelevant or Non-functional Code. Demonstrates no understanding of the chart
        structure or data. Code is inexecutable or produces a completely unrelated chart.
        1 point: Attempted Redraw with Major Discrepancies. Partial understanding of basic chart structure
        or data. Generated chart has very little in common with the original.
        2 points: Partially Correct Code with Key Errors or Omissions. Basic understanding of chart structure
        or data but with significant errors. Chart somewhat resembles the original but key inaccuracies are evident.
        3 points: Mostly Accurate; Good Understanding with Minor Errors/Omissions. Accurately reflects most of
        the chart's structure and data. Generated chart is similar to the original but has a few minor errors.
        4 points: Highly Accurate; Very Good Understanding with Minor Room for Improvement. Accurately presents
        the chart's structure and data in full. Generated chart is very close to the original, with negligible
        differences.
        5 points: Comprehensive, Accurate Code; Excellent Understanding, No Errors. Perfectly replicates all
        details and data of the chart. Generated chart is indistinguishable from the original, flawless.
        Score the redrawing code on this scale, providing a single value without providing ant reasons.
        Please provide your evaluation score at the very end in the format: 'Score: <score>'.
        """
}


def build_qa_prompt(query):
    # Construct prompt manually without langchain dependency inside this loop to keep things simple and portable
    # Following FewShotPromptTemplate logic

    prompt = QA_PREFIX + "\n\n"
    for ex in QA_EXAMPLES:
        prompt += f"    User: {ex['query']}\n    AI: {ex['answer']}\n\n"

    prompt += QA_SUFFIX.format(query=query)
    return prompt


def ChartX_auxeval(model, item):
    category = item.get('category', 'unknown').lower()

    # 1. Structure: handled outside or skipped here
    if category == 'structure':
        return {
            'score': 0,
            'log': 'Structure extraction should be evaluated with SCRM.'}

    # 2. QA
    if category == 'qa':
        # Construct query
        q = item['question']
        gt = item['answer']
        pred = item['prediction']
        query_str = f"<question> {q} <groundtruth answer> {gt} <answer> {pred}"
        prompt = build_qa_prompt(query_str)

        # Call model
        response = model.generate(prompt)

        score = 0
        try:
            hits = re.findall(r'Answer:\s*(True|False)', response, re.IGNORECASE)
            if hits:
                score = 1 if hits[-1].lower() == 'true' else 0
            else:
                # Fallback strictly to first occurrence if formatting failed
                response_lower = response.lower()
                hits = re.findall(r'\b(true|false)\b', response_lower)
                if hits:
                    score = 1 if hits[0] == 'true' else 0
        except Exception:
            pass

        return {'score': score, 'log': response}

    # 3. Desc / Summary / Redrawing
    # Mapping
    crit_key = None
    if 'description' in category:
        crit_key = 'description'
    elif 'summary' in category:
        crit_key = 'summary'
    elif 'redrawing' in category:
        crit_key = 'redrawing'

    if crit_key:
        criterion = CRITERIA[crit_key]

        # Metadata from combined metadata column (JSON)
        metadata_str = item.get('metadata', '{}')
        try:
            metadata = json.loads(metadata_str)
        except Exception:
            metadata = {}

        title = metadata.get('title', 'Unknown Title')
        chart_type = metadata.get('chart_type', 'Unknown Type')
        csv_gt = metadata.get('csv', 'No data')

        pred_ans = item['prediction']

        # Construct content exactly as official script for high fidelity
        # Structure often: "data: {csv} <title> {title} <type>
        # {chart_type}\n{pred}"
        content = f"data: {csv_gt} <title> {title} <type> {chart_type}\n{pred_ans}"

        prompt = [
            {"role": "user", "content": criterion},
            {"role": "user", "content": content}
        ]

        # Model generate usually takes list of strings or list of dicts (messages)
        # VLMEvalKit model wrapper handles conversation list.
        response = model.generate(prompt)

        # Extract score 0-5
        score = 0
        try:
            hits = re.findall(r'Score:\s*([0-5])', response, re.IGNORECASE)
            if hits:
                score = int(hits[-1])
            else:
                # Fallback
                hits = re.findall(r'[0-5]', response)
                if hits:
                    score = int(hits[-1])
        except Exception:
            pass

        return {'score': score, 'log': response}

    return {'score': 0, 'log': 'Unknown category'}
