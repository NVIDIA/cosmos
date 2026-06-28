import re

# ############################ Idea Generation ##############################


def format_idea_data(idea_data):
    fields = [
        "Idea",
        "ImplementationSteps",
        "ImplementationOrder",
        "Dataset",
        "EvaluationMetrics",
        "ExpectedOutcome"
    ]

    formatted_text = ""
    for field in fields:
        if field in idea_data and idea_data[field]:
            formatted_text += f"{field}: {idea_data[field]}\n\n"

    return formatted_text.strip()


def get_context_from_data(data):
    context_fields = [
        "related_work",
        "challenge",
        "limitation",
        "motivation",
        "task_objective",
        "existing_solutions"
    ]
    context = ""
    for field in context_fields:
        if field in data and data[field]:
            context += f"{field}: {data[field]}\n\n"

    return context.strip()


def flip_evaluation_result(result):
    flipped = {}
    mapping = {
        "win_A": "win_B",
        "win_B": "win_A"
    }

    for key, value in result.items():
        if isinstance(value, dict) and "judgment" in value:
            flipped[key] = {
                "judgment": mapping.get(value["judgment"], value["judgment"]),
                "reason": value.get("reason", "")
            }
        else:
            flipped[key] = mapping.get(value, value)

    return flipped


def get_evaluation_prompt_modified(hypothesis_A, hypothesis_B, context=None):
    context_text = f"Context:\n{context}\n\n" if context else ""

    prompt = f"""
You are assisting researchers tasked with comparing TWO research hypotheses (Hypothesis A and Hypothesis B).
Your job is to evaluate both hypotheses across five separate dimensions defined below, and to choose a winner
(either Hypothesis A or Hypothesis B) for each dimension. Ties are NOT allowed — you MUST pick one winner per
dimension. Base your judgments on scientific principles and the provided context only.

##Background context:
{context_text}

##Hypothesis A:
{hypothesis_A}

##Hypothesis B:
{hypothesis_B}

##Definition of each dimension:
###1) Effectiveness
Which hypothesis is more likely to produce a successful experimental or empirical outcome in service of the stated
research objective? Evaluate the likelihood that, if implemented using standard practices in the relevant discipline,
the hypothesis will achieve the intended measurable result. Focus on mechanistic plausibility, causal logic, and
whether the hypothesis addresses the core problem directly.

###2)Novelty
Novelty: Which hypothesis presents more innovative or original approaches? Compare the similarity between the idea
and the related work and existing solutions in the background to assess its novelty. A lower similarity to the core
idea indicates greater novelty.

###3) Detailedness (Level of Specification)
Which hypothesis provides clearer, more actionable, and more complete specification of mechanisms, assumptions,
experimental steps, required variables, and dependencies? Detailedness rewards clarity that would enable a competent
researcher to design an experiment or implementation with minimal ambiguity.

###4) Feasibility
Which hypothesis presents a more realistic and implementable solution given current technological constraints?

###5) Overall
Considering the overall aspects together but emphasizing conceptual coherence and scientific grounding, which
hypothesis is superior overall? This is a synthesis judgment: prefer the hypothesis that is logically consistent,
grounded in accepted principles, avoids critical unstated assumptions or contradictions, and is most defensible as
a scientific proposition.

Unified constraints:
- Use only the provided context and widely accepted scientific principles in the relevant discipline. Do NOT invent
facts external to the context unless they are broadly standard domain knowledge.
- When a dimension explicitly says to ignore other factors (e.g., Novelty should ignore feasibility), strictly follow
that guidance for that dimension. When evaluating a certain dimension, it should focus on this dimension itself and
ignore the influence of other dimensions.
- Be concise but specific: for each dimension provide a short judgment line (exact format below) plus 1–3 sentences
of succinct reasoning grounded in the definitions above.
- Format must match exactly (case-insensitive for "Win A/Win B") and include a reason after "because".


##Output format (MUST FOLLOW EXACTLY)

Format your response exactly as follows:
Effectiveness: [Win A/Win B] because ...
Novelty: [Win A/Win B] because ...
Detailedness: [Win A/Win B] because ...
Feasibility: [Win A/Win B] because ...
Overall: [Win A/Win B] because ...
"""
    return prompt


def parse_evaluation_result(result):
    dimensions = ["effectiveness", "novelty", "detailedness", "feasibility", "overall"]
    parsed_results = {}
    all_valid = True

    for dim in dimensions:
        judgment = extract_win_lose(result, dim.capitalize())
        reason = extract_reason(result, dim.capitalize())

        if judgment is None:
            all_valid = False
            break

        parsed_results[dim] = {
            "judgment": judgment,
            "reason": reason
        }

    if not all_valid:
        return None

    return parsed_results


def extract_win_lose(result_text, dimension):
    pattern = rf"{dimension}\s*:\s*\[\s*(Win\s*A|Win\s*B)\s*\]"
    match = re.search(pattern, result_text, re.IGNORECASE)
    if match:
        judgment = match.group(1).strip().upper()
        if "WIN A" in judgment:
            return "win_A"
        else:
            return "win_B"

    backup_pattern = rf"{dimension}\s*:\s*(Win\s*A|Win\s*B)\s+"
    match = re.search(backup_pattern, result_text, re.IGNORECASE)
    if match:
        judgment = match.group(1).strip().upper()
        if "WIN A" in judgment:
            return "win_A"
        else:
            return "win_B"

    line_pattern = rf"{dimension}[^\n]*?(Win\s*A|Win\s*B)"
    match = re.search(line_pattern, result_text, re.IGNORECASE)
    if match:
        judgment = match.group(1).strip().upper()
        if "WIN A" in judgment:
            return "win_A"
        else:
            return "win_B"

    return None


def extract_reason(result_text, dimension):
    pattern = rf"{dimension}\s*:\s*\[[^\]]+\]\s*because\s*(.*?)(?=\n\w|$)"
    match = re.search(pattern, result_text, re.IGNORECASE | re.DOTALL)
    if match:
        reason = match.group(1).strip()
        return reason

    backup_pattern = rf"{dimension}\s*:[^\n]*?(because|due to|as|since)([^\n]+)"
    match = re.search(backup_pattern, result_text, re.IGNORECASE)
    if match:
        reason = match.group(2).strip()
        return reason

    fallback_pattern = rf"{dimension}\s*:[^\n]*(.*?)(?=\n\w+:|$)"
    match = re.search(fallback_pattern, result_text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
        reason = re.sub(r"\[(Win\s*A|Win\s*B)\]", "", text).strip()
        return reason

    return "No specific reason provided"


# ############################ Idea Generation ##############################


def mean(lst: list):
    assert len(lst) > 0, "list length must > 0"
    return sum(lst) / len(lst)


def show_results(results: list[dict], metric_name: str, category_name: str = None, precision: int = 2, scale=1):
    category_dict = {}
    for item in results:
        if category_name is None:
            item_category = 'default'
        else:
            item_category = item[category_name]
        if item_category not in category_dict:
            category_dict[item_category] = []

        item_metric = float(item[metric_name])
        category_dict[item_category].append(item_metric)

    for k, v in category_dict.items():
        category_dict[k] = round(mean(v) * scale, precision)

    if category_name is None:
        return category_dict['default']
    else:
        return category_dict
