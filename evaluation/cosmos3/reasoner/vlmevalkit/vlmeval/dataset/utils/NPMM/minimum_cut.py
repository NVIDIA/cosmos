def validation(graph, answer):
    """
    Validate the given answer for the Balanced Minimum Bisection Problem.

    Parameters:
    graph: dict - adjacency list of the graph (keys and neighbor keys are strings)
    answer: str - answer string containing the partition after 'Answer:'

    Returns:
    (bool, float, str):
        - bool: True if invalid, False if valid
        - float: cut weight if valid, else a large penalty value
        - str: message explaining result
    """
    # Compute a large penalty value (sum of all edge weights)
    total_weight_sum = 0
    for u_str, neighbors in graph.items():
        u = int(u_str)
        for v_str, weight in neighbors.items():
            v = int(v_str)
            if u < v:
                total_weight_sum += weight
    if total_weight_sum == 0:
        total_weight_sum = 1e9

    # Extract answer content
    if "Answer:" in answer:
        cut_str = answer.split("Answer:")[-1].strip()
    else:
        return True, total_weight_sum, "invalid answer: no 'Answer:' in answer"

    cut_str = cut_str.strip()
    if not (cut_str.startswith('[') and cut_str.endswith(']')):
        return True, total_weight_sum, "answer should be in format [[subset1], [subset2]]"

    try:
        import ast
        subsets = ast.literal_eval(cut_str)
        if not isinstance(subsets, list) or len(subsets) != 2:
            return True, total_weight_sum, "answer should contain exactly two subsets"
        subset1, subset2 = subsets
        if not (isinstance(subset1, list) and isinstance(subset2, list)):
            return True, total_weight_sum, "each subset should be a list of nodes"
    except Exception:
        return True, total_weight_sum, "invalid answer format"

    # Convert to sets
    set1, set2 = set(subset1), set(subset2)

    # 1. Check disjointness
    if set1 & set2:
        return True, total_weight_sum, f"subsets are not disjoint: common nodes {set1 & set2}"

    # 2. Check coverage
    all_nodes = set(int(node) for node in graph.keys())
    union = set1 | set2
    if union != all_nodes:
        missing = all_nodes - union
        extra = union - all_nodes
        errors = []
        if missing:
            errors.append(f"missing nodes: {missing}")
        if extra:
            errors.append(f"extra nodes: {extra}")
        return True, total_weight_sum, "; ".join(errors)

    # 3. Check balance constraint
    n = len(all_nodes)
    diff = abs(len(set1) - len(set2))
    if not (diff == 0 or (n % 2 == 1 and diff == 1)):
        return True, total_weight_sum, (
            f"balance constraint violated: subset sizes {len(set1)} and {len(set2)} "
            f"for total nodes {n}"
        )

    # 4. Compute cut weight
    cut_weight = 0
    for node1 in set1:
        for node2 in set2:
            try:
                weight = graph[str(node1)].get(str(node2), 0)
                cut_weight += weight
            except KeyError:
                return True, total_weight_sum, f"invalid node in subset: {node1} or {node2}"

    return False, cut_weight, f"Cut weight: {cut_weight}"
