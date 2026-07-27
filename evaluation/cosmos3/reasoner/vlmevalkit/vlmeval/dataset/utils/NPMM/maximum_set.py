def validation(graph, answer):
    """
    Validate if the given answer is a valid maximum independent set

    Parameters:
    graph: JSON graph data from the question (adjacency list format)
    answer: String answer containing the proposed independent set after "Answer:"

    Returns:
    tuple: (is_invalid, size, message)
            is_invalid - True if invalid, False if valid
            size - size of the independent set (-1 if invalid)
            message - validation feedback
    """
    # Parse answer string
    if "Answer:" not in answer:
        return True, -1, "Invalid answer format: missing 'Answer:' prefix"

    answer_part = answer.split("Answer:")[-1].split('\n')[0].strip()

    try:
        # Handle different answer formats
        if answer_part.startswith('[') and answer_part.endswith(']'):
            independent_set = eval(answer_part)
        else:
            # Try to parse other formats (comma separated, etc.)
            independent_set = [int(x.strip()) for x in answer_part.replace('[', '').replace(']', '').split(',')]

        if not isinstance(independent_set, list):
            return True, -1, "Answer must be a list of vertices"
    except Exception:
        return True, -1, "Could not parse the independent set from answer"

    # Convert all node keys to strings for consistency
    graph_nodes = set(graph.keys())

    # Check all nodes exist in graph
    invalid_nodes = [str(node) for node in independent_set if str(node) not in graph_nodes]
    if invalid_nodes:
        return True, -1, f"Nodes not in graph: {', '.join(invalid_nodes)}"

    # Check for duplicate nodes
    if len(independent_set) != len(set(independent_set)):
        return True, -1, "Duplicate nodes found in the set"

    # Check for independence (no two nodes are adjacent)
    for i in range(len(independent_set)):
        for j in range(i + 1, len(independent_set)):
            node1 = str(independent_set[i])
            node2 = str(independent_set[j])
            if node2 in graph.get(node1, {}):
                return True, -1, f"Nodes {node1} and {node2} are adjacent (violates independence)"

    # Check for maximality (not necessarily maximum, just not obviously improvable)
    # Note: This doesn't verify if it's the absolute maximum, just if it's valid
    neighbor_nodes = set()
    for node in independent_set:
        neighbors = graph.get(str(node), [])
        if isinstance(neighbors, dict):
            neighbor_nodes.update(neighbors.keys())
        else:
            neighbor_nodes.update(str(n) for n in neighbors)

    remaining_nodes = graph_nodes - set(map(str, independent_set)) - neighbor_nodes
    if remaining_nodes:
        return False, len(independent_set), f"Valid independent set (size {len(independent_set)}), but possibly not maximal - could potentially add nodes: {remaining_nodes}"  # noqa: E501

    return False, len(independent_set), f"Valid maximal independent set (size {len(independent_set)})"
