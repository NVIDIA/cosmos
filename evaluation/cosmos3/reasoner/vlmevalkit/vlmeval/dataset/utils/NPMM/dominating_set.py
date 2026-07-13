import re


def validation(graph, answer):
    """
    验证给定的答案是否是一个有效的支配集方案
    参数:
    graph: 图的表示，包含邻接表
    answer: 字符串答案，包含"Answer："后的顶点列表
    返回:
    (bool, int, str): 布尔值表示答案是否有效，True代表无效，False代表有效，
                       整数表示支配集的大小（越小越好）。如果答案无效，返回一个比任何可能解都大的值。
                       字符串提供错误或成功信息。
    """
    num_vertices = len(graph)
    penalty_value = num_vertices + 1
    if "Answer:" in answer:
        vertices_str = answer.split("Answer:")[-1].strip()
    else:
        return True, penalty_value, "invalid answer: no 'Answer:' in answer"

    # 清理字符串
    vertices_str = vertices_str.strip().replace("'", "").replace('"', '')
    # 提取顶点列表
    pattern = r'\[([^\]]*)\]'
    match = re.search(pattern, vertices_str)
    if not match:
        return True, penalty_value, "vertices must be in list format"
    vertices_content = match.group(1)
    # 解析顶点
    try:
        if vertices_content.strip() == '':
            # 空列表可能对于非常小的图有效
            vertices = []
        else:
            vertices = [int(x.strip()) for x in vertices_content.split(',') if x.strip() != '']
        # 去除重复并排序
        vertices = sorted(list(set(vertices)))
    except Exception as e:
        return True, penalty_value, f"vertices must be integers: {str(e)}"
    # 检查所有顶点是否有效
    for v in vertices:
        if v < 0 or v >= num_vertices:
            return True, penalty_value, f"invalid vertex {v}: must be in range [0, {num_vertices - 1}]"
    # 检查是否是有效的支配集
    dominating_set = set(vertices)
    # 检查每个顶点是否在支配集中或与支配集中的顶点相邻
    for node in range(num_vertices):
        node_str = str(node)
        # 检查节点是否在支配集中
        if node in dominating_set:
            continue
        # 检查节点是否与支配集中的任何顶点相邻
        is_dominated = False
        neighbors = graph.get(node_str, [])
        for neighbor in neighbors:
            if isinstance(neighbor, str):
                neighbor = int(neighbor)
            if neighbor in dominating_set:
                is_dominated = True
                break
        if not is_dominated:
            return True, penalty_value, f"invalid dominating set: vertex {node} is not dominated"
    # 有效的支配集
    dominating_size = len(vertices)
    # 特殊情况：空支配集
    if dominating_size == 0:
        # 空图
        if num_vertices == 0:
            return False, 0, "valid dominating set: empty set for empty graph"
        else:
            # 对于非空图，空集不是有效的支配集
            return True, penalty_value, "invalid dominating set: empty set cannot dominate non-empty graph"
    return False, dominating_size, f"valid dominating set with {dominating_size} vertices"
