import re


def validation(graph, answer):
    """
    验证给定的答案是否是一个有效的顶点覆盖方案

    参数:
    graph: 图的表示，包含邻接表
    answer: 字符串答案，包含"Answer："后的顶点列表

    返回:
    (bool, int, str): 布尔值表示答案是否有效，True代表无效，False代表有效，
                       整数表示顶点覆盖的大小（越小越好）。如果答案无效，返回一个比任何可能解都大的值。
                       字符串提供错误或成功信息。
    """
    num_vertices = len(graph)
    # 定义一个惩罚值，用于无效答案
    penalty_value = num_vertices + 1

    # 解析answer字符串，提取"Answer:"后面的内容
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
            # 空列表可能对于没有边的图有效
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

    # 检查是否是有效的顶点覆盖
    vertex_cover = set(vertices)

    # 检查所有边
    for node_str, neighbors in graph.items():
        node = int(node_str)

        for neighbor in neighbors:
            if isinstance(neighbor, str):
                neighbor = int(neighbor)

            # 只检查每条边一次（避免同时检查 (u,v) 和 (v,u)）
            if node < neighbor:
                # 检查这条边是否被覆盖
                if node not in vertex_cover and neighbor not in vertex_cover:
                    # 边 (node, neighbor) 没有被覆盖
                    return True, penalty_value, f"invalid vertex cover: edge ({node}, {neighbor}) is not covered"

    # 有效的顶点覆盖
    cover_size = len(vertices)

    # 特殊情况：空顶点覆盖
    if cover_size == 0:
        # 检查图是否有边
        has_edges = any(len(neighbors) > 0 for neighbors in graph.values())
        if has_edges:
            return True, penalty_value, "invalid vertex cover: empty set cannot cover edges"
        else:
            return False, 0, "valid vertex cover: empty set for graph with no edges"

    return False, cover_size, f"valid vertex cover with {cover_size} vertices"
