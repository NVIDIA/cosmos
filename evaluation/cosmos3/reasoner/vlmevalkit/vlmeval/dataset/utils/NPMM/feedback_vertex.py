import re
from collections import defaultdict


def validation(graph, answer):
    """
    验证给定的答案是否是一个有效的反馈顶点集方案
    参数:
    graph: 图的表示，包含邻接表
    answer: 字符串答案，包含"Answer："后的顶点列表

    返回:
    (bool, int, str): 布尔值表示答案是否有效，True代表无效，False代表有效，
                       整数表示反馈顶点集的大小（越小越好）。如果答案无效，返回一个比任何可能解都大的值。
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
            # 空列表对于无环图有效
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
    # 构建移除反馈顶点集后的邻接表
    remaining_adj = defaultdict(list)
    removed_set = set(vertices)
    for node_str, neighbors in graph.items():
        node = int(node_str)
        if node in removed_set:
            continue
        for neighbor_str in neighbors:
            if isinstance(neighbor_str, str):
                neighbor = int(neighbor_str)
            else:
                neighbor = neighbor_str

            if neighbor not in removed_set:
                remaining_adj[node].append(neighbor)

    # 检查剩余图是否无环
    def has_cycle_dfs(adj, num_verts):
        """使用DFS检查图是否有环"""
        visited = set()
        rec_stack = set()

        def dfs(node, parent):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent:
                    # 发现后向边（检测到环）
                    return True

            rec_stack.remove(node)
            return False

        # 检查所有连通分量
        for node in range(num_verts):
            if node in adj and node not in visited:
                if dfs(node, -1):
                    return True

        return False

    is_acyclic = not has_cycle_dfs(remaining_adj, num_vertices)

    if not is_acyclic:
        # 解决方案没有移除所有环
        return True, penalty_value, "invalid feedback vertex set: remaining graph still contains cycles"

    # 有效的反馈顶点集
    fvs_size = len(vertices)

    return False, fvs_size, f"valid feedback vertex set with {fvs_size} vertices"
