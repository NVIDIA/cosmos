import re


def validation(question, answer):
    """
    验证给定的答案是否是一个有效的最大割方案

    参数:
    question: 问题表示，包含 num_vertices 和 edges（边及其权重）
    answer: 字符串答案，包含"Answer："后的两个集合

    返回:
    (bool, int, str): 布尔值表示答案是否有效，True代表无效，False代表有效，
                       整数表示割的总权重（越大越好）。如果答案无效，返回0。
                       字符串提供错误或成功信息。
    """
    # 解析问题
    if isinstance(question, dict):
        edges_with_weights = question.get("edges", {})
        num_vertices = question.get("num_vertices", 0)
    else:
        return True, 0, "invalid question format"

    penalty_value = 0

    # 解析answer字符串，提取"Answer:"后面的内容
    if "Answer:" in answer:
        partition_str = answer.split("Answer:")[-1].strip()
    else:
        return True, penalty_value, "invalid answer: no 'Answer:' in answer"

    # 清理字符串
    partition_str = partition_str.strip().replace("'", "").replace('"', '')

    # 解析嵌套列表结构 [[set1], [set2]]
    try:
        # 匹配外层括号
        pattern = r'\[\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]\s*\]'
        match = re.search(pattern, partition_str)

        if not match:
            return True, penalty_value, "partition must be in format [[set1], [set2]]"

        set1_str = match.group(1).strip()
        set2_str = match.group(2).strip()

        # 解析每个集合
        if set1_str == '':
            set1 = []
        else:
            set1 = [int(x.strip()) for x in set1_str.split(',') if x.strip() != '']

        if set2_str == '':
            set2 = []
        else:
            set2 = [int(x.strip()) for x in set2_str.split(',') if x.strip() != '']

        # 去除每个集合内的重复元素
        set1 = sorted(list(set(set1)))
        set2 = sorted(list(set(set2)))

    except Exception as e:
        return True, penalty_value, f"failed to parse partition: {str(e)}"

    # 检查划分是否有效（所有顶点恰好出现一次）
    all_vertices = set(set1) | set(set2)
    if len(set1) + len(set2) != len(all_vertices):
        return True, penalty_value, "partition contains duplicate vertices"

    if all_vertices != set(range(num_vertices)):
        return True, penalty_value, f"partition must include all {num_vertices} vertices"

    # 检查所有顶点是否有效
    for v in set1 + set2:
        if v < 0 or v >= num_vertices:
            return True, penalty_value, f"invalid vertex {v}: must be in range [0, {num_vertices - 1}]"

    # 转换为集合以便高效查找
    set1_set = set(set1)
    set2_set = set(set2)

    # 计算割的值
    cut_value = 0
    for edge_key, weight in edges_with_weights.items():
        u, v = map(int, edge_key.split('-'))

        # 检查这条边是否跨越割
        if (u in set1_set and v in set2_set) or (u in set2_set and v in set1_set):
            cut_value += weight

    return False, cut_value, f"valid partition with cut value {cut_value}"
