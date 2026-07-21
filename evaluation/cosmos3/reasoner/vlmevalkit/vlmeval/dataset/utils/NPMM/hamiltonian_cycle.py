def validation(graph, answer):
    """
    验证给定的答案是否是一个有效的哈密顿回路

    参数:
    graph: 图的表示，包含nodes和adjacency_list
    answer: 字符串答案，包含"Answer："后的路径

    返回:
    (bool, str): 布尔值表示答案是否正确，字符串提供错误信息（如有）
    """
    # 解析answer字符串，提取"Answer："后面的内容
    if "Answer:" in answer:
        path_str = answer.split("Answer:")[-1].strip()
    else:
        return True, -1, "invalid answer: no 'Answer:' in answer"

    # 标准化路径字符串格式
    path_str = path_str.strip()

    # 如果已经是 [x,x,x,...,x] 格式，直接使用
    if path_str.startswith('[') and path_str.endswith(']'):
        path_str = path_str[1:-1]  # 去掉方括号
    else:
        # 处理 -> 格式
        if '->' in path_str:
            # 分割并清理每个节点
            nodes = [node.strip() for node in path_str.split('->')]
            # 重新组合成标准格式
            path_str = ','.join(nodes)
        else:
            # 处理其他括号格式 {x,x,x} 或 (x,x,x) 或 ['x','x','x']
            # 移除所有引号
            path_str = path_str.replace("'", "").replace('"', '')

            # 使用正则表达式匹配括号内容
            import re
            pattern = r'[{\[\(]([^)}\]]*)[}\])]'
            match = re.search(pattern, path_str)

            if match:
                # 提取括号内的内容
                path_str = match.group(1)
            else:
                # 如果没有任何括号，假设是逗号分隔的列表
                pass

    # 尝试将路径字符串转换为节点列表
    try:
        # 分割并转换为整数列表
        try:
            path = [int(x.strip()) for x in path_str.split(',')]
        except Exception:
            return True, -1, "path must be a list of integers"

        if not path:
            return True, -1, "path cannot be empty"
    except Exception:
        return True, -1, "invalid answer format"

    # 获取图的节点数和邻接表
    adjacency_list = graph

    # 检查路径是否是哈密顿回路

    # 2. 检查起点和终点是否相同（闭环）
    if path[0] != path[-1]:
        return True, -1, f"path is not a cycle: start{path[0]} and end{path[-1]} are different"

    # 3. 检查是否访问了所有节点且只访问一次（除了起点/终点）
    visited = set()
    for i in range(len(path) - 1):
        if path[i] in visited:
            return True, -1, f"node {path[i]} is visited more than once"
        visited.add(path[i])

    # 4. 检查相邻节点之间是否有边连接
    for i in range(len(path) - 1):
        current = str(path[i])  # 转为字符串，因为adjacency_list的键可能是字符串
        next_node = path[i + 1]  # 保持为整数

        if current not in adjacency_list:
            return True, -1, f"node {current} is not in adjacency list"

        # 邻接表中的值可能是整数列表
        neighbors = adjacency_list[current]
        if next_node not in neighbors:
            return True, -1, f"node {current} and {next_node} are not connected"

    return False, len(path), f"path length: {len(path)}"
