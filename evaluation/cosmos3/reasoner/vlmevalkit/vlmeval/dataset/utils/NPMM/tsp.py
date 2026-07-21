def validation(graph, answer):
    """
    验证给定的答案是否是一个有效的TSP路线

    参数:
    graph: json中question1字段, 表示城市之间的距离
    answer: 字符串答案，包含"Answer："后的路径

    返回:
    (bool, float, str): 布尔值表示答案是否有效，True代表无效，False代表有效，
                       浮点数表示路径总距离。如果答案无效，返回一个比任何可能路径都大的值。
                       字符串提供错误或成功信息。
    """
    # 计算一个大于任何可能路径长度的值，作为无效答案的得分。
    # 这里使用所有边距离之和作为惩罚值，因为任何有效路径的长度都不会超过它。
    total_distance_sum = 0
    cities = list(graph.keys())
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            u, v = cities[i], cities[j]
            total_distance_sum += graph[u].get(v, 0)

    # 如果图中没有距离，使用一个默认的大值来惩罚无效格式的答案。
    if total_distance_sum == 0:
        total_distance_sum = 1e9

    # 解析answer字符串，提取"Answer："后面的内容
    if "Answer:" in answer:
        path_str = answer.split("Answer:")[-1].strip()
    else:
        return True, total_distance_sum, "invalid answer: no 'Answer:' in answer"

    # 标准化路径字符串格式
    path_str = path_str.strip()
    if path_str.startswith('[') and path_str.endswith(']'):
        path_str = path_str[1:-1]
    else:
        if '->' in path_str:
            nodes = [node.strip() for node in path_str.split('->')]
            path_str = ','.join(nodes)
        else:
            path_str = path_str.replace("'", "").replace('"', '')
            import re
            pattern = r'[{\[\(]([^)}\]]*)[}\])]'
            match = re.search(pattern, path_str)
            if match:
                path_str = match.group(1)

    try:
        try:
            path = [int(x.strip()) for x in path_str.split(',')]
        except Exception:
            return True, total_distance_sum, "path must be a list of integers"
        if not path:
            return True, total_distance_sum, "path cannot be empty"
    except Exception:
        return True, total_distance_sum, "invalid answer format"

    #  TSP specific validation

    # 1. 检查起点和终点是否相同
    if not path or path[0] != path[-1]:
        return True, total_distance_sum, f"path is not a cycle: start {path[0]} and end {path[-1]} are different"

    # 2. 检查是否访问了所有城市且只访问一次（除了起点/终点）
    num_cities = len(graph)
    if len(path) != num_cities + 1:
        return True, total_distance_sum, f"path length is incorrect. Expected {num_cities + 1} cities in path, but got {len(path)}"  # noqa: E501

    visited = set()
    for city in path[:-1]:  # Exclude the last city (which is the same as the first)
        if city in visited or city < 0 or city >= num_cities:
            return True, total_distance_sum, f"invalid city: {city} (either repeated, negative, or out of range)"
        visited.add(city)
    if len(visited) != num_cities:
        return True, total_distance_sum, f"not all cities are visited. Expected {num_cities}, but visited {len(visited)}"  # noqa: E501
    # 3. 计算总距离
    total_distance = 0
    for i in range(len(path) - 1):
        current = str(path[i])
        next_city = str(path[i + 1])
        try:
            distance = graph[current][next_city]  # 直接访问距离
            total_distance += distance
        except KeyError:
            return True, total_distance_sum, f"no distance found between cities {current} and {next_city}"

    return False, total_distance, f"Total distance: {total_distance}"
