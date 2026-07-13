def validation(graph, answer):
    """
    验证给定的答案是否是图中的一个有效团（clique），并返回该团的大小。

    参数:
      graph: dict，图的邻接表示。兼容以下几种常见形式：
        1) {int: [int, ...]} 或 {str: [str/int, ...]}
        2) {int/str: {int/str: 0/1/权重,...}}（邻接字典/加权）
        3) {int/str: set(...)}

      answer: str，格式为 "Answer: [0, 1, 2]"

    返回:
      (bool, int, str):
        - bool: True 表示非法；False 表示合法
        - int: 团大小（非法时为 -1）
        - str: 说明
    """
    import ast
    from collections import defaultdict

    # ---------- 1) 解析答案 ----------
    if "Answer:" not in answer:
        return True, -1, "invalid answer: no 'Answer:' in answer"

    cut_str = answer.split("Answer:", 1)[-1].strip()
    try:
        clique = ast.literal_eval(cut_str)
    except Exception:
        return True, -1, "invalid format: cannot parse list after 'Answer:'"

    if not isinstance(clique, list):
        return True, -1, "invalid format: answer must be a list like [0, 1, 2]"
    if len(clique) == 0:
        return True, -1, "invalid clique: empty list"
    # 去重检查
    if len(set(clique)) != len(clique):
        return True, -1, "invalid clique: duplicated vertices in the list"

    # 尝试把节点转成 int（如果本来就是 str 的编号也兼容）
    norm_clique = []
    for x in clique:
        try:
            norm_clique.append(int(x))
        except Exception:
            return True, -1, f"invalid vertex id: {x!r} is not an integer"

    # ---------- 2) 规范化图为: {int: set(int,...)} ----------
    # 允许 graph 的 key/邻居是 str 或 int；允许邻居是 list/set/dict（值为权重）
    neighbors = defaultdict(set)

    # 收集所有节点（键 + 邻居里出现的点）
    all_vertices = set()
    for u_raw, adj in graph.items():
        try:
            u = int(u_raw)
        except Exception:
            # 忽略无法转为 int 的键
            continue
        all_vertices.add(u)

        if isinstance(adj, dict):
            # 邻接字典：取权重>0或存在即视为有边
            for v_raw, w in adj.items():
                try:
                    v = int(v_raw)
                except Exception:
                    continue
                all_vertices.add(v)
                # 只要有条边（权重大于0或存在即认为有边）
                if isinstance(w, (int, float)):
                    if w != 0:
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                else:
                    # 非数值，保守认为存在边
                    neighbors[u].add(v)
                    neighbors[v].add(u)
        elif isinstance(adj, (list, set, tuple)):
            for v_raw in adj:
                try:
                    v = int(v_raw)
                except Exception:
                    continue
                all_vertices.add(v)
                neighbors[u].add(v)
                neighbors[v].add(u)
        else:
            # 未知结构，跳过
            pass

    # 若图没有任何边但有点，neighbors 里也应包含孤立点
    for u in list(all_vertices):
        neighbors[u] = set(neighbors[u])  # ensure key exists

    # ---------- 3) 基本合法性检查 ----------
    # 节点是否都存在
    missing = [u for u in norm_clique if u not in all_vertices]
    if missing:
        return True, -1, f"invalid vertices (not in graph): {sorted(missing)}"

    # ---------- 4) 团检查：两两必须相连 ----------
    cset = set(norm_clique)
    for i in range(len(norm_clique)):
        u = norm_clique[i]
        # 为了 O(1) 查询，使用 set
        Nu = neighbors.get(u, set())
        # clique 中除 u 自身外的所有顶点都必须在 Nu 中
        if not (cset - {u}).issubset(Nu):
            # 找出缺的边，便于提示
            missing_neighbors = sorted((cset - {u}) - Nu)
            return True, -1, f"invalid clique: vertex {u} is not connected to {missing_neighbors}"

    # ---------- 5) 返回 ----------
    size = len(norm_clique)
    return False, size, f"valid clique of size {size} (note: not necessarily maximum)"
