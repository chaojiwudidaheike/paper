import networkx as nx
import heapq
from Tools import isConflict


# ========== 贪心启发式：寻找较重的独立集作为初始解 ==========

def greedy_heuristic(G):
    """Best-in贪心启发式：按 H(v) = w(v) / (1 + sum(w(N(v)))) 选点"""
    remaining = set(G.nodes())
    best_is = set()

    # 在多个诱导子图上运行启发式取最优
    candidates = sorted(remaining, key=lambda v: _heuristic_score(G, v), reverse=True)
    k = min(len(candidates), 10 + len(candidates) // 50)

    for v_init in candidates[:k]:
        iset = _greedy_on_subgraph(G, v_init, remaining)
        if _weight(G, iset) > _weight(G, best_is):
            best_is = iset

    return best_is


def _heuristic_score(G, v):
    """H_G(v) = w(v) / (1 + w(N(v)))"""
    w_v = G.nodes[v].get('weight', 0)
    w_neighbors = sum(G.nodes[u].get('weight', 0) for u in G.neighbors(v))
    return w_v / (1 + w_neighbors) if (1 + w_neighbors) > 0 else 0


def _greedy_on_subgraph(G, v_init, remaining):
    """在 G[{v_init} ∪ N_bar(v_init)] 上运行贪心"""
    non_neighbors = {v_init}
    for v in remaining:
        if v != v_init and not G.has_edge(v, v_init):
            non_neighbors.add(v)

    sub_remaining = set(non_neighbors)
    iset = set()

    while sub_remaining:
        best_v = max(sub_remaining, key=lambda v: _heuristic_score(G, v))
        iset.add(best_v)
        sub_remaining.discard(best_v)
        sub_remaining -= set(G.neighbors(best_v))

    return iset


def _weight(G, iset):
    return sum(G.nodes[v].get('weight', 0) for v in iset)


# ========== 加权团覆盖上界（Method A: Enhanced Weighted Greedy Method） ==========

def compute_upper_bound_method_a(G, candidate_nodes, best_is_weight):
    """
    Method A (Enhanced Weighted Greedy Method) 计算加权团覆盖上界。
    输入:
        G: 原始图
        candidate_nodes: 当前子问题的候选节点集合
        best_is_weight: 当前已知最优独立集权重（用于Z_i剪枝）
    返回: 上界值
    """
    if not candidate_nodes:
        return 0

    nodes = list(candidate_nodes)
    # 残余权重
    residual = {v: G.nodes[v].get('weight', 0) for v in nodes}
    total_cover_weight = 0.0

    remaining = set(nodes)

    while remaining:
        # 找一个极大团 K_hat
        clique = _find_maximal_clique(G, remaining)

        if not clique:
            break

        # W_i = min{residual(v) : v in K_hat}
        W_i = min(residual[v] for v in clique)

        # 更新残余权重
        for v in clique:
            residual[v] -= W_i

        total_cover_weight += W_i

        # 移除权重为0的节点 (U_i)
        remaining -= {v for v in list(remaining) if residual[v] <= 1e-12}

        # Z_i 剪枝: 移除残余权重超过 gap 的节点
        gap = best_is_weight - total_cover_weight
        remaining -= {v for v in list(remaining) if residual[v] > gap}

    return total_cover_weight


def _find_maximal_clique(G, node_set):
    """在 node_set 中贪心找一个极大团"""
    if not node_set:
        return set()

    # 选残余权重最快的节点作为起点
    start = max(node_set, key=lambda v: G.nodes[v].get('weight', 0))
    clique = {start}
    candidates = node_set - {start}

    for v in candidates:
        # 检查 v 是否与团中所有节点相邻
        if all(G.has_edge(v, u) for u in clique):
            clique.add(v)

    return clique


# ========== 加权团覆盖上界（Method B: Enhanced Babel Method） ==========

def compute_upper_bound_method_b(G, candidate_nodes, best_is_weight):
    """
    Method B (Enhanced Babel Method) 计算加权团覆盖上界。
    使用广义团度(GKD)选择节点，构建加权团覆盖。
    返回: (上界值, 排序后的分支顶点列表)
    """
    if not candidate_nodes:
        return 0, []

    nodes = list(candidate_nodes)
    uncovered = {v: G.nodes[v].get('weight', 0) for v in nodes}
    cliques = []  # [(成员set, 权重)]

    uncovered_set = set(nodes)

    while uncovered_set:
        # 选择 GKD 最大的未覆盖节点
        best_v = max(uncovered_set, key=lambda v: _gkd(v, cliques, G))
        W_v = uncovered[best_v]

        if W_v <= 1e-12:
            uncovered_set.discard(best_v)
            continue

        # 检查已有团
        inserted = False
        new_cliques = []
        for members, W in cliques:
            if inserted:
                new_cliques.append((members, W))
                continue

            # 检查 best_v 是否与团中所有成员相邻
            all_neighbors = all(G.has_edge(best_v, u) for u in members)

            if all_neighbors:
                if W < W_v:
                    # W < W_v: 加入该团，减少 W_v
                    new_members = members | {best_v}
                    new_cliques.append((new_members, W))
                    W_v -= W
                    uncovered[best_v] = W_v
                elif abs(W - W_v) < 1e-12:
                    # W = W_v: 加入该团，结束
                    new_members = members | {best_v}
                    new_cliques.append((new_members, W))
                    W_v = 0
                    uncovered[best_v] = 0
                    inserted = True
                else:
                    # W > W_v: 分裂团
                    new_members = members | {best_v}
                    new_cliques.append((new_members, W_v))
                    new_cliques.append((members, W - W_v))
                    W_v = 0
                    uncovered[best_v] = 0
                    inserted = True
            else:
                new_cliques.append((members, W))

        if W_v > 1e-12:
            # 新建单节点团
            new_cliques.append(({best_v}, W_v))
            uncovered[best_v] = 0

        cliques = new_cliques
        uncovered_set.discard(best_v)

    # 计算上界
    upper_bound = sum(W for _, W in cliques)

    # 计算 r'(v) 用于分支顶点排序
    r_prime = {}
    for v in nodes:
        r_prime[v] = 0
        for members, W in cliques:
            if v in members:
                r_prime[v] += W

    # 按r'降序排列分支顶点
    branch_vertices = sorted(nodes, key=lambda v: r_prime[v], reverse=True)
    # 只保留 r'(v) > best_is_weight 的顶点
    branch_vertices = [v for v in branch_vertices if r_prime[v] > best_is_weight]

    return upper_bound, branch_vertices


def _gkd(v, cliques, G):
    """计算节点v的广义团度 GKD(v) = sum(W_i) for cliques K_i s.t. K_i ∩ N_bar(v) ≠ ∅"""
    total = 0
    for members, W in cliques:
        for u in members:
            if u != v and not G.has_edge(v, u):
                total += W
                break
    return total


# ========== 预处理：顶点消除 ==========

def vertex_elimination(G, candidate_nodes, best_is_weight):
    """
    Babel顶点消除技术：
    对于v∈V, K(v)是覆盖v或v的非邻居的团集合。
    如果 K(v) 的总权重 ≤ best_is_weight，则v可被消除。
    """
    eliminated = set()

    for v in candidate_nodes:
        non_neighbors = {u for u in candidate_nodes if u != v and not G.has_edge(v, u)}
        sub_nodes = {v} | non_neighbors

        # 快速上界估计：v及非邻居的权重之和
        ub_estimate = sum(G.nodes[u].get('weight', 0) for u in sub_nodes)
        if ub_estimate <= best_is_weight:
            eliminated.add(v)

    return candidate_nodes - eliminated


# ========== Branch-and-Bound 主算法 (Algorithm AB) ==========

def max_weight_max_independent_set(G):
    """
    使用Branch-and-Bound方法（Algorithm AB）求解最大权独立集。
    - 前3代使用Method B计算上界
    - 更深层使用Method A计算上界
    - 根节点执行顶点消除预处理
    - 使用贪心启发式获取初始解
    - 最优上界优先的节点选择策略
    """
    if len(G.nodes()) == 0:
        return set(), 0

    # 1. 贪心启发式获取初始解
    best_is = greedy_heuristic(G)
    best_weight = _weight(G, best_is)
    print(f"[BnB] Initial heuristic weight: {best_weight:.4f}")

    # 2. 根节点顶点消除
    all_nodes = set(G.nodes())
    all_nodes = vertex_elimination(G, all_nodes, best_weight)

    if not all_nodes:
        return best_is, best_weight

    # 3. 使用Method B计算根节点上界和分支顺序
    root_ub, branch_order = compute_upper_bound_method_b(G, all_nodes, best_weight)
    print(f"[BnB] Root upper bound: {root_ub:.4f}, branch vertices: {len(branch_order)}")

    if root_ub <= best_weight:
        return best_is, best_weight

    # 4. Branch-and-Bound 搜索
    # 优先队列: (-upper_bound, counter, included_set, excluded_set, depth)
    counter = 0
    heap = []
    # 从branch_order中的顶点逐个分支
    # 初始状态：无强制包含、无强制排除
    heapq.heappush(heap, (-root_ub, counter, frozenset(), frozenset(), 0))
    counter += 1

    nodes_explored = 0

    while heap:
        neg_ub, _, included, excluded, depth = heapq.heappop(heap)
        current_ub = -neg_ub

        # 剪枝：上界不优于当前最优
        if current_ub <= best_weight:
            continue

        nodes_explored += 1

        # 计算当前子问题的候选节点
        cand = all_nodes - excluded
        # 从cand中移除与included冲突的节点
        for v in included:
            for u in G.neighbors(v):
                cand.discard(u)
        # included中的节点本身不在cand中（它们已被选中）
        cand -= included

        if not cand:
            # 无候选节点，检查当前解
            current_weight = _weight(G, included)
            if current_weight > best_weight:
                best_is = set(included)
                best_weight = current_weight
            continue

        # 分支策略：选择候选中权重最高的未分支节点
        # 按邻居权重之和升序排列（Carraghan-Pardalos策略）
        remaining_branch = [v for v in branch_order if v in cand]
        if not remaining_branch:
            # branch_order中的顶点已全部分支，尝试剩余候选
            remaining_branch = sorted(cand, key=lambda v: sum(G.nodes[u].get('weight', 0) for u in G.neighbors(v) if u in cand))

        if not remaining_branch:
            current_weight = _weight(G, included)
            if current_weight > best_weight:
                best_is = set(included)
                best_weight = current_weight
            continue

        # 取第一个分支顶点
        x = remaining_branch[0]

        # --- 分支1: 包含 x ---
        new_included = included | {x}
        # 排除x的邻居（它们与x冲突）
        new_excluded = excluded | {u for u in G.neighbors(x) if u in cand}

        # 计算子问题上界
        child1_cand = all_nodes - new_excluded
        for v in new_included:
            for u in G.neighbors(v):
                child1_cand.discard(u)
        child1_cand -= new_included

        if child1_cand:
            child1_depth = depth + 1
            # 前3代用Method B，之后用Method A
            if child1_depth < 3:
                child1_ub, _ = compute_upper_bound_method_b(G, child1_cand, best_weight - _weight(G, new_included))
            else:
                child1_ub = compute_upper_bound_method_a(G, child1_cand, best_weight - _weight(G, new_included))
            child1_ub += _weight(G, new_included)
        else:
            child1_ub = _weight(G, new_included)

        if child1_ub > best_weight:
            if not child1_cand:
                # 叶节点
                current_weight = _weight(G, new_included)
                if current_weight > best_weight:
                    best_is = set(new_included)
                    best_weight = current_weight
            else:
                heapq.heappush(heap, (-child1_ub, counter, frozenset(new_included), frozenset(new_excluded), child1_depth))
                counter += 1

        # --- 分支2: 不包含 x ---
        new_included2 = included
        new_excluded2 = excluded | {x}

        # 计算子问题上界（排除x后）
        child2_cand = all_nodes - new_excluded2
        for v in new_included2:
            for u in G.neighbors(v):
                child2_cand.discard(u)
        child2_cand -= new_included2

        if child2_cand:
            child2_depth = depth + 1
            if child2_depth < 3:
                child2_ub, _ = compute_upper_bound_method_b(G, child2_cand, best_weight - _weight(G, new_included2))
            else:
                child2_ub = compute_upper_bound_method_a(G, child2_cand, best_weight - _weight(G, new_included2))
            child2_ub += _weight(G, new_included2)
        else:
            child2_ub = _weight(G, new_included2)

        if child2_ub > best_weight:
            if not child2_cand:
                current_weight = _weight(G, new_included2)
                if current_weight > best_weight:
                    best_is = set(new_included2)
                    best_weight = current_weight
            else:
                heapq.heappush(heap, (-child2_ub, counter, frozenset(new_included2), frozenset(new_excluded2), child2_depth))
                counter += 1

    print(f"[BnB] Nodes explored: {nodes_explored}")
    return best_is, best_weight
