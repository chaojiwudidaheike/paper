import networkx as nx
from collections import deque
from Tools import isConflict
import random
import math


def total_weight(graph, nodes):
    """以固定节点顺序精确累加权重，消除哈希种子造成的漂移。"""
    ordered_nodes = sorted(
        nodes,
        key=lambda node: (graph.nodes[node].get('num', 0), node),
    )
    return math.fsum(graph.nodes[node]['weight'] for node in ordered_nodes)

# def find_all_clique_queues(G): #一个节点可能在两个队列中存在
#     """将图节点按worker、car、uav编号分组并按权重降序排列"""
#     # 初始化队列字典
#     queues = {}
#
#     # 遍历图中节点并分组
#     for node in G.nodes():
#         # 处理worker
#         if node[0] != 'None' and node[0].startswith('worker'):
#             worker_id = node[0].split('|')[0]
#             if worker_id not in queues:
#                 queues[worker_id] = []
#             queues[worker_id].append(node)
#
#         # 处理car
#         if node[1] != 'None' and node[1].startswith('car'):
#             car_id = node[1].split('|')[0]
#             if car_id not in queues:
#                 queues[car_id] = []
#             queues[car_id].append(node)
#
#         # 处理uav
#         if node[2] != 'None' and node[2].startswith('uav'):
#             uav_id = node[2].split('|')[0]
#             if uav_id not in queues:
#                 queues[uav_id] = []
#             queues[uav_id].append(node)
#
#     # 对每个队列按权重降序排列
#     for key in queues:
#         queues[key].sort(key=lambda x: (G.nodes[x].get('weight', 0), x), reverse=True)
#
#     return list(queues.values())
########################################################################
# def find_all_clique_queues(G):# 节点随机存在与他相关的某个队列中
#     """将图节点按worker、car、uav编号分组并按权重降序排列，每个节点随机选择一个队列"""
#     # 初始化队列字典
#     queues = {}
#
#     # 为每个节点记录可用的队列选项
#     node_queue_options = {}
#
#     # 第一遍遍历：收集每个节点可用的队列选项
#     for node in G.nodes():
#         options = []
#
#         # 处理worker
#         if node[0] != 'None' and node[0].startswith('worker'):
#             worker_id = node[0].split('|')[0]
#             options.append(worker_id)
#
#         # 处理car
#         if node[1] != 'None' and node[1].startswith('car'):
#             car_id = node[1].split('|')[0]
#             options.append(car_id)
#
#         # 处理uav
#         if node[2] != 'None' and node[2].startswith('uav'):
#             uav_id = node[2].split('|')[0]
#             options.append(uav_id)
#
#         node_queue_options[node] = options
#
#     # 第二遍遍历：随机分配节点到队列
#     for node, options in node_queue_options.items():
#         if options:  # 如果有可用的队列选项
#             # 随机选择一个队列
#             selected_queue = random.choice(options)
#
#             # 初始化队列（如果不存在）
#             if selected_queue not in queues:
#                 queues[selected_queue] = []
#
#             # 将节点添加到选定的队列
#             queues[selected_queue].append(node)
#
#     # 对每个队列按权重降序排列
#     for key in queues:
#         queues[key].sort(key=lambda x: (G.nodes[x].get('weight', 0), x), reverse=True)
#
#     return list(queues.values())


def find_all_clique_queues(G):  # 节点选择元素最少的队列
    """将图节点按worker、car、uav编号分组并按权重降序排列，节点选择元素最少的队列"""
    # 初始化队列字典
    queues = {}

    # 为每个节点记录可用的队列选项
    node_queue_options = {}

    # 第一遍遍历：收集每个节点可用的队列选项
    for node in G.nodes():
        options = []

        # 处理worker
        if node[0] != 'None' and node[0].startswith('worker'):
            worker_id = node[0].split('|')[0]
            options.append(worker_id)

        # 处理car
        if node[1] != 'None' and node[1].startswith('car'):
            car_id = node[1].split('|')[0]
            options.append(car_id)

        # 处理uav
        if node[2] != 'None' and node[2].startswith('uav'):
            uav_id = node[2].split('|')[0]
            options.append(uav_id)

        node_queue_options[node] = options

    # 按可选队列数量排序（可选队列少的先处理，这样约束更强的节点先分配）
    sorted_nodes = sorted(node_queue_options.items(), key=lambda x: len(x[1]))

    # 按排序顺序分配节点到队列
    for node, options in sorted_nodes:
        if options:  # 如果有可用的队列选项
            # 选择当前元素最少的队列
            selected_queue = min(options, key=lambda q: len(queues.get(q, [])))

            # 初始化队列（如果不存在）
            if selected_queue not in queues:
                queues[selected_queue] = []

            # 将节点添加到选定的队列
            queues[selected_queue].append(node)

    # 对每个队列按权重降序排列
    for key in queues:
        queues[key].sort(key=lambda x: (G.nodes[x].get('weight', 0), x), reverse=True)

    return list(queues.values())


def pop_max_from_each_queue(big_queue):
    """从每个队列弹出权重最大的节点，确保结果集合无重复"""
    result = []
    seen_nodes = set()  # 用于记录已选择的节点

    # 按队列长度降序处理，优先处理节点多的队列
    sorted_queues = sorted(big_queue, key=len, reverse=True)

    for queue in sorted_queues:
        while queue:  # 当队列不为空时循环
            candidate = queue.pop(0)  # 弹出队列头部（权重最大）
            # 检查节点是否已存在，不存在则添加
            if candidate not in seen_nodes:
                result.append(candidate)
                seen_nodes.add(candidate)
                break  # 找到合适节点后跳出当前队列的循环

    return result


def compute_independent_set(graph):
    """ 贪心计算图的最大权重独立集 """
    sorted_nodes = sorted(graph.nodes(),
                          key=lambda x: (graph.nodes[x].get('weight'), graph.nodes[x].get('num')),
                          reverse=True)
    independent_set = set()
    covered = set()

    for node in sorted_nodes:
        if node not in covered:
            independent_set.add(node)
            covered.add(node)
            covered.update(graph.neighbors(node))

    return independent_set


def max_weight_max_independent_set(G):
    queues = find_all_clique_queues(G)
    print(f"总队列数量: {len(queues)}")
    temp_G = nx.Graph()

    """ 建立初始图 """
    new_nodes = pop_max_from_each_queue(queues)
    for node in new_nodes:
        # 检查节点是否在原图中存在
        if node not in temp_G.nodes:
            # 获取节点的权重
            weight = G.nodes[node].get('weight', 0)
            # 将节点添加到新图，并设置权重属性
            temp_G.add_node(
                node, weight=weight, num=G.nodes[node].get('num')
            )

    nodes = list(temp_G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            if isConflict(node1, node2):
                temp_G.add_edge(node1, node2)
    best_independent_set = compute_independent_set(temp_G)
    best_weight = total_weight(G, best_independent_set)
    iset = best_independent_set.copy()

    """ 不断弹出结点 """
    #用来判断循环是否可以提前结束
    P = True
    while any(queues):
        # 1. 弹出并添加新节点
        new_nodes = pop_max_from_each_queue(queues)
        print(new_nodes)
        for node in new_nodes:
            # 检查节点是否在原图中存在
            if node not in temp_G.nodes:
                # 获取节点的权重
                weight = G.nodes[node].get('weight', 0)
                # 将节点添加到新图，并设置权重属性
                temp_G.add_node(
                    node, weight=weight, num=G.nodes[node].get('num')
                )

        # 2. 建立新连接
        nodes = list(temp_G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]
                if isConflict(node1, node2):
                    temp_G.add_edge(node1, node2)

        # 3. 添加新结点进入独立集
        for node in new_nodes:
            add1 = True
            for node2 in best_independent_set:
                if temp_G.has_edge(node, node2):
                    add1 = False
                    break
            if add1:
                best_independent_set.add(node)

        best_weight = total_weight(G, best_independent_set)
        remaining_nodes = temp_G.nodes() - best_independent_set
        # 4. 构建初始队列
        remaining_nodes_list = sorted(remaining_nodes, key=lambda x: (G.nodes[x].get('weight'), G.nodes[x].get('num')), reverse=True)
        best_independent_list = sorted(best_independent_set, key=lambda x: (G.nodes[x].get('weight'), G.nodes[x].get('num')), reverse=True)
        independent_queue, remaining_queue = build_queues(temp_G, best_independent_list, remaining_nodes_list)
        # 5. 优化循环
        improved = True
        while improved:
            improved = False

            # 遍历剩余结点队列寻找改进
            for remaining_item in list(remaining_queue):
                # 找到可替换的独立集结点
                replace_candidates = []
                for neighbor in remaining_item['neighbors']:
                    for indep_item in independent_queue:
                        if indep_item['node'] == neighbor:
                            replace_candidates.append(indep_item)
                            break

                # 替换每个候选，创建临时独立集
                temp_iset = best_independent_set.copy()
                temp_iset.add(remaining_item['node'])
                for candidate in replace_candidates:
                    temp_iset.remove(candidate['node'])

                # 尝试添加被替换节点的邻居
                for candidate in replace_candidates:
                    for neighbor in candidate['neighbors']:
                        # 检查该邻居是否可以加入
                        can_add = True
                        for existing_node in temp_iset:
                            if temp_G.has_edge(neighbor, existing_node):
                                can_add = False
                                break
                        if can_add:
                            temp_iset.add(neighbor)

                temp_weight = total_weight(G, temp_iset)

                if temp_weight > best_weight:
                    best_independent_set = temp_iset.copy()
                    remaining_nodes = temp_G.nodes() - best_independent_set
                    best_weight = temp_weight
                    best_independent_list = sorted(best_independent_set, key=lambda x: (G.nodes[x].get('weight'), G.nodes[x].get('num')), reverse=True)
                    remaining_nodes_list = sorted(remaining_nodes, key=lambda x: (G.nodes[x].get('weight'), G.nodes[x].get('num')), reverse=True)
                    independent_queue, remaining_queue = build_queues(temp_G, best_independent_list, remaining_nodes_list)
                    improved = True
                    break

        if iset != best_independent_set:
            iset = best_independent_set.copy()
        elif iset == best_independent_set:
            P = False
            break

    if P == False:
        all_nodes = merge_and_sort_nodes(G, queues)
        best_independent_set = build_independent_set(G, all_nodes, best_independent_set)
        best_weight = total_weight(G, best_independent_set)


    return best_independent_set, best_weight


def build_queues(graph, independent_set, remaining_nodes):
    """ 构建独立集队列和剩余队列 """
    independent_queue = deque()
    remaining_queue = deque()

    # 独立集队列（节点+其在剩余集中的邻居）
    for node in independent_set:
        neighbors = set(graph.neighbors(node)) & set(remaining_nodes)
        neighbors = sorted(neighbors, key=lambda x: (graph.nodes[x].get('weight'), graph.nodes[x].get('num')), reverse=True)
        independent_queue.append({
            'node': node,
            'neighbors': neighbors,
            'weight': graph.nodes[node]['weight']
        })

    # 剩余队列（节点+其在独立集中的邻居）
    for node in remaining_nodes:
        neighbors = set(graph.neighbors(node)) & set(independent_set)
        neighbors = sorted(neighbors, key=lambda x: (graph.nodes[x].get('weight'), graph.nodes[x].get('num')), reverse=True)
        remaining_queue.append({
            'node': node,
            'neighbors': neighbors,
            'weight': graph.nodes[node]['weight']
        })

    return independent_queue, remaining_queue

def merge_and_sort_nodes(G, queues):

    all_nodes = []

    # 遍历所有队列
    for queue in queues:
        for node in queue:
            all_nodes.append(node)

    # 去重（基于节点对象，不是基于节点内容）
    unique_nodes = []
    seen = set()
    for node in all_nodes:
        if node not in seen:
            seen.add(node)
            unique_nodes.append(node)

    # 按权重降序排序
    unique_nodes.sort(key=lambda x: (G.nodes[x].get('weight'), G.nodes[x].get('num')), reverse=True)

    return unique_nodes


def build_independent_set(G, all_nodes, independent_set):
    forbidden_nodes = independent_set.copy()  # 初始化禁止集合（已存在的节点不能再次加入）
    independent_set = independent_set.copy()
    # 遍历所有候选节点（按权重降序）
    for node in all_nodes:
        # 如果节点已被禁止，跳过
        if node in forbidden_nodes:
            continue

        # 检查是否与 independent_set 中的任何节点有边
        has_conflict = any(G.has_edge(node, existing_node) for existing_node in independent_set)

        # 如果没有冲突，则加入独立集
        if not has_conflict:
            independent_set.add(node)
            forbidden_nodes.update(G.neighbors(node))  # 禁止邻居
            forbidden_nodes.add(node)  # 禁止自己（避免重复处理）

    return independent_set

def no_mulPQ_max_weight_max_independent_set(G):

    temp_G = G

    best_independent_set = compute_independent_set(temp_G)
    best_weight = total_weight(G, best_independent_set)
    remaining_nodes = temp_G.nodes() - best_independent_set
    remaining_nodes_list = sorted(remaining_nodes, key=lambda x: (G.nodes[x].get('weight'), G.nodes[x].get('num')), reverse=True)
    best_independent_list = sorted(best_independent_set, key=lambda x: (G.nodes[x].get('weight'), G.nodes[x].get('num')), reverse=True)
    independent_queue, remaining_queue = build_queues(temp_G, best_independent_list, remaining_nodes_list)

    improved = True
    while improved:
        improved = False

        # 遍历剩余结点队列寻找改进
        for remaining_item in list(remaining_queue):
            # 找到可替换的独立集结点
            replace_candidates = []
            for neighbor in remaining_item['neighbors']:
                for indep_item in independent_queue:
                    if indep_item['node'] == neighbor:
                        replace_candidates.append(indep_item)
                        break

            # 替换每个候选，创建临时独立集
            temp_iset = best_independent_set.copy()
            temp_iset.add(remaining_item['node'])
            for candidate in replace_candidates:
                temp_iset.remove(candidate['node'])

            # 尝试添加被替换节点的邻居
            for candidate in replace_candidates:
                for neighbor in candidate['neighbors']:
                    # 检查该邻居是否可以加入
                    can_add = True
                    for existing_node in temp_iset:
                        if temp_G.has_edge(neighbor, existing_node):
                            can_add = False
                            break
                    if can_add:
                        temp_iset.add(neighbor)

            temp_weight = total_weight(G, temp_iset)

            if temp_weight > best_weight:
                best_independent_set = temp_iset.copy()
                remaining_nodes = temp_G.nodes() - best_independent_set
                best_weight = temp_weight
                best_independent_list = sorted(best_independent_set, key=lambda x: (G.nodes[x].get('weight'), G.nodes[x].get('num')), reverse=True)
                remaining_nodes_list = sorted(remaining_nodes, key=lambda x: (G.nodes[x].get('weight'), G.nodes[x].get('num')), reverse=True)
                independent_queue, remaining_queue = build_queues(temp_G, best_independent_list, remaining_nodes_list)
                improved = True
                break


    return best_independent_set, best_weight

