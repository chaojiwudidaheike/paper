import networkx as nx
import re
import numpy as np
import math
from collections import defaultdict


def relative_improvement(current_distance, next_distance):
    if current_distance == 0:
        return 0.0
    return math.tanh(
        (current_distance - next_distance) / current_distance
    )


def build_candidate_graph(Env, Agent_Data, task_time, CurrTimeAge, SingleStep,
                          TimeLimit, faulty_agents=None):
    """生成全部可行候选节点，但暂不物化冲突边。"""
    faulty_agents = set() if faulty_agents is None else faulty_agents
    agent_count = Agent_Data.shape[0]
    decision_time = (CurrTimeAge - 1) * SingleStep
    none = "None"

    positions = {
        i: (int(Agent_Data[i, 2] - 1), int(Agent_Data[i, 3] - 1))
        for i in range(agent_count)
    }
    labels = {}

    def agent_label(kind, index):
        key = kind, index
        if key not in labels:
            row, col = positions[index]
            labels[key] = f"{kind}{index}|{row},{col}"
        return labels[key]

    workers, cars, uavs = [], [], []
    for index in range(agent_count):
        if index in faulty_agents:
            continue
        if not (Agent_Data[index, 1] <= decision_time
                < Agent_Data[index, 4]):
            continue
        agent_type = int(Agent_Data[index, 5])
        if agent_type == 1:
            workers.append(index)
        elif agent_type == 2:
            cars.append(index)
        elif agent_type == 3:
            uavs.append(index)

    tasks = [
        (int(row), int(col), f"task{int(row)},{int(col)}")
        for row, col in np.argwhere(Env[1] == 1)
    ]
    charges = [
        (int(row), int(col), f"charging{int(row)},{int(col)}")
        for row, col in np.argwhere(Env[0] == 1)
    ]

    print("Workers:", [agent_label("worker", i) for i in workers])
    print("Cars:", [agent_label("car", i) for i in cars])
    print("Uavs:", [agent_label("uav", i) for i in uavs])
    print("Tasks:", [task[2] for task in tasks])
    print("Charging:", [charge[2] for charge in charges])

    distance_cache = {}

    def agent_distance(index, row, col):
        key = index, row, col
        if key not in distance_cache:
            agent_row, agent_col = positions[index]
            distance_cache[key] = calculate_distance(
                row, col, agent_row, agent_col
            )
        return distance_cache[key]

    nearest_charge = {}
    for row, col, _ in tasks:
        nearest_charge[row, col] = min(
            (calculate_distance(row, col, charge_row, charge_col)
             for charge_row, charge_col, _ in charges),
            default=float("inf"),
        )

    task_motion = {}

    def task_values(index, task):
        row, col, _ = task
        key = index, row, col
        if key not in task_motion:
            distance = agent_distance(index, row, col)
            start = max(decision_time, Agent_Data[index, 1])
            arrival = start + math.ceil(distance / Agent_Data[index, 6])
            finish = start + math.ceil(
                distance / Agent_Data[index, 6] + task_time[row, col]
            )
            task_motion[key] = arrival, finish, distance
        return task_motion[key]

    charge_motion = {}

    def charge_values(index, charge):
        row, col, _ = charge
        key = index, row, col
        if key not in charge_motion:
            distance = agent_distance(index, row, col)
            start = max(decision_time, Agent_Data[index, 1])
            charge_motion[key] = (
                start + math.ceil(distance / Agent_Data[index, 6]),
                distance,
            )
        return charge_motion[key]

    nodes = []
    node_set = set()
    node_parts = {}

    def agent_parts(index):
        row, col = positions[index]
        return index, row, col

    def add_node(node, parts):
        if node not in node_set:
            node_set.add(node)
            nodes.append(node)
            node_parts[node] = parts

    def add_wut(worker, uav, task):
        _, worker_finish, _ = task_values(worker, task)
        _, uav_finish, uav_distance = task_values(uav, task)
        row, col, task_label = task
        finish = max(worker_finish, uav_finish)
        consumption = math.ceil(uav_distance + nearest_charge[row, col])
        if (consumption <= Agent_Data[uav, 7]
                and finish <= Agent_Data[uav, 4]
                and finish <= Agent_Data[worker, 4]
                and finish <= TimeLimit):
            node = (agent_label("worker", worker), none,
                    agent_label("uav", uav), task_label, none)
            add_node(node, (
                agent_parts(worker), (), agent_parts(uav), (row, col), ()
            ))

    def add_cus(car, uav, charge):
        car_arrival, _ = charge_values(car, charge)
        uav_arrival, uav_distance = charge_values(uav, charge)
        uav_consumption = math.ceil(uav_distance)
        remaining_power = Agent_Data[uav, 7] - uav_consumption
        charge_time = math.ceil(
            (Agent_Data[uav, 8] - remaining_power) / Agent_Data[car, 8]
        )
        finish = max(car_arrival, uav_arrival) + charge_time
        if (uav_consumption <= Agent_Data[uav, 7]
                and finish <= Agent_Data[uav, 4]
                and finish <= Agent_Data[car, 4]
                and finish <= TimeLimit):
            node = (none, agent_label("car", car),
                    agent_label("uav", uav), none, charge[2])
            add_node(node, (
                (), agent_parts(car), agent_parts(uav), (),
                (charge[0], charge[1])
            ))

    # 保持原构图的节点类型和插入顺序。
    for worker in workers:
        for uav in uavs:
            for task in tasks:
                add_wut(worker, uav, task)

    for car in cars:
        for uav in uavs:
            for charge in charges:
                add_cus(car, uav, charge)

    for worker in workers:
        for task in tasks:
            arrival, _, _ = task_values(worker, task)
            if arrival <= Agent_Data[worker, 4] and arrival <= TimeLimit:
                node = (agent_label("worker", worker), none, none,
                        task[2], none)
                add_node(node, (
                    agent_parts(worker), (), (), (task[0], task[1]), ()
                ))

    for uav in uavs:
        for task in tasks:
            arrival, _, distance = task_values(uav, task)
            row, col, task_label = task
            consumption = math.ceil(distance + nearest_charge[row, col])
            if (consumption <= Agent_Data[uav, 7]
                    and arrival <= Agent_Data[uav, 4]
                    and arrival <= TimeLimit):
                node = (none, none, agent_label("uav", uav),
                        task_label, none)
                add_node(node, (
                    (), (), agent_parts(uav), (row, col), ()
                ))

    for car in cars:
        for charge in charges:
            arrival, _ = charge_values(car, charge)
            if arrival <= Agent_Data[car, 4] and arrival <= TimeLimit:
                node = (none, agent_label("car", car), none, none,
                        charge[2])
                add_node(node, (
                    (), agent_parts(car), (), (), (charge[0], charge[1])
                ))

    for uav in uavs:
        for charge in charges:
            arrival, distance = charge_values(uav, charge)
            if (math.ceil(distance) <= Agent_Data[uav, 7]
                    and arrival <= Agent_Data[uav, 4]
                    and arrival <= TimeLimit):
                node = (none, none, agent_label("uav", uav), none,
                        charge[2])
                add_node(node, (
                    (), (), agent_parts(uav), (), (charge[0], charge[1])
                ))

    for worker in workers:
        node = (agent_label("worker", worker), none, none, none, none)
        add_node(node, (agent_parts(worker), (), (), (), ()))
    for uav in uavs:
        node = (none, none, agent_label("uav", uav), none, none)
        add_node(node, ((), (), agent_parts(uav), (), ()))
    for car in cars:
        node = (none, agent_label("car", car), none, none, none)
        add_node(node, ((), agent_parts(car), (), (), ()))

    graph = nx.Graph()
    graph.add_nodes_from(
        (node, {"parts": node_parts[node]}) for node in nodes
    )

    print(len(graph.nodes))
    return graph


def build_conflict_graph(candidate_graph, selected_nodes=None):
    """只为选中的候选节点建立冲突边。"""
    if selected_nodes is None:
        nodes = list(candidate_graph.nodes)
    else:
        selected = set(selected_nodes)
        nodes = [node for node in candidate_graph.nodes if node in selected]

    graph = nx.Graph()
    graph.add_nodes_from(
        (node, dict(candidate_graph.nodes[node])) for node in nodes
    )
    none = "None"

    # 按冲突资源建立索引，避免对所有节点做 O(V²) 两两判断。
    by_worker = defaultdict(list)
    by_uav = defaultdict(list)
    by_task = defaultdict(list)
    cus_by_charge_car = defaultdict(lambda: defaultdict(list))
    cs_by_charge = defaultdict(list)
    charged_by_car_charge = defaultdict(lambda: defaultdict(list))

    for node in nodes:
        if node[0] != none:
            by_worker[node[0]].append(node)
        if node[2] != none:
            by_uav[node[2]].append(node)
        if node[3] != none:
            by_task[node[3]].append(node)
        if node[4] != none and node[1] != none:
            charged_by_car_charge[node[1]][node[4]].append(node)
            if node[2] != none:
                cus_by_charge_car[node[4]][node[1]].append(node)
            else:
                cs_by_charge[node[4]].append(node)

    node_id = {node: index for index, node in enumerate(nodes)}
    edge_batch = []

    for index, node in enumerate(nodes):
        neighbors = set()
        if node[0] != none:
            neighbors.update(by_worker[node[0]])
        if node[2] != none:
            neighbors.update(by_uav[node[2]])
        if node[3] != none:
            neighbors.update(by_task[node[3]])

        charge, car, uav = node[4], node[1], node[2]
        if charge != none and car != none:
            if uav != none:
                # CUS：同充电点的其他车辆，以及该充电点全部 CS。
                for other_car, group in cus_by_charge_car[charge].items():
                    if other_car != car:
                        neighbors.update(group)
                neighbors.update(cs_by_charge[charge])
            else:
                # CS：与同充电点全部 CUS 冲突。
                for group in cus_by_charge_car[charge].values():
                    neighbors.update(group)

            # 同一车辆不能被分配到不同充电点。
            for other_charge, group in charged_by_car_charge[car].items():
                if other_charge != charge:
                    neighbors.update(group)

        neighbors.discard(node)
        edge_batch.extend(
            (node, neighbor)
            for neighbor in neighbors
            if node_id[neighbor] > index
        )
        if len(edge_batch) >= 200_000:
            graph.add_edges_from(edge_batch)
            edge_batch.clear()

    if edge_batch:
        graph.add_edges_from(edge_batch)

    return graph


def build_graph(Env, Agent_Data, task_time, CurrTimeAge, SingleStep, TimeLimit,
                faulty_agents=None):
    """兼容旧接口：生成与 BGUP-Cache 等价的完整冲突图。"""
    candidates = build_candidate_graph(
        Env, Agent_Data, task_time, CurrTimeAge, SingleStep, TimeLimit,
        faulty_agents=faulty_agents,
    )
    return build_conflict_graph(candidates)


def creatWeight(G, Env, agent_data, task_time, CurrTimeAge, SingleStep):
    """计算节点权重，并按唯一坐标复用空间势能与点对距离。"""
    task_coords = set()
    charge_coords = set()
    uav_coords = set()

    for node, data in G.nodes(data=True):
        parts = data.get("parts")
        if parts is None:
            parts = tuple(
                tuple(map(int, re.findall(r'\d+', item)))
                for item in node
            )
            data["parts"] = parts
        worker, car, uav, task, charging = parts

        if worker and task:
            task_coords.update(((worker[1], worker[2]), task))
        if uav and task:
            uav_coords.update(((uav[1], uav[2]), task))
        if car and charging:
            charge_coords.update(((car[1], car[2]), charging))
        if uav and charging:
            uav_coords.update(((uav[1], uav[2]), charging))

    task_grid = Env[1]
    charge_grid = Env[0]
    task_potential = {
        point: calculate_average_distance(task_grid, *point)
        for point in task_coords
    }
    charge_potential = {
        point: calculate_average_distance(charge_grid, *point)
        for point in charge_coords
    }
    uav_potential = {
        point: calculate_uav_average_distance(
            charge_grid, task_grid, *point
        )
        for point in uav_coords
    }

    distance_cache = {}

    def distance(target, agent):
        key = target[0], target[1], agent[1], agent[2]
        if key not in distance_cache:
            distance_cache[key] = calculate_distance(*key)
        return distance_cache[key]

    decision_time = (CurrTimeAge - 1) * SingleStep
    for num, (node, data) in enumerate(G.nodes(data=True), start=1):
        worker, car, uav, task, charging = data["parts"]
        # WUT
        if worker and not car and uav and task and not charging:
            w_distance = task_potential[worker[1], worker[2]]
            u_distance = uav_potential[uav[1], uav[2]]
            w_next_distance = task_potential[task]
            u_next_distance = uav_potential[task]
            e_w = relative_improvement(w_distance, w_next_distance)
            e_u = relative_improvement(u_distance, u_next_distance)

            # 到达任务点+完成任务所需时间
            w_mint = math.ceil(distance(task, worker) / agent_data[worker[0], 6] + task_time[task[0], task[1]])
            u_mint = math.ceil(distance(task, uav) / agent_data[uav[0], 6] + task_time[task[0], task[1]])
            # 开始时间
            w_start = max(decision_time, agent_data[worker[0], 1])
            u_start = max(decision_time, agent_data[uav[0], 1])
            # 任务结束后时间
            mint = max(w_start + w_mint, u_start + u_mint)
            t = max(math.ceil((mint - decision_time) / SingleStep), 1)

            p_d = math.ceil(distance(task, uav))
            p_f = agent_data[uav[0], 8]
            G.nodes[node]['weight'] = ((1 - p_d / p_f) / t) * (1 + e_w) + ((1 - p_d / p_f) / t) * (1 + e_u)
            G.nodes[node]['num'] = num
        # CUS
        if not worker and car and uav and not task and charging:
            c_distance = charge_potential[car[1], car[2]]
            u_distance = uav_potential[uav[1], uav[2]]
            c_next_distance = charge_potential[charging]
            u_next_distance = uav_potential[charging]
            e_c = relative_improvement(c_distance, c_next_distance)
            e_u = relative_improvement(u_distance, u_next_distance)

            # 完成充电所需时间
            uav_consumption = math.ceil(distance(charging, uav))
            remaining_power = agent_data[uav[0], 7] - uav_consumption
            charge_time = math.ceil((agent_data[uav[0], 8] - remaining_power) / agent_data[car[0], 8])

            # 到达位置+完成充电所需时间
            c_mint = math.ceil(distance(charging, car) / agent_data[car[0], 6] + charge_time)
            u_mint = math.ceil(distance(charging, uav) / agent_data[uav[0], 6] + charge_time)
            # 开始时刻
            c_start = max(decision_time, agent_data[car[0], 1])
            u_start = max(decision_time, agent_data[uav[0], 1])
            # 完成充电后时刻
            mint = max(c_start + c_mint, u_start + u_mint)
            t = max(math.ceil((mint - decision_time) / SingleStep), 1)

            p_f = agent_data[uav[0], 8]
            p_c = agent_data[uav[0], 7]
            G.nodes[node]['weight'] = ((1 - p_c / p_f) / t) * (1 + e_c) + ((1 - p_c / p_f) / t) * (1 + e_u)
            G.nodes[node]['num'] = num
        # WT
        if worker and not car and not uav and task and not charging:
            w_distance = task_potential[worker[1], worker[2]]
            w_next_distance = task_potential[task]
            e_w = relative_improvement(w_distance, w_next_distance)
            w_t = math.ceil(distance(task, worker) / (agent_data[worker[0], 6] * SingleStep))
            t = max(w_t, 1)
            G.nodes[node]['weight'] = (1 / t) * (1 + e_w) * 0.1
            G.nodes[node]['num'] = num
        # UT
        if not worker and not car and uav and task and not charging:
            u_distance = uav_potential[uav[1], uav[2]]
            u_next_distance = uav_potential[task]
            e_u = relative_improvement(u_distance, u_next_distance)
            u_t = math.ceil(distance(task, uav) / (agent_data[uav[0], 6] * SingleStep))
            t = max(u_t, 1)
            p_d = math.ceil(distance(task, uav))
            p_f = agent_data[uav[0], 8]
            G.nodes[node]['weight'] = ((1 - p_d / p_f) / t) * (1 + e_u) * 0.1
            G.nodes[node]['num'] = num
        # CS
        if not worker and car and not uav and not task and charging:
            c_distance = charge_potential[car[1], car[2]]
            c_next_distance = charge_potential[charging]
            e_c = relative_improvement(c_distance, c_next_distance)
            c_t = math.ceil(distance(charging, car) / (agent_data[car[0], 6] * SingleStep))
            t = max(c_t, 1)
            G.nodes[node]['weight'] = (1 / t) * (1 + e_c) * 0.1
            G.nodes[node]['num'] = num
        # US
        if not worker and not car and uav and not task and charging:
            u_distance = uav_potential[uav[1], uav[2]]
            u_next_distance = uav_potential[charging]
            e_u = relative_improvement(u_distance, u_next_distance)
            u_t = math.ceil(distance(charging, uav) / (agent_data[uav[0], 6] * SingleStep))
            t = max(u_t, 1)
            p_d = math.ceil(distance(charging, uav))
            p_f = agent_data[uav[0], 8]
            G.nodes[node]['weight'] = ((1 - p_d / p_f) / t) * (1 + e_u) * 0.1
            G.nodes[node]['num'] = num
        # W
        if worker and not car and not uav and not task and not charging:
            G.nodes[node]['weight'] = 0.01
            G.nodes[node]['num'] = num
        # U
        if not worker and not car and uav and not task and not charging:
            G.nodes[node]['weight'] = 0.01
            G.nodes[node]['num'] = num
        # C
        if not worker and car and not uav and not task and not charging:
            G.nodes[node]['weight'] = 0.01
            G.nodes[node]['num'] = num
    return G

def calculate_distance(destination_x, destination_y, x, y):
    distance = math.sqrt((destination_x - x) ** 2 + (destination_y - y) ** 2)
    return distance

def calculate_average_distance(matrix, x, y):

    total_distance = 0
    count_ones = 0

    # 遍历矩阵中的每个元素
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                distance = math.sqrt((i - x) ** 2 + (j - y) ** 2)
                total_distance += distance
                count_ones += 1

    # 如果没有1存在，返回0或其他适当值
    if count_ones == 0:
        return 0.0

    return total_distance / count_ones

def calculate_uav_average_distance(matrix0, matrix1, x, y):
    Row = matrix0.shape[0]
    Lin = matrix0.shape[1]

    total_distance = 0
    count_ones = 0

    # 遍历矩阵中的每个元素
    for i in range(Row):
        for j in range(Lin):
            if matrix0[i][j] == 1:
                distance = math.sqrt((i - x) ** 2 + (j - y) ** 2)
                total_distance += distance
                count_ones += 1
            if matrix1[i][j] == 1:
                distance = math.sqrt((i - x) ** 2 + (j - y) ** 2)
                total_distance += distance
                count_ones += 1

    # 如果没有1存在，返回0或其他适当值
    if count_ones == 0:
        return 0.0

    return total_distance / count_ones

def updateEnv(Env, agent_data, task_time, best_set, CurrTimeAge, SingleStep,
              TimeLimit=None):
    env = np.copy(Env)
    agent_data = np.copy(agent_data)

    result = []
    uav_flight_dist_task = 0.0
    uav_flight_dist_charge = 0.0
    worker_dist = 0.0
    car_dist = 0.0
    charge_count = 0
    # task_responses: list of (T_complete - T_assign) for each completed task
    task_responses = []

    for group in best_set:
        group_numbers = []
        for item in group:
            numbers = list(map(int, re.findall(r'\d+', item)))
            group_numbers.append(numbers)
        result.append(group_numbers)

    # print(result)

    decision_time = (CurrTimeAge - 1) * SingleStep
    deadline = float('inf') if TimeLimit is None else TimeLimit
    charging_groups = {}
    other_actions = []
    for group in result:
        worker, car, uav, task, charging = group
        if not worker and car and uav and not task and charging:
            key = (car[0], charging[0], charging[1])
            charging_groups.setdefault(key, []).append(group)
        else:
            other_actions.append(group)

    # 同一补给车在同一充电点按 UAV 到达时间先后串行充电。
    for groups in charging_groups.values():
        car = groups[0][1]
        charging = groups[0][4]
        c_dist = calculate_distance(
            charging[0], charging[1], car[1], car[2]
        )
        car_ready = max(decision_time, agent_data[car[0], 1]) + math.ceil(
            c_dist / agent_data[car[0], 6]
        )
        queue = []
        for group in groups:
            uav = group[2]
            uav_dist = math.ceil(calculate_distance(
                charging[0], charging[1], uav[1], uav[2]
            ))
            remaining_power = agent_data[uav[0], 7] - uav_dist
            if remaining_power < 0:
                continue
            uav_arrival = max(
                decision_time, agent_data[uav[0], 1]
            ) + math.ceil(uav_dist / agent_data[uav[0], 6])
            charge_time = math.ceil(
                (agent_data[uav[0], 8] - remaining_power)
                / agent_data[car[0], 8]
            )
            queue.append((uav_arrival, uav[0], uav, uav_dist, charge_time))

        served = False
        for uav_arrival, _, uav, uav_dist, charge_time in sorted(queue):
            finish = max(car_ready, uav_arrival) + charge_time
            if finish > agent_data[car[0], 4] or finish > deadline:
                break
            if finish > agent_data[uav[0], 4]:
                continue
            agent_data[uav[0], 2] = charging[0] + 1
            agent_data[uav[0], 3] = charging[1] + 1
            agent_data[uav[0], 1] = finish
            agent_data[uav[0], 7] = agent_data[uav[0], 8]
            car_ready = finish
            uav_flight_dist_charge += uav_dist
            charge_count += 1
            served = True

        if served:
            agent_data[car[0], 2] = charging[0] + 1
            agent_data[car[0], 3] = charging[1] + 1
            agent_data[car[0], 1] = car_ready
            car_dist += c_dist

    result = other_actions

    for group in result:
        worker = group[0]
        car = group[1]
        uav = group[2]
        task = group[3]
        charging = group[4]
        # WUT
        if worker and not car and uav and task and not charging:
            uav_dist = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]))
            w_dist = calculate_distance(task[0], task[1], worker[1], worker[2])
            completion_cost = task_time[task[0], task[1]]
            # 位置更新
            agent_data[worker[0], 2] = task[0] + 1
            agent_data[worker[0], 3] = task[1] + 1
            agent_data[uav[0], 2] = task[0] + 1
            agent_data[uav[0], 3] = task[1] + 1
            # 续航更新
            agent_data[uav[0], 7] = agent_data[uav[0], 7] - uav_dist
            # 智能体起始时刻更新
            w_mint = math.ceil(calculate_distance(task[0], task[1], worker[1], worker[2]) / agent_data[worker[0], 6] + completion_cost)
            u_mint = math.ceil(uav_dist / agent_data[uav[0], 6] + completion_cost)
            w_start = max((CurrTimeAge - 1) * SingleStep, int(agent_data[worker[0], 1]))
            u_start = max((CurrTimeAge - 1) * SingleStep, int(agent_data[uav[0], 1]))
            mint = max(w_start + w_mint, u_start + u_mint)
            agent_data[worker[0], 1] = mint
            agent_data[uav[0], 1] = mint
            # 任务已被承诺给该组合，避免后续重复派发。
            env[1, task[0], task[1]] = 0
            task_time[task[0], task[1]] = 0
            # metrics tracking
            uav_flight_dist_task += uav_dist
            worker_dist += w_dist
            T_assign = (CurrTimeAge - 1) * SingleStep
            task_responses.append(mint - T_assign)
        # WT
        if worker and not car and not uav and task and not charging:
            w_dist = calculate_distance(task[0], task[1], worker[1], worker[2])
            # 位置更新
            agent_data[worker[0], 2] = task[0] + 1
            agent_data[worker[0], 3] = task[1] + 1
            # 智能体起始时刻更新
            w_mint = math.ceil(calculate_distance(task[0], task[1], worker[1], worker[2]) / agent_data[worker[0], 6])
            w_start = max((CurrTimeAge - 1) * SingleStep, int(agent_data[worker[0], 1]))
            mint = w_start + w_mint
            agent_data[worker[0], 1] = mint
            # metrics tracking
            worker_dist += w_dist
        # UT
        if not worker and not car and uav and task and not charging:
            uav_dist = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]))
            # 位置更新
            agent_data[uav[0], 2] = task[0] + 1
            agent_data[uav[0], 3] = task[1] + 1
            # 续航更新
            agent_data[uav[0], 7] = agent_data[uav[0], 7] - uav_dist
            # 智能体起始时刻更新
            u_mint = math.ceil(uav_dist / agent_data[uav[0], 6])
            u_start = max((CurrTimeAge - 1) * SingleStep, int(agent_data[uav[0], 1]))
            mint = u_start + u_mint
            agent_data[uav[0], 1] = mint
            # metrics tracking
            uav_flight_dist_task += uav_dist
        # CS
        if not worker and car and not uav and not task and charging:
            c_dist = calculate_distance(charging[0], charging[1], car[1], car[2])
            # 位置更新
            agent_data[car[0], 2] = charging[0] + 1
            agent_data[car[0], 3] = charging[1] + 1
            # 智能体起始时刻更新
            c_mint = math.ceil(calculate_distance(charging[0], charging[1], car[1], car[2]) / agent_data[car[0], 6])
            c_start = max((CurrTimeAge - 1) * SingleStep, int(agent_data[car[0], 1]))
            agent_data[car[0], 1] = c_start + c_mint
            # metrics tracking
            car_dist += c_dist
        # US
        if not worker and not car and uav and not task and charging:
            uav_dist = math.ceil(calculate_distance(charging[0], charging[1], uav[1], uav[2]))
            # 位置更新
            agent_data[uav[0], 2] = charging[0] + 1
            agent_data[uav[0], 3] = charging[1] + 1
            # 续航更新
            agent_data[uav[0], 7] = agent_data[uav[0], 7] - uav_dist
            # 智能体起始时刻更新
            u_mint = math.ceil(uav_dist / agent_data[uav[0], 6])
            u_start = max((CurrTimeAge - 1) * SingleStep, int(agent_data[uav[0], 1]))
            agent_data[uav[0], 1] = u_start + u_mint
            # metrics tracking
            uav_flight_dist_charge += uav_dist

    return env, agent_data, task_time, uav_flight_dist_task, uav_flight_dist_charge, charge_count, task_responses, worker_dist, car_dist


def min_euclidean_distance_to_one(grid, target_x, target_y):
    ones = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                ones.append((i, j))
    min_dist = float('inf')
    for (x, y) in ones:
        d = math.sqrt((x - target_x)**2 + (y - target_y)**2)
        if d < min_dist:
            min_dist = d
    return min_dist

def isConflict(node1, node2):
    if (node1[0] != 'None' and node2[0] != 'None' and node1[0] == node2[0]):
        return True
    if (node1[2] != 'None' and node2[2] != 'None' and node1[2] == node2[2]):
        return True
    if (node1[3] != 'None' and node2[3] != 'None' and node1[3] == node2[3]):
        return True
    if (node1[4] != 'None' and node2[4] != 'None' and node1[4] == node2[4]):# 同一个充电点
        if (node1[1] != 'None' and node2[1] != 'None' and node1[2] != 'None' and node2[2] != 'None'):# CUS-CUS车辆不同有冲突
            if (node1[1] != node2[1]):
                return True
            else:
                return False
        if (node1[1] != 'None' and node2[1] != 'None' and node1[2] == 'None' and node2[2] == 'None'):# CS-CS无冲突
            return False
        if (node1[1] == 'None' and node2[1] == 'None' and node1[2] != 'None' and node2[2] != 'None'):# US-US无冲突
            return False
        if (node1[1] != 'None' and node2[1] != 'None' and node1[2] != 'None' and node2[2] == 'None'):# CUS-CS
            return True
        if (node1[1] != 'None' and node2[1] != 'None' and node1[2] == 'None' and node2[2] != 'None'):# CS-CUS
            return True
        if (node1[1] != 'None' and node2[1] == 'None' and node1[2] != 'None' and node2[2] != 'None'):# CUS-US
            return False
        if (node1[1] == 'None' and node2[1] != 'None' and node1[2] != 'None' and node2[2] != 'None'):# US-CUS
            return False
        if (node1[1] == 'None' and node2[1] != 'None' and node1[2] != 'None' and node2[2] == 'None'):# US-CS
            return False
        if (node1[1] != 'None' and node2[1] == 'None' and node1[2] == 'None' and node2[2] != 'None'):# CS-US
            return False
    if (node1[4] != 'None' and node2[4] != 'None' and node1[4] != node2[4]):
        if (node1[1] != 'None' and node2[1] != 'None' and node1[2] != 'None' and node2[2] != 'None'):  # CUS-CUS
            if (node1[1] == node2[1]):  # 如果是同一辆车，即使在不同充电桩也是冲突的
                return True
            else:
                return False
        if (node1[1] != 'None' and node2[1] != 'None' and node1[2] == 'None' and node2[2] == 'None'):  # CS-CS
            if (node1[1] == node2[1]):
                return True
            else:
                return False
        if (node1[1] != 'None' and node2[1] != 'None' and node1[2] != 'None' and node2[2] == 'None'):  # CUS-CS
            if (node1[1] == node2[1]):
                return True
            else:
                return False
        if (node1[1] != 'None' and node2[1] != 'None' and node1[2] == 'None' and node2[2] != 'None'):  # CS-CUS
            if (node1[1] == node2[1]):
                return True
            else:
                return False
    return False
