import networkx as nx
import re
import numpy as np
import math


def relative_improvement(current_distance, next_distance):
    if current_distance == 0:
        return 0.0
    return math.tanh(
        (current_distance - next_distance) / current_distance
    )


def build_graph(Env, Agent_Data, task_time, CurrTimeAge, SingleStep, TimeLimit,
                faulty_agents=None):

    if faulty_agents is None:
        faulty_agents = set()

    Row = Env.shape[1]
    Lin = Env.shape[2]
    Agent_Num = Agent_Data.shape[0]
    decision_time = (CurrTimeAge - 1) * SingleStep
    workers = []
    cars = []
    uavs = []
    tasks = []
    charging = []
    for i in range(Agent_Num):
        # 故障智能体不可调度
        if i in faulty_agents:
            continue
        # 全局调度：所有在线且未故障的智能体均可选
        if Agent_Data[i, 1] <= decision_time < Agent_Data[i, 4]:
            if Agent_Data[i, 5] == 1:
                workers.append("worker" + str(i) + "|" + str(int(Agent_Data[i, 2] - 1)) + "," + str(int(Agent_Data[i, 3] - 1)))
            if Agent_Data[i, 5] == 2:
                cars.append("car" + str(i) + "|" + str(int(Agent_Data[i, 2] - 1)) + "," + str(int(Agent_Data[i, 3] - 1)))
            if Agent_Data[i, 5] == 3:
                uavs.append("uav" + str(i) + "|" + str(int(Agent_Data[i, 2] - 1)) + "," + str(int(Agent_Data[i, 3] - 1)))

    for i in range(Row):
        for j in range(Lin):
            # 所有任务和充电点可选
            if Env[1, i, j] == 1:
                new_task = "task" + str(i) + "," + str(j)
                if new_task not in tasks:
                    tasks.append(new_task)
            if Env[0, i, j] == 1:
                new_charging = "charging" + str(i) + "," + str(j)
                if new_charging not in charging:
                    charging.append(new_charging)


    print("Workers:", workers)
    print("Cars:", cars)
    print("Uavs:", uavs)
    print("Tasks:", tasks)
    print("Charging:", charging)


    nodes = []
    #WUT
    for w in workers:
        for u in uavs:
            for t in tasks:
                nodes.append((w, 'None', u, t, 'None'))
    #CUS
    for c in cars:
        for u in uavs:
            for s in charging:
                nodes.append(('None', c, u, 'None', s))
    #WT
    for w in workers:
        for t in tasks:
            nodes.append((w, 'None', 'None', t, 'None'))
    #UT
    for u in uavs:
        for t in tasks:
            nodes.append(('None', 'None', u, t, 'None'))
    #CS
    for c in cars:
        for s in charging:
            nodes.append(('None', c, 'None', 'None', s))
    #US
    for u in uavs:
        for s in charging:
            nodes.append(('None', 'None', u, 'None', s))
    #W
    for w in workers:
        nodes.append((w, 'None', 'None', 'None', 'None'))
    #U
    for u in uavs:
        nodes.append(('None', 'None', u, 'None', 'None'))
    #C
    for c in cars:
        nodes.append(('None', c, 'None', 'None', 'None'))

    G = nx.Graph()
    G.add_nodes_from(nodes)

    # 将续航不足以维持运动的结点或做任务时间不够的结点移除
    nodes_to_remove = []
    for node in G.nodes():
        node_numbers = []
        for item in node:
            numbers = list(map(int, re.findall(r'\d+', item)))
            node_numbers.append(numbers)
        worker = node_numbers[0]
        car = node_numbers[1]
        uav = node_numbers[2]
        task = node_numbers[3]
        charging = node_numbers[4]
        # WUT
        if worker and not car and uav and task and not charging:
            # 完成任务所需时间
            w_mint = math.ceil(calculate_distance(task[0], task[1], worker[1], worker[2]) / Agent_Data[worker[0], 6] + task_time[task[0], task[1]])
            u_mint = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]) / Agent_Data[uav[0], 6] + task_time[task[0], task[1]])
            # 开始时间
            w_start = max((CurrTimeAge - 1) * SingleStep, Agent_Data[worker[0], 1])
            u_start = max((CurrTimeAge - 1) * SingleStep, Agent_Data[uav[0], 1])
            # 任务结束后时间
            mint = max(w_start + w_mint, u_start + u_mint)
            # 无人机所耗续航
            u_consum = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]) + min_euclidean_distance_to_one(Env[0, :, :], task[0], task[1]))
            # 续航不够，完成任务超过下线时间的结点删除
            if u_consum > Agent_Data[uav[0], 7] or mint > Agent_Data[uav[0], 4] or mint > Agent_Data[worker[0], 4] or mint > TimeLimit:
                nodes_to_remove.append(node)
        # UT
        if not worker and not car and uav and task and not charging:
            # 到达位置所需时间
            u_mint = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]) / Agent_Data[uav[0], 6])
            # 无人机所耗续航
            u_consum = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]) + min_euclidean_distance_to_one(Env[0, :, :], task[0], task[1]))
            # 开始时间
            start_t = max((CurrTimeAge - 1) * SingleStep, Agent_Data[uav[0], 1])
            # 续航不够，到达位置超过下线时间的结点删除
            if u_consum > Agent_Data[uav[0], 7] or u_mint + start_t > Agent_Data[uav[0], 4] or u_mint + start_t > TimeLimit:
                nodes_to_remove.append(node)
        # WT
        if worker and not car and not uav and task and not charging:
            # 到达位置所需时间
            w_mint = math.ceil(calculate_distance(task[0], task[1], worker[1], worker[2]) / Agent_Data[worker[0], 6])
            # 开始时刻
            w_start = max((CurrTimeAge - 1) * SingleStep, Agent_Data[worker[0], 1])
            # 到达位置后超过下线时间的结点删除
            if w_start + w_mint > Agent_Data[worker[0], 4] or w_start + w_mint > TimeLimit:
                nodes_to_remove.append(node)
        # CUS
        if not worker and car and uav and not task and charging:
            # 到达位置所需时间
            c_mint = math.ceil(calculate_distance(charging[0], charging[1], car[1], car[2]) / Agent_Data[car[0], 6])
            u_mint = math.ceil(calculate_distance(charging[0], charging[1], uav[1], uav[2]) / Agent_Data[uav[0], 6])
            # 开始时刻
            c_start = max((CurrTimeAge - 1) * SingleStep, Agent_Data[car[0], 1])
            u_start = max((CurrTimeAge - 1) * SingleStep, Agent_Data[uav[0], 1])
            # 无人机所耗续航
            u_consum = math.ceil(calculate_distance(charging[0], charging[1], uav[1], uav[2]))
            remaining_power = Agent_Data[uav[0], 7] - u_consum
            charge_time = math.ceil((Agent_Data[uav[0], 8] - remaining_power) / Agent_Data[car[0], 8])
            # 完成充电后时刻
            mint = max(c_start + c_mint, u_start + u_mint) + charge_time
            # 续航不够，充电超过下线时间的结点删除
            if u_consum > Agent_Data[uav[0], 7] or mint > Agent_Data[uav[0], 4] or mint > Agent_Data[car[0], 4] or mint > TimeLimit:
                nodes_to_remove.append(node)
        # US
        if not worker and not car and uav and not task and charging:
            # 到达位置所需时间
            u_mint = math.ceil(calculate_distance(charging[0], charging[1], uav[1], uav[2]) / Agent_Data[uav[0], 6])
            # 无人机所耗续航
            u_consum = math.ceil(calculate_distance(charging[0], charging[1], uav[1], uav[2]))
            # 开始时刻
            start_t = max((CurrTimeAge - 1) * SingleStep, Agent_Data[uav[0], 1])
            # 续航不够，到达位置后超过下线时间的结点删除
            if u_consum > Agent_Data[uav[0], 7] or u_mint + start_t > Agent_Data[uav[0], 4] or u_mint + start_t > TimeLimit:
                nodes_to_remove.append(node)
        # CS
        if not worker and car and not uav and not task and charging:
            # 到达位置所需时间
            c_mint = math.ceil(calculate_distance(charging[0], charging[1], car[1], car[2]) / Agent_Data[car[0], 6])
            # 开始时刻
            c_start = max((CurrTimeAge - 1) * SingleStep, int(Agent_Data[car[0], 1]))
            # 到达位置后超过下线时间的结点删除
            if c_start + c_mint > Agent_Data[car[0], 4] or c_start + c_mint > TimeLimit:
                nodes_to_remove.append(node)

    for node in nodes_to_remove:
        if node in G:  # 确保节点仍在图中
            G.remove_node(node)

    print(len(G.nodes))
    nodes = list(G.nodes())
    #连边
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            if isConflict(node1, node2):
                G.add_edge(node1, node2)



    return G

def creatWeight(G, Env, agent_data, task_time, CurrTimeAge, SingleStep):
    num = 0
    for node in G.nodes():
        num = num + 1
        node_numbers = []
        for item in node:
            numbers = list(map(int, re.findall(r'\d+', item)))
            node_numbers.append(numbers)
        worker = node_numbers[0]
        car = node_numbers[1]
        uav = node_numbers[2]
        task = node_numbers[3]
        charging = node_numbers[4]
        # WUT
        if worker and not car and uav and task and not charging:
            w_distance = calculate_average_distance(Env[1], worker[1], worker[2])
            u_distance = calculate_uav_average_distance(Env[0], Env[1], uav[1], uav[2])
            w_next_distance = calculate_average_distance(Env[1], task[0], task[1])
            u_next_distance = calculate_uav_average_distance(Env[0], Env[1], task[0], task[1])
            e_w = relative_improvement(w_distance, w_next_distance)
            e_u = relative_improvement(u_distance, u_next_distance)

            # 到达任务点+完成任务所需时间
            w_mint = math.ceil(calculate_distance(task[0], task[1], worker[1], worker[2]) / agent_data[worker[0], 6] + task_time[task[0], task[1]])
            u_mint = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]) / agent_data[uav[0], 6] + task_time[task[0], task[1]])
            # 开始时间
            w_start = max((CurrTimeAge - 1) * SingleStep, agent_data[worker[0], 1])
            u_start = max((CurrTimeAge - 1) * SingleStep, agent_data[uav[0], 1])
            # 任务结束后时间
            mint = max(w_start + w_mint, u_start + u_mint)
            t = max(math.ceil((mint - (CurrTimeAge - 1) * SingleStep) / SingleStep), 1)

            p_d = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]))
            p_f = agent_data[uav[0], 8]
            G.nodes[node]['weight'] = ((1 - p_d / p_f) / t) * (1 + e_w) + ((1 - p_d / p_f) / t) * (1 + e_u)
            G.nodes[node]['num'] = num
        # CUS
        if not worker and car and uav and not task and charging:
            c_distance = calculate_average_distance(Env[0], car[1], car[2])
            u_distance = calculate_uav_average_distance(Env[0], Env[1], uav[1], uav[2])
            c_next_distance = calculate_average_distance(Env[0], charging[0], charging[1])
            u_next_distance = calculate_uav_average_distance(Env[0], Env[1], charging[0], charging[1])
            e_c = relative_improvement(c_distance, c_next_distance)
            e_u = relative_improvement(u_distance, u_next_distance)

            # 完成充电所需时间
            uav_consumption = math.ceil(calculate_distance(charging[0], charging[1], uav[1], uav[2]))
            remaining_power = agent_data[uav[0], 7] - uav_consumption
            charge_time = math.ceil((agent_data[uav[0], 8] - remaining_power) / agent_data[car[0], 8])

            # 到达位置+完成充电所需时间
            c_mint = math.ceil(calculate_distance(charging[0], charging[1], car[1], car[2]) / agent_data[car[0], 6] + charge_time)
            u_mint = math.ceil(calculate_distance(charging[0], charging[1], uav[1], uav[2]) / agent_data[uav[0], 6] + charge_time)
            # 开始时刻
            c_start = max((CurrTimeAge - 1) * SingleStep, agent_data[car[0], 1])
            u_start = max((CurrTimeAge - 1) * SingleStep, agent_data[uav[0], 1])
            # 完成充电后时刻
            mint = max(c_start + c_mint, u_start + u_mint)
            t = max(math.ceil((mint - (CurrTimeAge - 1) * SingleStep) / SingleStep), 1)

            p_f = agent_data[uav[0], 8]
            p_c = agent_data[uav[0], 7]
            G.nodes[node]['weight'] = ((1 - p_c / p_f) / t) * (1 + e_c) + ((1 - p_c / p_f) / t) * (1 + e_u)
            G.nodes[node]['num'] = num
        # WT
        if worker and not car and not uav and task and not charging:
            w_distance = calculate_average_distance(Env[1], worker[1], worker[2])
            w_next_distance = calculate_average_distance(Env[1], task[0], task[1])
            e_w = relative_improvement(w_distance, w_next_distance)
            w_t = math.ceil(calculate_distance(task[0], task[1], worker[1], worker[2]) / (agent_data[worker[0], 6] * SingleStep))
            t = max(w_t, 1)
            G.nodes[node]['weight'] = (1 / t) * (1 + e_w) * 0.1
            G.nodes[node]['num'] = num
        # UT
        if not worker and not car and uav and task and not charging:
            u_distance = calculate_uav_average_distance(Env[0], Env[1], uav[1], uav[2])
            u_next_distance = calculate_uav_average_distance(Env[0], Env[1], task[0], task[1])
            e_u = relative_improvement(u_distance, u_next_distance)
            u_t = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]) / (agent_data[uav[0], 6] * SingleStep))
            t = max(u_t, 1)
            p_d = math.ceil(calculate_distance(task[0], task[1], uav[1], uav[2]))
            p_f = agent_data[uav[0], 8]
            G.nodes[node]['weight'] = ((1 - p_d / p_f) / t) * (1 + e_u) * 0.1
            G.nodes[node]['num'] = num
        # CS
        if not worker and car and not uav and not task and charging:
            c_distance = calculate_average_distance(Env[0], car[1], car[2])
            c_next_distance = calculate_average_distance(Env[0], charging[0], charging[1])
            e_c = relative_improvement(c_distance, c_next_distance)
            c_t = math.ceil(calculate_distance(charging[0], charging[1], car[1], car[2]) / (agent_data[car[0], 6] * SingleStep))
            t = max(c_t, 1)
            G.nodes[node]['weight'] = (1 / t) * (1 + e_c) * 0.1
            G.nodes[node]['num'] = num
        # US
        if not worker and not car and uav and not task and charging:
            u_distance = calculate_uav_average_distance(Env[0], Env[1], uav[1], uav[2])
            u_next_distance = calculate_uav_average_distance(Env[0], Env[1], charging[0], charging[1])
            e_u = relative_improvement(u_distance, u_next_distance)
            u_t = math.ceil(calculate_distance(charging[0], charging[1], uav[1], uav[2]) / (agent_data[uav[0], 6] * SingleStep))
            t = max(u_t, 1)
            p_d = math.ceil(calculate_distance(charging[0], charging[1], uav[1], uav[2]))
            p_f = agent_data[uav[0], 8]
            G.nodes[node]['weight'] = ((1 - p_d / p_f) / t) * (1 + e_u) * 0.1
            G.nodes[node]['num'] = num
        # W
        if worker and not car and not uav and not task and not charging:
            w_distance = calculate_average_distance(Env[1], worker[1], worker[2])
            w_next_distance = calculate_average_distance(Env[1], worker[1], worker[2])
            e_w = relative_improvement(w_distance, w_next_distance)
            t = 1
            G.nodes[node]['weight'] = 0.01
            G.nodes[node]['num'] = num
        # U
        if not worker and not car and uav and not task and not charging:
            u_distance = calculate_uav_average_distance(Env[0], Env[1], uav[1], uav[2])
            u_next_distance = calculate_uav_average_distance(Env[0], Env[1], uav[1], uav[2])
            e_u = relative_improvement(u_distance, u_next_distance)
            t = 1
            G.nodes[node]['weight'] = 0.01
            G.nodes[node]['num'] = num
        # C
        if not worker and car and not uav and not task and not charging:
            c_distance = calculate_average_distance(Env[0], car[1], car[2])
            c_next_distance = calculate_average_distance(Env[0], car[1], car[2])
            e_c = relative_improvement(c_distance, c_next_distance)
            t = 1
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


def select_optimal_region(Env, agent_data, center_scope, CurrTimeAge, SingleStep):
    ROW = Env.shape[1]
    LIN = Env.shape[2]
    Agent_Num = agent_data.shape[0]
    max_agents = -1
    min_diff = float('inf')
    best_row, best_lin = 0, 0

    # 遍历所有可能的区域
    for row in range(ROW - center_scope + 1):
        for lin in range(LIN - center_scope + 1):
            # 初始化计数器
            total_agents = 0
            workers = 0
            uavs = 0

            # 遍历每个智能体
            for i in range(Agent_Num):
                if CurrTimeAge*SingleStep >= agent_data[i, 1] and CurrTimeAge*SingleStep < agent_data[i, 4]:
                    # 找到智能体i的位置
                    x, y = agent_data[i, 2] - 1, agent_data[i, 3] - 1
                    # 检查是否在当前区域内
                    if row <= x < row + center_scope and lin <= y < lin + center_scope:
                        total_agents += 1
                        if agent_data[i, 5] == 1:  # 工人
                            workers += 1
                        elif agent_data[i, 5] == 3:  # 无人机
                            uavs += 1

            # 计算工人和无人机的数量差
            diff = abs(workers - uavs)

            # 更新最优区域
            if total_agents > max_agents:
                min_diff = diff
                max_agents = total_agents
                best_row, best_lin = row, lin
            elif total_agents == max_agents:
                if diff < min_diff:
                    min_diff = diff
                    best_row, best_lin = row, lin

    return best_row, best_lin

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
