import random
import math
import numpy as np

# --- K-WTA 相关的辅助常量和函数 ---

# K-近邻候选点数量，用于UAV选择任务点或充电点
K_UAV_CANDIDATES = 5
# K-近邻候选点数量，用于Worker和Car匹配
K_HELPER_CANDIDATES = 5
# 用于效用计算的系数
K_URGENCY_FACTOR = 1.0


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


def _calculate_distance(x1, y1, x2, y2):
    """计算欧几里得距离"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _get_k_closest(grid, current_x, current_y, k):
    """获取距离当前位置最近的 K 个有效点 (Env grid value = 1)"""
    points = []
    # 遍历 Env grid (0-based)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                # 计算距离
                dist = _calculate_distance(i, j, current_x, current_y)
                points.append((dist, i, j))

    # 按距离排序并取前 K 个
    points.sort(key=lambda x: x[0])
    top_k = [(p[1], p[2]) for p in points[:k]]
    return top_k


def _get_k_closest_agents(agent_type, target_r, target_l, k, Agent_Data, centerRow, centerLin, Center_Scope):
    """
    寻找距离目标点 (target_r, target_l) 最近的 K 个可用智能体 (Worker/Car)
    这些智能体必须在决策中心范围内。
    返回: 智能体索引 (index) 列表
    """

    agents_list = []

    for i in range(Agent_Data.shape[0]):
        agent = Agent_Data[i, :]
        agent_r, agent_l = agent[2] - 1, agent[3] - 1  # 0-based

        # 1. 检查类型
        if agent[5] != agent_type:
            continue

        # 2. 检查是否在调度中心范围内
        is_in_scope = (centerRow <= agent_r < centerRow + Center_Scope and
                       centerLin <= agent_l < centerLin + Center_Scope)
        if not is_in_scope:
            continue

        dist = _calculate_distance(agent_r, agent_l, target_r, target_l)
        agents_list.append((dist, i))  # Store (distance, index)

    # 按距离排序并取前 K 个
    agents_list.sort(key=lambda x: x[0])
    top_k_indices = [idx for dist, idx in agents_list[:k]]

    return top_k_indices


def _calculate_utility(uav_data, target_data, target_type, Task_Time, current_time, Agent_Data, Env):
    """
    计算无人机的效用 (UAV Type 3)。使用简化逻辑：只看最近的目标和最近的助手。
    这部分逻辑仅用于UAV决定是去任务还是去充电的分组，不用于最终匹配。
    """
    uPow = uav_data[7]  # 当前电量
    Fullpower = uav_data[8]  # 满电量
    speed = uav_data[6]  # 移动速度
    uav_pos_r, uav_pos_l = uav_data[2] - 1, uav_data[3] - 1  # 0-based

    # 找到最近的目标点
    target_pos = []
    for r in range(target_data.shape[0]):
        for l in range(target_data.shape[1]):
            if target_data[r, l] == 1:
                target_pos.append((r, l))

    if not target_pos:
        return 0, (int(uav_pos_r), int(uav_pos_l))

    min_dist = float('inf')
    best_target_pos = None
    for r, l in target_pos:
        dist = _calculate_distance(uav_pos_r, uav_pos_l, r, l)
        if dist < min_dist:
            min_dist = dist
            best_target_pos = (r, l)

    if min_dist == float('inf') or speed == 0:
        return 0, (int(uav_pos_r), int(uav_pos_l))

    # 电量检查
    if target_type == 'Task':
        reserve_dist = min_euclidean_distance_to_one(Env[0], best_target_pos[0], best_target_pos[1])
        u_consum = math.ceil(min_dist + reserve_dist)
    else:  # Charge
        u_consum = math.ceil(min_dist)

    if u_consum > uPow:
        return 0, (int(uav_pos_r), int(uav_pos_l))

    # 1. 计算移动到目标的时间
    t_uav_move = min_dist / speed

    # 2. 计算任务/充电时间
    if target_type == 'Task':
        task_r, task_l = best_target_pos
        T_cost = Task_Time[task_r, task_l]

        # 寻找最近的 Worker (Type 1)
        workers = Agent_Data[Agent_Data[:, 5] == 1]
        if len(workers) == 0:
            t_worker_move = float('inf')
        else:
            worker_pos = workers[:, 2:4] - 1
            worker_speeds = workers[:, 6]
            dist_to_task = np.linalg.norm(worker_pos - np.array([task_r, task_l]), axis=1)
            min_dist_to_task = np.min(dist_to_task)
            min_worker_speed = worker_speeds[np.argmin(dist_to_task)]
            t_worker_move = min_dist_to_task / min_worker_speed if min_worker_speed > 0 else float('inf')

        Total_Time = max(t_uav_move, t_worker_move) + T_cost
        delta_pow = min_dist

    elif target_type == 'Charge':
        charge_r, charge_l = best_target_pos

        # 寻找最近的 Car (Type 2)
        cars = Agent_Data[Agent_Data[:, 5] == 2]
        if len(cars) == 0:
            t_car_move = float('inf')
            Car_Charge_Power = 0
        else:
            car_pos = cars[:, 2:4] - 1
            car_speeds = cars[:, 6]
            car_charge_pow = cars[:, 8]
            dist_to_charge = np.linalg.norm(car_pos - np.array([charge_r, charge_l]), axis=1)
            min_dist_to_charge = np.min(dist_to_charge)
            min_car_speed = car_speeds[np.argmin(dist_to_charge)]
            Car_Charge_Power = car_charge_pow[np.argmin(dist_to_charge)]
            t_car_move = min_dist_to_charge / min_car_speed if min_car_speed > 0 else float('inf')

        power_needed = max(0, Fullpower - uPow)
        t_charge = power_needed / Car_Charge_Power if Car_Charge_Power > 0 else float('inf')

        Total_Time = max(t_uav_move, t_car_move) + t_charge
        delta_pow = -(Fullpower - uPow)

    else:
        return 0, (int(uav_pos_r), int(uav_pos_l))

    # 3. K-WTA 效用计算（Urgency & Reward）

    if Total_Time == 0 or Total_Time == float('inf'):
        return 0, (int(uav_pos_r), int(uav_pos_l))

    if target_type == 'Task':
        delta_urge = delta_pow / Fullpower
        lambda_factor = K_URGENCY_FACTOR
        Reward = (1 - lambda_factor * delta_urge) / Total_Time

    else:  # Charge
        delta_urge_prime = -delta_pow / Fullpower
        lambda_factor = K_URGENCY_FACTOR
        Reward = (lambda_factor * delta_urge_prime) / Total_Time

    return Reward, best_target_pos


def _calculate_total_time_cost(uav_data, helper_data, target_pos, Task_Time, target_type):
    """计算 (UAV, Helper, Target) 三元组的完成时间成本"""
    uav_r, uav_l = uav_data[2] - 1, uav_data[3] - 1
    helper_r, helper_l = helper_data[2] - 1, helper_data[3] - 1
    target_r, target_l = target_pos
    uav_speed = uav_data[6]
    helper_speed = helper_data[6]

    if uav_speed <= 0 or helper_speed <= 0:
        return float('inf')

    # 移动时间
    t_uav_move = _calculate_distance(uav_r, uav_l, target_r, target_l) / uav_speed
    t_helper_move = _calculate_distance(helper_r, helper_l, target_r, target_l) / helper_speed

    if target_type == 'Task':
        # 任务执行时间
        T_cost = Task_Time[target_r, target_l]
        Total_Time = max(t_uav_move, t_helper_move) + T_cost
    else:  # Charge
        # 充电时间
        uPow = uav_data[7]
        Fullpower = uav_data[8]
        Car_Charge_Power = helper_data[8]
        power_needed = max(0, Fullpower - uPow)
        t_charge = power_needed / Car_Charge_Power if Car_Charge_Power > 0 else float('inf')

        Total_Time = max(t_uav_move, t_helper_move) + t_charge

    return Total_Time


def action(Env, Agent_Data, Task_Time, centerRow, centerLin, Center_Scope, CurrTimeAge, SingleStep, TimeLimit):
    Row = Env.shape[1]
    Lin = Env.shape[2]
    Agent_Num = Agent_Data.shape[0]

    # 存储最终动作: 初始全部为原地待命 (1-based)
    final_actions = [(int(Agent_Data[i, 2]), int(Agent_Data[i, 3])) for i in range(Agent_Num)]

    # 筛选在决策中心范围内的可用智能体
    uav_indices = []

    for i in range(Agent_Num):
        agent = Agent_Data[i, :]
        # 智能体位置 (0-based)
        agent_r, agent_l = agent[2] - 1, agent[3] - 1

        # 检查是否在线且在调度范围内
        is_online = (agent[1] <= CurrTimeAge * SingleStep < agent[4])
        is_in_scope = (centerRow <= agent_r < centerRow + Center_Scope and
                       centerLin <= agent_l < centerLin + Center_Scope)

        if is_online and is_in_scope:
            if agent[5] == 3:  # 无人机
                uav_indices.append(i)

    # --- K-WTA Step 1: UAV 效用评估与分组 ---
    uav_task_candidates = []
    uav_charge_candidates = []

    for i in uav_indices:
        uav = Agent_Data[i, :]
        # 任务效用 (使用 Env[1] - 任务点)
        utility_task, _ = _calculate_utility(uav, Env[1, :, :], 'Task', Task_Time, CurrTimeAge * SingleStep, Agent_Data, Env)
        # 充电效用 (使用 Env[0] - 充电点)
        utility_charge, _ = _calculate_utility(uav, Env[0, :, :], 'Charge', Task_Time, CurrTimeAge * SingleStep,
                                               Agent_Data, Env)

        # 电量不够去任务也不够去充电，则不参与匹配（utility为0表示电量不足）
        if utility_task >= utility_charge and utility_task > 0:
            uav_task_candidates.append({'index': i, 'utility': utility_task})
        elif utility_charge > 0:
            uav_charge_candidates.append({'index': i, 'utility': utility_charge})

    # --- K-WTA Step 2: K-Nearest Neighbors & 构建匹配三元组 ---

    # 任务匹配三元组 (UAV, Worker, Task)
    task_triples = []
    for uav_candidate in uav_task_candidates:
        uav_idx = uav_candidate['index']
        uav_data = Agent_Data[uav_idx, :]
        uav_r, uav_l = uav_data[2] - 1, uav_data[3] - 1

        # 电量检查：飞到任务点+最近充电点保留电量
        # UAV 选择 K_UAV_CANDIDATES 个最近任务点
        nearest_tasks = _get_k_closest(Env[1, :, :], uav_r, uav_l, K_UAV_CANDIDATES)

        for task_r, task_l in nearest_tasks:
            # 电量检查
            dist_to_task = _calculate_distance(uav_r, uav_l, task_r, task_l)
            reserve_dist = min_euclidean_distance_to_one(Env[0], task_r, task_l)
            u_consum = math.ceil(dist_to_task + reserve_dist)
            if u_consum > uav_data[7]:
                continue

            target_pos = (task_r, task_l)

            # 找到 K_HELPER_CANDIDATES 个匹配该任务点的最近 Worker
            worker_indices = _get_k_closest_agents(1, task_r, task_l, K_HELPER_CANDIDATES, Agent_Data, centerRow,
                                                   centerLin, Center_Scope)

            for worker_idx in worker_indices:
                worker_data = Agent_Data[worker_idx, :]
                time_cost = _calculate_total_time_cost(uav_data, worker_data, target_pos, Task_Time, 'Task')

                if time_cost != float('inf'):
                    # 奖励为时间成本的倒数
                    reward = 1 / time_cost
                    task_triples.append({
                        'uav_idx': uav_idx,
                        'helper_idx': worker_idx,
                        'target_pos': target_pos,
                        'reward': reward,
                        'type': 'Task'
                    })

    # 续航匹配三元组 (UAV, Car, Charge)
    charge_triples = []
    for uav_candidate in uav_charge_candidates:
        uav_idx = uav_candidate['index']
        uav_data = Agent_Data[uav_idx, :]
        uav_r, uav_l = uav_data[2] - 1, uav_data[3] - 1

        # UAV 选择 K_UAV_CANDIDATES 个最近充电点
        nearest_charges = _get_k_closest(Env[0, :, :], uav_r, uav_l, K_UAV_CANDIDATES)

        for charge_r, charge_l in nearest_charges:
            # 电量检查：飞到充电点所需电量
            dist_to_charge = _calculate_distance(uav_r, uav_l, charge_r, charge_l)
            u_consum = math.ceil(dist_to_charge)
            if u_consum > uav_data[7]:
                continue

            target_pos = (charge_r, charge_l)

            # 找到 K_HELPER_CANDIDATES 个匹配该充电点的最近 Car
            car_indices = _get_k_closest_agents(2, charge_r, charge_l, K_HELPER_CANDIDATES, Agent_Data, centerRow,
                                                centerLin, Center_Scope)

            for car_idx in car_indices:
                car_data = Agent_Data[car_idx, :]
                time_cost = _calculate_total_time_cost(uav_data, car_data, target_pos, Task_Time, 'Charge')

                if time_cost != float('inf'):
                    # 奖励为时间成本的倒数
                    reward = 1 / time_cost
                    charge_triples.append({
                        'uav_idx': uav_idx,
                        'helper_idx': car_idx,
                        'target_pos': target_pos,
                        'reward': reward,
                        'type': 'Charge'
                    })

    # --- K-WTA Step 3: Winner-Takes-All 贪心匹配 ---

    all_triples = sorted(task_triples + charge_triples, key=lambda x: x['reward'], reverse=True)

    assigned_uavs = set()
    assigned_helpers = set()
    assigned_targets = set()

    for triple in all_triples:
        uav_idx = triple['uav_idx']
        helper_idx = triple['helper_idx']
        target_pos = triple['target_pos']

        # 检查三元组的三个元素是否都未被分配
        if (uav_idx not in assigned_uavs and
                helper_idx not in assigned_helpers and
                target_pos not in assigned_targets):
            # 标记分配 (WTA 胜利)
            assigned_uavs.add(uav_idx)
            assigned_helpers.add(helper_idx)
            assigned_targets.add(target_pos)

            # 更新动作 (1-based)
            target_r, target_l = target_pos
            final_actions[uav_idx] = (target_r + 1, target_l + 1)
            final_actions[helper_idx] = (target_r + 1, target_l + 1)

    # 对于未赢得匹配的 Workers/Cars，其动作保持为原地待命 (在 final_actions 初始化时已完成)
    return final_actions


def calculate_distance(destination_x, destination_y, x, y):
    """用于 updateEnv 的距离计算"""
    distance = math.sqrt((destination_x - x) ** 2 + (destination_y - y) ** 2)
    return distance


def updateEnv(Env, Agent_data, task_time, Actions, CurrTimeAge, SingleStep, TimeLimit):
    ROW = Env.shape[1]
    LIN = Env.shape[2]
    Agent_Num = Agent_data.shape[0]

    # 存储每个智能体移动到新位置所需的时间
    move_times = np.zeros(Agent_Num)
    # 移动距离 (for UAV energy tracking)
    move_dists = np.zeros(Agent_Num)

    uav_flight_dist_task = 0.0
    uav_flight_dist_charge = 0.0
    charge_count = 0
    task_responses = []

    # 1. 更新所有智能体的位置和移动时间 (基于 Actions)
    for Agent_Index in range(Agent_Num):
        # 存储旧位置
        old_r, old_l = Agent_data[Agent_Index, 2], Agent_data[Agent_Index, 3]  # 1-based
        # Actions 存储的是新位置 1-based (R, L)
        Action_Row, Action_Lin = Actions[Agent_Index]

        # 计算移动距离和时间
        dist = calculate_distance(old_r, old_l, Action_Row, Action_Lin)
        move_dists[Agent_Index] = dist
        time = dist / Agent_data[Agent_Index, 6]
        move_times[Agent_Index] = time

        # Track UAV flight distance - distinguish task vs charge
        if Agent_data[Agent_Index, 5] == 3:
            # Check if UAV is going to a charge point (Env[0]) or task point (Env[1])
            if Env[0, Action_Row - 1, Action_Lin - 1] == 1:
                uav_flight_dist_charge += dist
            elif Env[1, Action_Row - 1, Action_Lin - 1] == 1:
                uav_flight_dist_task += dist

        # 更新智能体位置 (1-based)
        Agent_data[Agent_Index, 2] = Action_Row
        Agent_data[Agent_Index, 3] = Action_Lin

    # 2. 任务/续航点完成和状态更新 (基于新的位置)

    completed_tasks_pos = set()

    for i in range(ROW):
        for ii in range(LIN):
            current_pos = (i, ii)  # 0-based

            if current_pos in completed_tasks_pos:
                continue

            # 统计当前位置的智能体
            workers = []
            cars = []
            uavs = []
            for Agent_Index in range(Agent_Num):
                is_online = (Agent_data[Agent_Index, 1] <= CurrTimeAge * SingleStep < Agent_data[Agent_Index, 4])

                if is_online and Agent_data[Agent_Index, 2] - 1 == i and Agent_data[Agent_Index, 3] - 1 == ii:
                    if Agent_data[Agent_Index, 5] == 1:  # 工人
                        workers.append(Agent_Index)
                    elif Agent_data[Agent_Index, 5] == 2:  # 补给车
                        cars.append(Agent_Index)
                    elif Agent_data[Agent_Index, 5] == 3:  # 无人机
                        uavs.append(Agent_Index)

            # --- 任务完成 (Task Completion) ---
            if Env[1, i, ii] == 1 and workers and uavs:

                if uavs and workers:
                    uav_idx = uavs[0]
                    worker_idx = workers[0]

                    t_uav_move = move_times[uav_idx]
                    t_worker_move = move_times[worker_idx]
                    T_cost = task_time[i, ii]

                    # UAV开始时刻（考虑等待之前任务完成）
                    u_start = max((CurrTimeAge - 1) * SingleStep, Agent_data[uav_idx, 1])
                    w_start = max((CurrTimeAge - 1) * SingleStep, Agent_data[worker_idx, 1])

                    # 完成时刻
                    completion_time = max(u_start + t_uav_move + T_cost,
                                          w_start + t_worker_move + T_cost)

                    # 电量扣除：飞到任务点距离 + 最近充电点保留距离
                    uav_move_dist = math.ceil(move_dists[uav_idx])
                    reserve_dist = min_euclidean_distance_to_one(Env[0], i, ii)
                    u_consum = math.ceil(move_dists[uav_idx] + reserve_dist)

                    uav_end_time = min(Agent_data[uav_idx, 4], TimeLimit)
                    worker_end_time = min(Agent_data[worker_idx, 4], TimeLimit)

                    if u_consum <= Agent_data[uav_idx, 7] and \
                       completion_time <= uav_end_time and completion_time <= worker_end_time:
                        # 任务完成
                        Env[1, i, ii] = 0
                        completed_tasks_pos.add(current_pos)

                        # UAV 电量扣除（与CpOL-AsHs一致：扣除飞到任务点的距离）
                        Agent_data[uav_idx, 7] -= uav_move_dist

                        # 更新智能体的起始时刻为任务完成时刻
                        Agent_data[uav_idx, 1] = completion_time
                        Agent_data[worker_idx, 1] = completion_time

                        # Track task response time
                        T_assign = (CurrTimeAge - 1) * SingleStep
                        task_responses.append(completion_time - T_assign)

            # --- 续航完成 (Charge Completion) ---
            if Env[0, i, ii] == 1 and cars and uavs:

                if uavs and cars:
                    car_idx = cars[0]

                    total_charge_time = 0
                    for uav_idx in uavs:
                        uPow = Agent_data[uav_idx, 7]
                        Fullpower = Agent_data[uav_idx, 8]
                        Car_Charge_Power = Agent_data[car_idx, 8]
                        power_needed = max(0, Fullpower - uPow)
                        charge_time = power_needed / Car_Charge_Power if Car_Charge_Power > 0 else 0
                        total_charge_time += charge_time

                    t_car_move = move_times[car_idx]
                    # Car开始时刻（考虑等待）
                    c_start = max((CurrTimeAge - 1) * SingleStep, Agent_data[car_idx, 1])
                    completion_time_car = c_start + t_car_move + total_charge_time

                    car_end_time = min(Agent_data[car_idx, 4], TimeLimit)

                    # 检查所有UAV电量是否足够飞到充电点
                    all_uav_can_reach = True
                    for uav_idx in uavs:
                        u_consum = math.ceil(move_dists[uav_idx])
                        if u_consum > Agent_data[uav_idx, 7]:
                            all_uav_can_reach = False
                            break

                    if all_uav_can_reach and completion_time_car <= car_end_time:

                        # Car 状态更新
                        Agent_data[car_idx, 1] = completion_time_car
                        completed_tasks_pos.add(current_pos)

                        # UAV 状态更新
                        for uav_idx in uavs:
                            t_uav_move = move_times[uav_idx]
                            # UAV电量扣除飞到充电点的距离（充电前扣除）
                            uav_move_dist = math.ceil(move_dists[uav_idx])
                            Agent_data[uav_idx, 7] -= uav_move_dist

                            # UAV开始时刻（考虑等待）
                            u_start = max((CurrTimeAge - 1) * SingleStep, Agent_data[uav_idx, 1])
                            uav_end_time = min(Agent_data[uav_idx, 4], TimeLimit)
                            completion_time_uav = u_start + t_uav_move + total_charge_time

                            if completion_time_uav <= uav_end_time:
                                Agent_data[uav_idx, 7] = Agent_data[uav_idx, 8]  # 满电
                                Agent_data[uav_idx, 1] = completion_time_uav
                                charge_count += 1

    return Env, Agent_data, uav_flight_dist_task, uav_flight_dist_charge, charge_count, task_responses


def select_optimal_region(Env, agent_data, center_scope, CurrTimeAge, SingleStep):
    """选择最佳决策中心 (保持原有逻辑)"""
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
                if CurrTimeAge * SingleStep >= agent_data[i, 1] and CurrTimeAge * SingleStep < agent_data[i, 4]:
                    # 找到智能体i的位置 (0-based)
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


def find_closest_one(grid, target_x, target_y):
    # 保持不变
    closest_point = None
    min_dist = float('inf')

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if i == target_x and j == target_y:
                continue
            if grid[i][j] == 1:
                dist = math.sqrt((i - target_x) ** 2 + (j - target_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (i, j)
    return closest_point
