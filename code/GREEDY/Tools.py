import random
import math
import numpy as np

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

def utility_task(uav, Env, Task_Time, workers, CurrTime, SingleStep, TimeLimit):
    task_pos = find_closest_one(Env[1], int(uav[2]-1), int(uav[3]-1))
    if task_pos is None or len(workers) == 0:
        return 0

    # 电量检查：UAV飞到任务点+再飞到最近充电点所需电量
    d_u = calculate_distance(uav[2]-1, uav[3]-1, task_pos[0], task_pos[1])
    reserve_dist = min_euclidean_distance_to_one(Env[0], task_pos[0], task_pos[1])
    u_consum = math.ceil(d_u + reserve_dist)
    if u_consum > uav[7]:
        return 0

    # 最近 worker
    wp = None
    min_wd = float('inf')
    for w in workers:
        d = calculate_distance(w[2]-1, w[3]-1, task_pos[0], task_pos[1])
        if d < min_wd:
            min_wd = d
            wp = w

    t_u = d_u / uav[6]
    t_w = min_wd / wp[6]

    task_time = Task_Time[task_pos[0], task_pos[1]]
    T_task = max(t_u, t_w) + task_time

    # 剩余电量
    E = uav[7]
    Emax = uav[8]

    urgency = (Emax - E) / Emax
    power_cost = d_u / Emax

    T_left = min(TimeLimit, uav[4]) - CurrTime*SingleStep
    T_on = min(TimeLimit, uav[4]) - uav[1]
    T_left_p = T_left - T_task - SingleStep

    if T_left_p <= 0:
        return 0

    x = T_left_p / T_on
    lambda_factor = x * math.exp(-(1 - x))

    return (1 - lambda_factor * power_cost) / T_task


def utility_charge(uav, Env, cars, CurrTimeAge, SingleStep, TimeLimit):

    charge_pos = find_closest_one(Env[0], int(uav[2]-1), int(uav[3]-1))
    if charge_pos is None or len(cars) == 0:
        return 0

    # 电量检查：UAV飞到充电点所需电量
    d_u = calculate_distance(uav[2]-1, uav[3]-1, charge_pos[0], charge_pos[1])
    u_consum = math.ceil(d_u)
    if u_consum > uav[7]:
        return 0

    # 最近 car
    cp = None
    min_cd = float('inf')
    for c in cars:
        d = calculate_distance(c[2]-1, c[3]-1, charge_pos[0], charge_pos[1])
        if d < min_cd:
            min_cd = d
            cp = c

    t_u = d_u / uav[6]
    t_c = min_cd / cp[6]

    # === 电量参数 ===
    E = max(uav[7], 0)
    Emax = max(uav[8], 1e-6)

    charge_power = max(cp[8], 1e-6)

    charge_time = max((Emax - E) / charge_power, 0)

    T_charge = max(t_u, t_c) + charge_time

    # === 防御 T_charge = 0 ===
    if T_charge <= 1e-6:
        return 0

    T_left = max(min(TimeLimit, uav[4]) - CurrTimeAge*SingleStep, 0)
    T_on = max(min(TimeLimit, uav[4]) - uav[1], 1e-6)

    T_left_p = T_left - T_charge - SingleStep
    if T_left_p <= 0:
        return 0

    urgency = (Emax - E) / Emax

    x = T_left_p / T_on
    x = max(min(x, 1), 0)

    lambda_factor = x * math.exp(-(1 - x))

    return (lambda_factor * urgency) / T_charge



# ========== 主调度函数 ==========
def action(Env, Agent_Data, Task_Time, centerRow, centerLin, Center_Scope, CurrTimeAge, SingleStep, TimeLimit):

    actions = []
    workers = [a for a in Agent_Data if a[5] == 1]
    cars = [a for a in Agent_Data if a[5] == 2]

    for i in range(len(Agent_Data)):
        agent = Agent_Data[i]
        cur_r = int(agent[2]-1)
        cur_l = int(agent[3]-1)

        online = agent[1] <= CurrTimeAge*SingleStep < agent[4]
        in_center = centerRow <= cur_r < centerRow + Center_Scope and centerLin <= cur_l < centerLin + Center_Scope

        # 不可调度直接原地
        if not online or not in_center:
            actions.append((cur_r, cur_l))
            continue

        # ========= 工人 =========
        if agent[5] == 1:
            p = find_closest_one(Env[1], cur_r, cur_l)
            actions.append(p if p is not None else (cur_r, cur_l))
            continue

        # ========= 车辆 =========
        if agent[5] == 2:
            p = find_closest_one(Env[0], cur_r, cur_l)
            actions.append(p if p is not None else (cur_r, cur_l))
            continue

        # ========= UAV =========
        if agent[5] == 3:
            ut = utility_task(agent, Env, Task_Time, workers=workers,
                              CurrTime=CurrTimeAge, SingleStep=SingleStep, TimeLimit=TimeLimit)
            uc = utility_charge(agent, Env, cars=cars,
                                CurrTimeAge=CurrTimeAge, SingleStep=SingleStep, TimeLimit=TimeLimit)

            if ut >= uc and ut > 0:
                p = find_closest_one(Env[1], cur_r, cur_l)
            elif uc > 0:
                p = find_closest_one(Env[0], cur_r, cur_l)
            else:
                # 电量不够去任务也不够去充电，原地待命
                p = None

            actions.append(p if p is not None else (cur_r, cur_l))

    return actions


def calculate_distance(destination_x, destination_y, x, y):
    distance = math.sqrt((destination_x - x) ** 2 + (destination_y - y) ** 2)
    return distance

def updateEnv(Env, Agent_data, task_time, Actions, CurrTimeAge, SingleStep, TimeLimit):
    ROW = Env.shape[1]
    LIN = Env.shape[2]
    # 获取智能体数量
    Agent_Num = Agent_data.shape[0]
    times = []
    move_dists = np.zeros(Agent_Num)
    uav_flight_dist_task = 0.0
    uav_flight_dist_charge = 0.0
    charge_count = 0
    task_responses = []

    # 更新各个智能体位置
    for Agent_Index in range(Agent_Num):
        Agent_Row = Actions[Agent_Index][0]
        Agent_Lin = Actions[Agent_Index][1]
        move_dist = calculate_distance(Agent_data[Agent_Index, 2] - 1, Agent_data[Agent_Index, 3] - 1, Agent_Row, Agent_Lin)
        move_dists[Agent_Index] = move_dist
        time = move_dist / Agent_data[Agent_Index, 6]
        times.append(time)
        # Track UAV flight distance (agent type 3) - distinguish task vs charge
        if Agent_data[Agent_Index, 5] == 3:
            # Check if UAV is going to a charge point (Env[0]) or task point (Env[1])
            if Env[0, Agent_Row, Agent_Lin] == 1:
                uav_flight_dist_charge += move_dist
            elif Env[1, Agent_Row, Agent_Lin] == 1:
                uav_flight_dist_task += move_dist
        Agent_data[Agent_Index, 2] = Agent_Row + 1
        Agent_data[Agent_Index, 3] = Agent_Lin + 1

    for i in range(ROW):
        for ii in range(LIN):
            # 计算目标区域各种智能体的数量
            user_Num = 0
            car_Num = 0
            uav_Num = 0
            workers = []
            cars = []
            uavs = []
            for Agent_Index in range(Agent_Num):
                # 统计该位置参与者数量
                if Agent_data[Agent_Index, 2] - 1 == i and Agent_data[Agent_Index, 3] - 1 == ii and Agent_data[Agent_Index, 5] == 1:
                    user_Num = user_Num + 1
                    workers.append(Agent_Index)
                # 统计该位置补给车数量
                if Agent_data[Agent_Index, 2] - 1 == i and Agent_data[Agent_Index, 3] - 1 == ii and Agent_data[Agent_Index, 5] == 2:
                    car_Num = car_Num + 1
                    cars.append(Agent_Index)
                # 统计该位置能工作的无人机数量
                if Agent_data[Agent_Index, 2] - 1 == i and Agent_data[Agent_Index, 3] - 1 == ii and Agent_data[Agent_Index, 5] == 3:
                    uav_Num = uav_Num + 1
                    uavs.append(Agent_Index)

            # --- 任务完成 (Task Completion) ---
            if Env[1, i, ii] == 1 and workers and uavs:
                for worker_Index in workers:
                    P = False
                    for uav_Index in uavs:
                        T_cost = task_time[i, ii]
                        t_uav_move = times[uav_Index]
                        t_worker_move = times[worker_Index]
                        # UAV开始时刻（考虑等待之前任务完成）
                        u_start = max((CurrTimeAge - 1) * SingleStep, Agent_data[uav_Index, 1])
                        w_start = max((CurrTimeAge - 1) * SingleStep, Agent_data[worker_Index, 1])
                        # 完成时刻
                        completion_time = max(u_start + t_uav_move + T_cost, w_start + t_worker_move + T_cost)

                        # 电量扣除：飞到任务点的距离
                        uav_move_dist = math.ceil(move_dists[uav_Index])
                        # 检查电量是否足够（飞到任务点+最近充电点保留）
                        reserve_dist = min_euclidean_distance_to_one(Env[0], i, ii)
                        u_consum = math.ceil(move_dists[uav_Index] + reserve_dist)

                        if u_consum <= Agent_data[uav_Index, 7] and \
                           completion_time <= min(Agent_data[uav_Index, 4], TimeLimit) and \
                           completion_time <= min(Agent_data[worker_Index, 4], TimeLimit):
                            # 任务完成
                            Env[1, i, ii] = 0
                            P = True
                            # UAV电量扣除
                            Agent_data[uav_Index, 7] -= uav_move_dist
                            # 更新智能体起始时刻
                            Agent_data[uav_Index, 1] = completion_time
                            Agent_data[worker_Index, 1] = completion_time
                            # Track task response time
                            T_assign = (CurrTimeAge - 1) * SingleStep
                            task_responses.append(completion_time - T_assign)
                            break
                    if P:
                        break

            # --- 续航完成 (Charge Completion) ---
            if Env[0, i, ii] == 1 and cars and uavs:
                AllPower = 0
                for uav_Index in uavs:
                    AllPower = AllPower + Agent_data[uav_Index, 8] - Agent_data[uav_Index, 7]
                for car_Index in cars:
                    P = False
                    for uav_Index in uavs:
                        # 电量检查：飞到充电点所需电量
                        u_consum = math.ceil(move_dists[uav_Index])
                        t_car_move = times[car_Index]
                        t_uav_move = times[uav_Index]
                        # 开始时刻（考虑等待）
                        c_start = max((CurrTimeAge - 1) * SingleStep, Agent_data[car_Index, 1])
                        u_start = max((CurrTimeAge - 1) * SingleStep, Agent_data[uav_Index, 1])
                        # 充电时间
                        charge_time = AllPower / Agent_data[car_Index, 8]
                        # 完成时刻
                        completion_time_car = c_start + t_car_move + AllPower / Agent_data[car_Index, 8]
                        completion_time_uav = u_start + t_uav_move + charge_time
                        completion_time = max(completion_time_car, completion_time_uav)

                        if u_consum <= Agent_data[uav_Index, 7] and \
                           completion_time <= min(Agent_data[car_Index, 4], TimeLimit):
                            P = True
                            break
                    if P:
                        for uav_Index in uavs:
                            # UAV电量扣除飞到充电点的距离（充电前扣除）
                            uav_move_dist = math.ceil(move_dists[uav_Index])
                            Agent_data[uav_Index, 7] -= uav_move_dist
                            # 充满电
                            Agent_data[uav_Index, 7] = Agent_data[uav_Index, 8]
                            # 更新起始时刻
                            u_start_i = max((CurrTimeAge - 1) * SingleStep, Agent_data[uav_Index, 1])
                            t_uav_move_i = times[uav_Index]
                            completion_time_uav_i = u_start_i + t_uav_move_i + charge_time
                            Agent_data[uav_Index, 1] = completion_time_uav_i
                        # 更新car起始时刻
                        Agent_data[car_Index, 1] = completion_time_car
                        charge_count += len(uavs)
                        break
    return Env, Agent_data, uav_flight_dist_task, uav_flight_dist_charge, charge_count, task_responses


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


def find_closest_one(grid, x, y):
    points = np.argwhere(grid == 1)
    if len(points) == 0:
        return None
    dists = np.linalg.norm(points - np.array([x, y]), axis=1)
    return tuple(points[np.argmin(dists)])
