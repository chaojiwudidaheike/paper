import numpy as np
import math
import random
np.set_printoptions(threshold=np.inf)  # threshold 指定超过多少使用省略号，np.inf代表无限大

def s_obs_avail_action(Env, Agent_Data, position_row, position_lin, Center_Scope, CurrTimeAge, SingleStep, AllTime):

    ROW = Env.shape[1]
    LIN = Env.shape[2]
    # 获取智能体数量
    Agent_Num = Agent_Data.shape[0]
    s = np.zeros((6, ROW, LIN))
    # 赋值充电点和任务点
    s[0:2, :, :] = Env

    # 汇总智能体位置
    for i in range(Agent_Num):
        if Agent_Data[i, 5] == 1 and Agent_Data[i, 1] <= CurrTimeAge*SingleStep < Agent_Data[i, 4]:  # 参与者与补给车位置汇总
            s[2, int(Agent_Data[i, 2] - 1), int(Agent_Data[i, 3] - 1)] = s[2, int(Agent_Data[i, 2] - 1), int(Agent_Data[i, 3] - 1)] + 1
        if Agent_Data[i, 5] == 2 and Agent_Data[i, 1] <= CurrTimeAge*SingleStep < Agent_Data[i, 4]:  # 补给车位置汇总
            s[3, int(Agent_Data[i, 2] - 1), int(Agent_Data[i, 3] - 1)] = s[3, int(Agent_Data[i, 2] - 1), int(Agent_Data[i, 3] - 1)] + 1
        if Agent_Data[i, 5] == 3 and Agent_Data[i, 1] <= CurrTimeAge*SingleStep < Agent_Data[i, 4]:  # 记录无人机充电需求
            s[4, int(Agent_Data[i, 2] - 1), int(Agent_Data[i, 3] - 1)] = s[4, int(Agent_Data[i, 2] - 1), int(Agent_Data[i, 3] - 1)] + (0.37 ** int(Agent_Data[i, 7] / (Agent_Data[i, 6] * SingleStep)))

    matrix = [[0 for _ in range(LIN)] for _ in range(ROW)]

    # 计算范围边界
    start_row = position_row
    end_row = min(ROW, position_row + Center_Scope)
    start_lin = position_lin
    end_lin = min(LIN, position_lin + Center_Scope)

    # 将指定范围的元素标记为 1
    for i in range(start_row, end_row):
        for j in range(start_lin, end_lin):
            matrix[i][j] = 1

    s[5, :, :] = matrix
    # 将全局矩阵变成横向量
    s = s.flatten()


    # 智能体可执行行为
    avail_action = []
    # 局部观察obs应该包含全局观察、当前智能体所在位置、当前智能体编号、当前智能体类别
    obs = []
    for i in range(Agent_Num):
        # 智能体可选择行为范围计算过程
        temp_avail_action = np.zeros((ROW, LIN))
        temp_avail_action[int(Agent_Data[i, 2]-1), int(Agent_Data[i, 3]-1)] = 1
        if Agent_Data[i, 1] <= CurrTimeAge*SingleStep < Agent_Data[i, 4]:
            if Agent_Data[i, 5] == 1 and position_row <= Agent_Data[i, 2]-1 < position_row + Center_Scope and position_lin <= Agent_Data[i, 3]-1 < position_lin + Center_Scope:
                for ii in range(ROW):
                    for i_3 in range(LIN):
                        if Env[1, ii, i_3] == 1 and calculate_distance(ii, i_3, Agent_Data[i, 2]-1, Agent_Data[i, 3]-1) / Agent_Data[i, 6] + Agent_Data[i, 1] <= min(Agent_Data[i, 4], (CurrTimeAge+1)*SingleStep):
                            temp_avail_action[ii, i_3] = 1
            if Agent_Data[i, 5] == 2 and position_row <= Agent_Data[i, 2]-1 < position_row + Center_Scope and position_lin <=Agent_Data[i, 3]-1 < position_lin + Center_Scope:
                for ii in range(ROW):
                    for i_3 in range(LIN):
                        if Env[0, ii, i_3] == 1 and calculate_distance(ii, i_3, Agent_Data[i, 2]-1, Agent_Data[i, 3]-1) / Agent_Data[i, 6] + Agent_Data[i, 1] <= min(Agent_Data[i, 4], (CurrTimeAge+1)*SingleStep):
                            temp_avail_action[ii, i_3] = 1
            if Agent_Data[i, 5] == 3 and position_row <= Agent_Data[i, 2]-1 < position_row + Center_Scope and position_lin <=Agent_Data[i, 3]-1 < position_lin + Center_Scope:
                for ii in range(ROW):
                    for i_3 in range(LIN):
                        if (Env[0, ii, i_3] == 1 or Env[1, ii, i_3] == 1) and calculate_distance(ii, i_3, Agent_Data[i, 2]-1, Agent_Data[i, 3]-1) / Agent_Data[i, 6] + Agent_Data[i, 1] <= min(Agent_Data[i, 4], (CurrTimeAge+1)*SingleStep) and calculate_distance(ii, i_3, Agent_Data[i, 2]-1, Agent_Data[i, 3]-1) <= Agent_Data[i, 7]:
                            temp_avail_action[ii, i_3] = 1


        temp_avail_action = temp_avail_action.flatten()
        avail_action = np.concatenate((avail_action, temp_avail_action), 0)
        # 将该智能体位置信息加入局部观察，用于标识该智能体
        Agent_loc = np.zeros((ROW, LIN))
        Agent_loc[int(Agent_Data[i, 2]-1), int(Agent_Data[i, 3]-1)] = 1
        Agent_Loc_Temp = Agent_loc.flatten()
        obs = np.concatenate((obs, Agent_Loc_Temp), 0)

        # 记录当前智能体类别
        if Agent_Data[i, 5] == 1:
            obs = np.concatenate((obs, [1, 0, 0]), 0)
            obs = np.concatenate((obs, [0]), 0)
        if Agent_Data[i, 5] == 2:
            obs = np.concatenate((obs, [0, 1, 0]), 0)
            obs = np.concatenate((obs, [0]), 0)
        if Agent_Data[i, 5] == 3:
            obs = np.concatenate((obs, [0, 0, 1]), 0)
            obs = np.concatenate((obs, [0.37 ** int(Agent_Data[i, 7] / (Agent_Data[i, 6] * SingleStep))]), 0)

    return s, obs, avail_action


def Update_Agent_Loc(Env, Agent_Data, Task_Time, action, CurrTimeAge, SingleStep):

    ROW = Env.shape[1]
    LIN = Env.shape[2]
    # 获取智能体数量
    Agent_Num = Agent_Data.shape[0]
    distance = np.zeros(Agent_Num)
    move = np.ones(Agent_Num)
    # 更新各个智能体位置
    for Agent_Index in range(Agent_Num):
        Agent_Row = int(action[Agent_Index] / LIN)
        Agent_Lin = int(action[Agent_Index] % LIN)
        # 计算智能体移动距离
        distance[Agent_Index] = calculate_distance(Agent_Row, Agent_Lin, Agent_Data[Agent_Index, 2] - 1, Agent_Data[Agent_Index, 3] - 1)
        if Agent_Row == Agent_Data[Agent_Index, 2] - 1 and Agent_Lin == Agent_Data[Agent_Index, 3] - 1:
            move[Agent_Index] = 0

        Agent_Data[Agent_Index, 2] = Agent_Row + 1
        Agent_Data[Agent_Index, 3] = Agent_Lin + 1

    # 更新智能体电量
    for i in range(Agent_Num):
        if Agent_Data[i, 5] == 3 and move[i] == 1:
            Agent_Data[i, 7] = Agent_Data[i, 7] - distance[i]

    # 更新环境中的任务信息和奖励
    # 初始化奖励
    Task_r = 0
    Energy_r = 0
    for i in range(ROW):
        for ii in range(LIN):
            # 计算目标区域各种智能体的数量
            user_Num = 0
            car_Num = 0
            uav_Num = 0
            users = []
            cars = []
            uavs = []
            for Agent_Index in range(Agent_Num):
                # 统计该位置参与者数量
                if Agent_Data[Agent_Index, 2]-1 == i and Agent_Data[Agent_Index, 3]-1 == ii and Agent_Data[Agent_Index, 5] == 1 and Agent_Data[Agent_Index, 1] <= CurrTimeAge*SingleStep < Agent_Data[Agent_Index, 4]:
                    user_Num = user_Num + 1
                    users.append(Agent_Index)
                # 统计该位置补给车数量
                if Agent_Data[Agent_Index, 2]-1 == i and Agent_Data[Agent_Index, 3]-1 == ii and Agent_Data[Agent_Index, 5] == 2 and Agent_Data[Agent_Index, 1] <= CurrTimeAge*SingleStep < Agent_Data[Agent_Index, 4]:
                    car_Num = car_Num + 1
                    cars.append(Agent_Index)
                # 统计该位置能工作的无人机数量
                if Agent_Data[Agent_Index, 2]-1 == i and Agent_Data[Agent_Index, 3]-1 == ii and Agent_Data[Agent_Index, 5] == 3 and Agent_Data[Agent_Index, 1] <= CurrTimeAge*SingleStep < Agent_Data[Agent_Index, 4]:
                    uav_Num = uav_Num + 1
                    uavs.append(Agent_Index)
            # 有任务就做，没任务就拉倒
            if Env[1, i, ii] == 1 and user_Num > 0 and uav_Num > 0:
                Env[1, i, ii] = 0
                # 以单时间步长完成的任务数作为奖励
                Task_r = Task_r + 1
                # 随机选择一个用户和一个无人机
                user_Index = random.choice(users)
                UAV_Index = random.choice(uavs)
                for x in users:
                    if x == user_Index:
                        Agent_Data[x, 1] = Agent_Data[x, 1] + distance[x]/Agent_Data[x, 6] + Task_Time[i, ii]
                    else:
                        Agent_Data[x, 1] = Agent_Data[x, 1] + distance[x]/Agent_Data[x, 6]
                for y in uavs:
                    if y == UAV_Index:
                        Agent_Data[y, 1] = Agent_Data[y, 1] + distance[y]/Agent_Data[y, 6] + Task_Time[i, ii]
                    else:
                        Agent_Data[y, 1] = Agent_Data[y, 1] + distance[y]/Agent_Data[y, 6]
            # 查看该地是否有补给小车，有的话就查看是否有无人机更换电池
            if Env[0, i, ii] == 1 and car_Num > 0 and uav_Num > 0:
                car_Index = random.choice(cars)
                charge_power = 0
                for x in uavs:
                    charge_power = charge_power + Agent_Data[x, 8] - Agent_Data[x, 7]
                    Energy_r = Energy_r + 0.3
                    Agent_Data[x, 7] = Agent_Data[x, 8]
                    Agent_Data[x, 1] = Agent_Data[x, 1] + distance[x]/Agent_Data[x, 6] + charge_power / Agent_Data[x, 8]
                for x in cars:
                    if x == car_Index:
                        Agent_Data[x, 1] = Agent_Data[x, 1] + distance[x]/Agent_Data[x, 6] + charge_power / Agent_Data[x, 8]
                    else:
                        Agent_Data[x, 1] = Agent_Data[x, 1] + distance[x]/Agent_Data[x, 6]


    # 计算复合奖励
    r = Task_r + Energy_r

    return Env, Agent_Data, r

def calculate_distance(destination_x, destination_y, x, y):
    distance = math.sqrt((destination_x - x) ** 2 + (destination_y - y) ** 2)
    return distance


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








