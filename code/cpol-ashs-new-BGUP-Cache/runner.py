from Tools import build_graph
from Tools import updateEnv
from Tools import creatWeight
from Tools import select_optimal_region
from scipy.io import loadmat
from algorithm import max_weight_max_independent_set, no_mulPQ_max_weight_max_independent_set
import numpy as np
import time
import random

if __name__ == '__main__':

    Data = loadmat('../../data/Mat/TaxiTrajectories.mat', mat_dtype=True)
    # 环境信息包含两种信息：续航点位置点标识，任务点位置标识，size：2*Row*Lin
    Env = Data['Env']
    # 智能体数据：日期、起始时刻、【起始位置Row、起始位置Lin】、终止时刻、智能体类型、移动速度、起始时刻续航、满电续航（数据集中的位置以1开始不是以0开始）
    Agent_Data = Data['Order_Finall']
    # 完成每个任务所需的代价消耗
    Task_Time = Data['Task_Time']

    ROW = Env.shape[1]
    LIN = Env.shape[2]
    Agent_Num = Agent_Data.shape[0]
    print(ROW, LIN)
    Center_Scope = 20
    TimeLimit = 150
    SingleStep = 10
    CurrTimeAge = 0
    start_time = time.time()

    # ========== 故障参数 ==========
    # 是否开启智能体故障（每个智能体每步以 [0.1, 0.25] 的概率故障）
    ENABLE_AGENT_FAULT = False
    # 是否开启移动决策中心故障（每步以 [0.1, 0.2] 的概率故障，故障时跳过本轮调度）
    ENABLE_MDC_FAULT = False
    # 是否开启指令失效（最大独立集中每个节点以 [0.1, 0.2] 的概率失效，失效则从独立集中删除）
    ENABLE_INSTRUCTION_FAULT = False

    agent_fault_count = 0
    mdc_fault_count = 0
    instruction_fault_count = 0

    for Curr_Task_Rounds in range(1):
        Env_ = np.copy(Env)
        Agent_Data_ = np.copy(Agent_Data)
        Task_Time_ = np.copy(Task_Time)

        result = np.zeros(int(TimeLimit/SingleStep))
        AllTask = sum(sum(Env_[1, :, :]))
        print(AllTask)

        total_uav_flight_dist_task = 0.0
        total_uav_flight_dist_charge = 0.0
        total_charge_count = 0
        all_task_responses = []
        total_idle_uav_steps = 0
        total_schedulable_uav_steps = 0
        total_mdc_distance = 0.0
        prev_centerRow = None
        prev_centerLin = None
        total_worker_dist = 0.0
        total_car_dist = 0.0

        while CurrTimeAge < int(TimeLimit/SingleStep):
            CurrTimeAge += 1

            # --- 移动决策中心故障判定 ---
            if ENABLE_MDC_FAULT:
                mdc_fault_prob = random.uniform(0.1, 0.2)
                if random.random() < mdc_fault_prob:
                    mdc_fault_count += 1
                    print(f"[MDC FAULT] Time step {CurrTimeAge}: MDC故障，跳过本轮调度")
                    result[CurrTimeAge-1] = AllTask - sum(sum(Env_[1, :, :]))
                    print("----------------------------------------------")
                    continue

            # --- 智能体故障判定 ---
            faulty_agents = set()
            if ENABLE_AGENT_FAULT:
                for i in range(Agent_Num):
                    # 仅对在线的智能体判定故障
                    if Agent_Data_[i, 1] <= CurrTimeAge * SingleStep < Agent_Data_[i, 4]:
                        agent_fault_prob = random.uniform(0.1, 0.25)
                        if random.random() < agent_fault_prob:
                            faulty_agents.add(i)
                            agent_fault_count += 1
            if faulty_agents:
                print(f"[AGENT FAULT] Time step {CurrTimeAge}: 故障智能体索引 {sorted(faulty_agents)}")

            centerRow, centerLin = select_optimal_region(Env_, Agent_Data_, Center_Scope, CurrTimeAge, SingleStep)
            # centerRow, centerLin = 0, 0
            print("Center Row:", centerRow, "Center Lin:", centerLin)

            # --- MDC移动距离计算 ---
            if prev_centerRow is not None and prev_centerLin is not None:
                mdc_dist = ((centerRow - prev_centerRow) ** 2 + (centerLin - prev_centerLin) ** 2) ** 0.5
                total_mdc_distance += mdc_dist
            prev_centerRow = centerRow
            prev_centerLin = centerLin

            G = build_graph(Env_, Agent_Data_, Task_Time_, centerRow, centerLin, Center_Scope, CurrTimeAge, SingleStep, TimeLimit, faulty_agents=faulty_agents)
            weight_G = creatWeight(G, Env_, Agent_Data_, Task_Time_, CurrTimeAge, SingleStep)

            # 多优先级队列求解
            max_independent_set, weight = max_weight_max_independent_set(weight_G)
            # 没有多优先级队列求解
            # max_independent_set, weight = no_mulPQ_max_weight_max_independent_set(weight_G)
            in_set = sorted(max_independent_set, key=lambda x: (G.nodes[x]['weight'], x), reverse=True)
            print("Maximum Independent Set:", in_set, "Weight of Maximum Independent Set:", weight)

            # --- 统计本步可调度UAV和未执行任务/充电的UAV ---
            import re
            schedulable_uavs = set()
            for i in range(Agent_Num):
                if Agent_Data_[i, 5] == 3 and Agent_Data_[i, 1] <= CurrTimeAge * SingleStep < Agent_Data_[i, 4]:
                    schedulable_uavs.add(i)
            total_schedulable_uav_steps += len(schedulable_uavs)

            busy_uavs = set()
            for node in in_set:
                node_numbers = []
                for item in node:
                    numbers = list(map(int, re.findall(r'\d+', item)))
                    node_numbers.append(numbers)
                uav = node_numbers[2]
                task = node_numbers[3]
                charging = node_numbers[4]
                if uav and (task or charging):
                    busy_uavs.add(uav[0])
            idle_uavs = schedulable_uavs - busy_uavs
            total_idle_uav_steps += len(idle_uavs)

            # --- 指令失效判定 ---
            if ENABLE_INSTRUCTION_FAULT:
                fault_nodes = []
                for node in in_set[:]:
                    fault_prob = random.uniform(0.1, 0.2)
                    if random.random() < fault_prob:
                        fault_nodes.append(node)
                        in_set.remove(node)
                        instruction_fault_count += 1
                if fault_nodes:
                    print(f"[INSTRUCTION FAULT] Time step {CurrTimeAge}: 失效指令 {fault_nodes}")
                print("After Instruction Fault:", in_set)

            # for node in in_set:
            #     print(G.nodes[node].get('weight', 0))

            Env_, Agent_Data_, Task_Time_, uav_flight_dist_task, uav_flight_dist_charge, charge_count, task_responses, worker_dist, car_dist = updateEnv(Env_, Agent_Data_, Task_Time_, in_set, CurrTimeAge, SingleStep)

            total_uav_flight_dist_task += uav_flight_dist_task
            total_uav_flight_dist_charge += uav_flight_dist_charge
            total_charge_count += charge_count
            total_worker_dist += worker_dist
            total_car_dist += car_dist
            all_task_responses.extend(task_responses)

            result[CurrTimeAge-1] = AllTask - sum(sum(Env_[1, :, :]))
            print("----------------------------------------------")

        N_completed = int(result[-1])
        N_total = int(AllTask)

        # Energy Efficiency: EE = N_completed / E_total (E_total = UAV flight distance for tasks only, excluding charging)
        EE = N_completed / total_uav_flight_dist_task if total_uav_flight_dist_task > 0 else 0
        # Average Task Response Time: ATRT = sum(T_complete - T_assign) / N_completed
        ATRT = sum(all_task_responses) / N_completed if N_completed > 0 else 0
        # Average UAV flight distance per completed task
        Avg_Flight_Per_Task = total_uav_flight_dist_task / N_completed if N_completed > 0 else 0

        print(result)
        print(f"Completed Tasks: {N_completed}/{N_total}")
        print(f"Energy Efficiency (EE): {EE:.6f} tasks/unit_distance")
        print(f"Average Task Response Time (ATRT): {ATRT:.2f} time_units")
        print(f"Total Charge Count: {total_charge_count}")
        print(f"Average UAV Flight Distance Per Task: {Avg_Flight_Per_Task:.2f} unit_distance")
        print(f"MDC Total Movement Distance: {total_mdc_distance:.2f} unit_distance")
        print(f"UAV Total Flight Distance (task): {total_uav_flight_dist_task:.2f} unit_distance")
        print(f"UAV Total Flight Distance (charge): {total_uav_flight_dist_charge:.2f} unit_distance")
        print(f"Worker Total Movement Distance: {total_worker_dist:.2f} unit_distance")
        print(f"Car Total Movement Distance: {total_car_dist:.2f} unit_distance")
        mdc_uav_ratio = total_mdc_distance / total_uav_flight_dist_task if total_uav_flight_dist_task > 0 else 0
        print(f"MDC Distance / UAV Task Flight Distance: {mdc_uav_ratio:.2%}")
        # 每步未执行任务/充电的UAV占总可调度UAV的比例
        idle_uav_ratio = total_idle_uav_steps / total_schedulable_uav_steps if total_schedulable_uav_steps > 0 else 0
        print(f"Idle UAV Steps: {total_idle_uav_steps}/{total_schedulable_uav_steps} ({idle_uav_ratio:.2%})")

        print(f"Agent故障次数: {agent_fault_count}, MDC故障次数: {mdc_fault_count}, 指令失效次数: {instruction_fault_count}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"运行时间: {elapsed_time} 秒")
