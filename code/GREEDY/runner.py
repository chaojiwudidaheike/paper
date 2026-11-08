from Tools import action
from Tools import updateEnv
from Tools import select_optimal_region
from scipy.io import loadmat
import numpy as np
import time

if __name__ == '__main__':

    Data = loadmat('../../data/Mat/Random_1.mat', mat_dtype=True)
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
    Center_Scope = 8
    TimeLimit = 120
    SingleStep = 10
    CurrTimeAge = 0
    start_time = time.time()

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

        while CurrTimeAge < int(TimeLimit/SingleStep):
            CurrTimeAge += 1

            centerRow, centerLin = select_optimal_region(Env_, Agent_Data_, Center_Scope, CurrTimeAge, SingleStep)
            print("Center Row:", centerRow, "Center Lin:", centerLin)

            Actions = action(Env_, Agent_Data_, Task_Time_, centerRow, centerLin, Center_Scope, CurrTimeAge, SingleStep, TimeLimit)
            print("Actions:", Actions)
            Env_, Agent_Data_, uav_flight_dist_task, uav_flight_dist_charge, charge_count, task_responses = updateEnv(Env_, Agent_Data_, Task_Time_, Actions, CurrTimeAge, SingleStep, TimeLimit)

            total_uav_flight_dist_task += uav_flight_dist_task
            total_uav_flight_dist_charge += uav_flight_dist_charge
            total_charge_count += charge_count
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
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"运行时间: {elapsed_time} 秒")
