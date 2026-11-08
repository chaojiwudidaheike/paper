#该文件用于测试
import numpy as np
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大

from Tool import s_obs_avail_action
from Tool import Update_Agent_Loc
from Tool import select_optimal_region


def Test_Func(Data, qmix_algo, TimeLimit, SingleStep, Center_Scope, AllTime):
    # 环境信息包含两种信息：续航点位置点标识，任务点位置标识，size：2*Row*Lin
    Env = Data['Env']
    # 智能体数据：日期、起始时刻、【起始位置Lin、起始位置Row】、终止时刻、智能体类型、移动速度、起始时刻续航、满电续航
    Agent_Data = Data['Order_Finall']
    Task_Time = Data['Task_Time']
    #统计完成的任务
    ComplateTask = 0

    position_row = 0
    position_lin = 0

    for Curr_Task_Rounds in range(1):
        # 取出训练数据
        Env_ = np.copy(Env)
        Agent_Data_ = np.copy(Agent_Data)
        Task_Time_ = np.copy(Task_Time)
        #全局时间信息
        CurrTime = 0
        AllTask = sum(sum(Env_[1, :, :]))

        # 将环境信息转换成当前全局状态和局部观察以及可执行行为
        qmix_s, obs, avail_action = s_obs_avail_action(Env_, Agent_Data_, position_row, position_lin, Center_Scope, CurrTime, SingleStep, AllTime)

        while CurrTime < TimeLimit:
            #更新全局时间信息
            CurrTime = CurrTime + 1
            # 选择调度中心位置
            position_row, position_lin = select_optimal_region(Env_, Agent_Data_, Center_Scope, CurrTime, SingleStep)

            action = qmix_algo.max_action(qmix_s, obs, avail_action)
            Env_, Agent_Data_, r = Update_Agent_Loc(Env_, Agent_Data_, Task_Time_, action, CurrTime, SingleStep)

            qmix_s_next, obs_next, avail_action_next = s_obs_avail_action(Env_, Agent_Data_, position_row, position_lin, Center_Scope, CurrTime, SingleStep, AllTime)

            qmix_s = np.copy(qmix_s_next)
            obs = np.copy(obs_next)
            avail_action = np.copy(avail_action_next)

            if CurrTime == TimeLimit:
                ComplateTask = AllTask - sum(sum(Env_[1, :, :]))


        return ComplateTask