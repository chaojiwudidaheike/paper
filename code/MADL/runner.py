import numpy as np
import torch
import qmix
from scipy.io import loadmat
from Tool import s_obs_avail_action
from Tool import Update_Agent_Loc
from Tool import select_optimal_region
from test import Test_Func
import random


if __name__ == '__main__':

    Data = loadmat('../../data/Mat/Random_1.mat', mat_dtype=True)

    # 环境信息包含两种信息：续航点位置点标识，任务点位置标识，size：2*Row*Lin
    Env = Data['Env']
    # 智能体数据：日期、起始时刻、【起始位置Lin、起始位置Row】、终止时刻、智能体类型、移动速度、起始时刻续航、满电续航
    Agent_Data = Data['Order_Finall']
    # 完成每个任务所需的代价消耗
    Task_Time = Data['Task_Time']

    ##########*********至此数据假设部分完成**************############

    # 获取智能体数量
    Agent_Num = Agent_Data.shape[0]

    ROW = Env.shape[1]
    LIN = Env.shape[2]
    TrainNum = 50000
    Center_Scope = 20
    AllTime = 120
    SingleStep = 10
    TimeLimit = int(AllTime/SingleStep)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #模型存放路径
    filename_CNN = './model/CNN/CNN'
    filename_agent = './model/agent/agent'
    filename_qmixer = './model/qmixer/qmixer'


    #对象化强化学习算法
    qmix_algo = qmix.QMix(0.0001, 0, 0.1, 0.1, 5000, 64, 200,
                          Agent_Num, ROW, LIN)


    # print("PyTorch版本:", torch.__version__)
    # print("CUDA是否可用:", torch.cuda.is_available())
    # print("CUDA版本:", torch.version.cuda if torch.cuda.is_available() else "N/A")
    #
    # if torch.cuda.is_available():
    #     print("GPU数量:", torch.cuda.device_count())
    #     for i in range(torch.cuda.device_count()):
    #         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # else:
    #     print("没有检测到可用的GPU")

    for episode in range(TrainNum):
        # print(episode)
        for Curr_Task_Rounds in range(1):
            # 取出训练数据
            Env_ = np.copy(Env)
            Agent_Data_ = np.copy(Agent_Data)
            Task_Time_ = np.copy(Task_Time)
            # 设置是否到达终止目标的标识
            t = 0
            position_row = 0
            position_lin = 0
            CurrTime = 0
            loss = 0
            # 全局时间信息
            qmix_s, obs, avail_action = s_obs_avail_action(Env_, Agent_Data_, position_row, position_lin, Center_Scope, CurrTime, SingleStep, AllTime)
            while CurrTime < TimeLimit:
                # 更新全局时间信息

                CurrTime = CurrTime + 1
                # 选择调度中心位置
                position_row, position_lin = select_optimal_region(Env_, Agent_Data_, Center_Scope, CurrTime, SingleStep)

                # 得到action，行为可变的，在不同状态下，行为的选择范围不一样
                action = qmix_algo.choose_action(qmix_s, obs, avail_action)
                # 更新环境任务信息、智能体位置信息、无人机剩余电量，以及计算奖励
                Env_, Agent_Data_, r = Update_Agent_Loc(Env_, Agent_Data_, Task_Time_, action, CurrTime, SingleStep)
                # 更新下一全局状态和局部观察信息以及可执行行为
                qmix_s_next, obs_next, avail_action_next = s_obs_avail_action(Env_, Agent_Data_, position_row, position_lin, Center_Scope, CurrTime, SingleStep, AllTime)
                # 判断是否达到终止目标

                if CurrTime == TimeLimit:
                    t = 1
                # 存入经验池
                qmix_algo.store_transition(qmix_s, qmix_s_next, obs, obs_next, avail_action, avail_action_next, action, r, t)
                # 经验池中的数据量达到要求就要开始训练网络模型了
                if qmix_algo.memory_counter >= qmix_algo.memory_size:
                    loss = qmix_algo.learn()

                # 更新全局观察
                qmix_s = np.copy(qmix_s_next)
                obs = np.copy(obs_next)
                avail_action = np.copy(avail_action_next)


        if episode % 10 == 0:
            print(episode)
            qmix_algo.save_model(filename_CNN, filename_agent, filename_qmixer)
            #测试
            result = np.zeros((TimeLimit))  # 初始化联合行为矩阵
            for TimeLimit_ in range(TimeLimit):
                ComplateTask = Test_Func(Data, qmix_algo, TimeLimit_ + 1, SingleStep, Center_Scope, AllTime)
                result[TimeLimit_] = ComplateTask
            print(result)







