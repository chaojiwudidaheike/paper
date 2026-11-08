import CNN
import rnn_agent
import qmixer
import torch
import numpy as np
import random
from collections import deque
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class QMix:
    def __init__(self, learning_rate, reward_decay, INITIAL_EPSILON, FINAL_EPSILON, memory_size, batch_size,
                 update_target_network, Agent_Num, ROW, LIN):

        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.FINAL_EPSILON = FINAL_EPSILON
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_target_network = update_target_network
        # self.position_row = position_row
        # self.position_lin = position_lin
        # self.Center_Scope = Center_Scope

        # 智能体数量，包括参与者、补给小车、无人机
        self.Agent_Num = Agent_Num
        # 空间区域为矩形，边的宽和长为row和lin
        self.row = ROW
        self.lin = LIN
        # 全局观察size，全局观察为6通道，充电点、任务点、参与者、补给小车、无人机，调度中心覆盖范围
        self.Global_Obs = 6 * self.row * self.lin
        # 智能体位置标识，智能体ID编号
        self.Local_Obs = self.row * self.lin + 3 + 1
        # 矩形区域内的位置点均可移动
        self.agent_actions = self.row * self.lin
        # print(self.agent_actions)

        # 输入张量的通道数
        self.in_channels = 6
        # 输出张量的通道数
        self.out_channels = 6
        # 卷积核的大小，一般我们会使用5x5、3x3这种左右两个数相同的卷积核
        self.kernel_size = 3
        # 卷积核在图像窗口上每次平移的间隔，即所谓的步长
        self.stride = 1
        # Padding即所谓的图像填充，以padding = 1为例，
        # 若原始图像大小为32x32，那么padding后的图像大小就变成了34x34，而不是33x33
        self.padding = 1
        # 这个参数决定了是否采用空洞卷积，默认为1（不采用）
        self.dilation = 1
        # 池化窗口
        self.Pool_size = 1

        # 实例化卷积网络
        self.CNN_space = CNN.cnn(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                                 self.dilation, self.Pool_size).to(device)
        self.CNN_space_target = CNN.cnn(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                        self.padding, self.dilation, self.Pool_size).to(device)

        # self.Global_features = int(self.row * self.lin * self.out_channels / (self.Pool_size ** 2))
        self.Global_features = int((self.row // self.Pool_size) * (self.lin // self.Pool_size) * self.out_channels)

        self.agent_features = self.Global_features + self.Local_Obs

        # 实例化智能体网络
        self.agents = rnn_agent.RNNAgent(self.agent_features, self.agent_features * 10, self.agent_actions).to(device)
        self.target_agents = rnn_agent.RNNAgent(self.agent_features, self.agent_features*10, self.agent_actions).to(device)

        # 实例化混合网络
        self.qmixer = qmixer.QMixer(self.Agent_Num, self.Global_features, self.Agent_Num*10,
                                    self.Global_features*10).to(device)
        self.target_qmixer = qmixer.QMixer(self.Agent_Num, self.Global_features, self.Agent_Num*10,
                                           self.Global_features*10).to(device)

        # 目标网络赋值
        self.CNN_space_target.update(self.CNN_space)
        self.target_agents.update(self.agents)
        self.target_qmixer.update(self.qmixer)

        # 构建参数列表，其中包含卷积层参数、智能体参数和混合网络参数
        self.params = list(self.CNN_space.parameters())
        self.params += list(self.agents.parameters())
        self.params += list(self.qmixer.parameters())

        self.optimizer = torch.optim.RMSprop(params=self.params, lr=self.learning_rate, alpha=0.99, eps=0.00001)
        # 总的学习次数 计数器
        self.learn_step_counter = 0

        # 初始化记忆 [s, s_next, obs, obs_next, avail_action, avail_action_next, a, r, t, ]
        self.memoryDim = self.Global_Obs * 2 + (self.Local_Obs + self.agent_actions) * self.Agent_Num * 2 + self.Agent_Num + 2
        # 初始化经验池
        self.memory = np.zeros((self.memory_size, self.memoryDim))
        # 记录loss的变化列表
        self.loss_his = []

    def save_model(self, filename_CNN, filename_agent, filename_qmixer):
        torch.save(self.CNN_space.state_dict(), filename_CNN)
        torch.save(self.agents.state_dict(), filename_agent)
        torch.save(self.qmixer.state_dict(), filename_qmixer)


    # 存储经验池的函数
    def store_transition(self, qmix_s, qmix_s_next, obs, obs_next, avail_action, avail_action_next, a, r, t):
        # 如果没有'memory_counter'这个变量，就创建为0
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # 每条经验的存放形式
        transition = np.hstack((qmix_s, qmix_s_next, obs, obs_next, avail_action, avail_action_next, a, r, t))
        # 用新经验取代旧经验，循环列表，装满了就取代最久远的
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter = self.memory_counter + 1
        # print(transition.shape)

    # 输入的是智能体的观测值和可执行行为集合
    def choose_action(self, s, observation, Curr_avail_action):
        action = np.zeros((self.Agent_Num))  # 初始化联合行为矩阵
        # e-greedy算法判断是否随机
        if np.random.uniform() > self.INITIAL_EPSILON:
            # 先计算当前空间信息的卷积结果
            s_batch_Cov = s.reshape(1, 6, self.row, self.lin)
            s_batch_Cov = torch.tensor(s_batch_Cov, dtype=torch.float).to(device)
            Cov_output = self.CNN_space(s_batch_Cov)
            output = Cov_output.reshape(self.Global_features)
            # s = torch.tensor(s, dtype=torch.float).to(device)
            # output = s.reshape(self.Global_features)

            for agent_idx in range(self.Agent_Num):
                # 取出智能体agent_idx的局部观察，以计算当前局部观察所对应的q-value
                Dim_Start = self.Local_Obs * agent_idx
                Dim_End = self.Local_Obs * (agent_idx + 1)
                temp_observation = observation[Dim_Start:Dim_End]
                temp_observation = torch.tensor(temp_observation, dtype=torch.float).to(device)
                temp_observation = torch.cat((output, temp_observation), dim=0)
                q_eval_agent = self.agents(temp_observation)

                # 取出不能选择的区域标识
                Dim_Start = self.agent_actions * agent_idx
                Dim_End = self.agent_actions * (agent_idx + 1)
                temp_avail_action = Curr_avail_action[Dim_Start:Dim_End]

                temp_avail_action = torch.tensor(temp_avail_action, dtype=torch.float).to(device)
                q_eval_agent[temp_avail_action == 0.0] = -9999999
                action[agent_idx] = torch.argmax(q_eval_agent)  # 返回Q值最大的动作索引值，是从0开始到n_action-1！
        else:
            for agent_idx in range(self.Agent_Num):
                # 取出不能选择的区域标识
                Dim_Start = self.agent_actions * agent_idx
                Dim_End = self.agent_actions * (agent_idx + 1)
                temp_avail_action = Curr_avail_action[Dim_Start:Dim_End]
                # 需要将tuple转换成ndarray类型
                Curr_avail_action_idx = np.nonzero(temp_avail_action)[0]
                action[agent_idx] = np.random.choice(Curr_avail_action_idx)

        return action

    # 输入的是智能体的观测值和可执行行为集合
    def max_action(self, s, observation, Curr_avail_action):
        # 先计算当前空间信息的卷积结果
        s_batch_Cov = s.reshape(1, 6, self.row, self.lin)
        s_batch_Cov = torch.tensor(s_batch_Cov, dtype=torch.float).to(device)
        Cov_output = self.CNN_space(s_batch_Cov)
        output = Cov_output.reshape(self.Global_features)
        # s = torch.tensor(s, dtype=torch.float).to(device)
        # output = s.reshape(self.Global_features)

        action = np.zeros((self.Agent_Num))  # 初始化联合行为矩阵

        for agent_idx in range(self.Agent_Num):
            # 取出智能体agent_idx的局部观察，以计算当前局部观察所对应的q-value
            Dim_Start = self.Local_Obs * agent_idx
            Dim_End = self.Local_Obs * (agent_idx + 1)
            temp_observation = observation[Dim_Start:Dim_End]
            temp_observation = torch.tensor(temp_observation, dtype=torch.float).to(device)
            temp_observation = torch.cat((output, temp_observation), dim=0)

            # 取出不能选择的区域标识
            Dim_Start = self.agent_actions * agent_idx
            Dim_End = self.agent_actions * (agent_idx + 1)
            temp_avail_action = Curr_avail_action[Dim_Start:Dim_End]

            # 选出最优区域
            q_eval_agent = self.agents(temp_observation)
            temp_avail_action = torch.tensor(temp_avail_action, dtype=torch.float).to(device)
            q_eval_agent[temp_avail_action == 0.0] = -9999999
            action[agent_idx] = torch.argmax(q_eval_agent)  # 返回Q值最大的动作索引值，是从0开始到n_action-1！

        return action

    def learn(self):

        sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        # print("qmix", sample_index)
        batch_memory = self.memory[sample_index, :]  # 拿出索引训练数据

        # 取出经验池中的数据
        # 取出当前全局观察
        Dim_Start = 0
        Dim_End = self.Global_Obs
        s_batch = batch_memory[:, Dim_Start:Dim_End]
        # 取出下一时刻全局观察
        Dim_Start = Dim_End
        Dim_End = self.Global_Obs * 2
        s_next_batch = batch_memory[:, Dim_Start:Dim_End]
        # 取出当前智能体局部观察
        Dim_Start = Dim_End
        Dim_End = self.Global_Obs * 2 + self.Local_Obs * self.Agent_Num
        obs_batch = batch_memory[:, Dim_Start:Dim_End]
        # 取出下一时刻智能体局部观察
        Dim_Start = Dim_End
        Dim_End = self.Global_Obs * 2 + self.Local_Obs * self.Agent_Num * 2
        obs_next_batch = batch_memory[:, Dim_Start:Dim_End]
        # 取出当前时刻智能体可选行为集合
        Dim_Start = Dim_End
        Dim_End = self.Global_Obs * 2 + self.Local_Obs * self.Agent_Num * 2 + self.agent_actions * self.Agent_Num
        avail_action_batch = batch_memory[:, Dim_Start:Dim_End]
        # 取出下一时刻智能体可选行为集合
        Dim_Start = Dim_End
        Dim_End = self.Global_Obs * 2 + self.Local_Obs * self.Agent_Num * 2 + self.agent_actions * self.Agent_Num * 2
        avail_action_next_batch = batch_memory[:, Dim_Start:Dim_End]
        # 取出智能体行为集合
        Dim_Start = Dim_End
        Dim_End = self.Global_Obs * 2 + (self.Local_Obs + self.agent_actions) * self.Agent_Num * 2 + self.Agent_Num
        a_batch = batch_memory[:, Dim_Start:Dim_End]
        # 取出奖励值
        Dim = Dim_End
        r_batch = batch_memory[:, Dim]
        # 取出终止标识
        Dim = Dim + 1
        t_batch = batch_memory[:, Dim]

        # # 先计算当前空间信息的卷积结果
        s_batch_Cov = s_batch.reshape(self.batch_size, 6, self.row, self.lin)
        s_batch_Cov = torch.tensor(s_batch_Cov, dtype=torch.float).to(device)
        Cov_output = self.CNN_space(s_batch_Cov)
        output = Cov_output.reshape(self.batch_size, self.Global_features)

        # s_batch_tensor = torch.tensor(s_batch, dtype=torch.float).to(device)
        # output = s_batch_tensor.reshape(self.batch_size, self.Global_features)

        # 计算当前时刻e-greedy所选行为的q-value
        CurrValue = torch.zeros((self.batch_size, self.Agent_Num)).to(device)  # 初始化联合行为矩阵
        for agent_idx in range(0, self.Agent_Num):
            # 取出智能体agent_idx的局部观察
            Dim_Start = self.Local_Obs * agent_idx
            Dim_End = self.Local_Obs * (agent_idx + 1)
            obs_batch_temp = obs_batch[:, Dim_Start:Dim_End]
            obs_batch_temp = torch.tensor(obs_batch_temp, dtype=torch.float).to(device)
            obs_batch_temp = torch.cat((output, obs_batch_temp), dim=1)
            q_curr_agent = self.agents(obs_batch_temp)

            # 取出对应数据的q-value
            for temp_Index in range(self.batch_size):
                CurrValue[temp_Index, agent_idx] = q_curr_agent[temp_Index, a_batch[temp_Index, agent_idx].astype(int)]

        # 计算下一时刻空间信息的卷积结果
        s_next_batch_Cov = s_next_batch.reshape(self.batch_size, 6, self.row, self.lin)  # 变换成需要的格式
        s_next_batch_Cov = torch.tensor(s_next_batch_Cov, dtype=torch.float).to(device)
        Cov_output_next = self.CNN_space_target(s_next_batch_Cov)  # 卷积计算
        output_next = Cov_output_next.reshape(self.batch_size, self.Global_features)  # 恢复出向量
        # s_next_batch_tensor = torch.tensor(s_next_batch, dtype=torch.float).to(device)
        # output_next = s_next_batch_tensor.reshape(self.batch_size, self.Global_features)

        # 计算下一时刻最优q-value
        NextValue = torch.zeros((self.batch_size, self.Agent_Num)).to(device)  # 初始化联合行为矩阵
        avail_action_next_batch = torch.tensor(avail_action_next_batch, dtype=torch.float).to(device)
        for agent_idx in range(0, self.Agent_Num):
            # 取出智能体agent_idx的局部观察
            Dim_Start = self.Local_Obs * agent_idx
            Dim_End = self.Local_Obs * (agent_idx + 1)
            obs_next_batch_temp = obs_next_batch[:, Dim_Start:Dim_End]
            obs_next_batch_temp = torch.tensor(obs_next_batch_temp, dtype=torch.float).to(device)
            obs_next_batch_temp = torch.cat((output_next, obs_next_batch_temp), dim=1)

            q_next_agent = self.target_agents(obs_next_batch_temp)
            Dim_Start = self.agent_actions * agent_idx
            Dim_End = self.agent_actions * (agent_idx + 1)
            q_next_agent[avail_action_next_batch[:, Dim_Start:Dim_End] == 0.0] = -9999999
            selected_q_next = torch.max(q_next_agent, axis=1)[0]
            NextValue[:, agent_idx] = selected_q_next

        # 计算当前时刻所选行为的Q-Value-tot
        s_batch_temp = output
        chosen_action_qvals = self.qmixer(s_batch_temp, CurrValue)
        # 计算下一时刻最优行为的Q-Value-tot
        s_next_batch_temp = output_next
        target_max_qvals = self.target_qmixer(s_next_batch_temp, NextValue)

        # 计算目标值
        r_batch_temp = torch.tensor(r_batch, dtype=torch.float).to(device)
        t_batch_temp = torch.tensor(t_batch, dtype=torch.float).to(device)
        yi = r_batch_temp + self.reward_decay * (1 - t_batch_temp) * target_max_qvals

        # 计算均方差loss
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(chosen_action_qvals, yi)
        # print("#######qmix-loss")
        # print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimizer.step()

        self.learn_step_counter = self.learn_step_counter + 1

        if self.learn_step_counter % self.update_target_network == 0:
            # 更新目标模型参数
            self.CNN_space_target.update(self.CNN_space)
            self.target_agents.update(self.agents)
            self.target_qmixer.update(self.qmixer)
            # 输出评估模型训练损失
            # print(loss)

        # 可变的探测概率
        if self.learn_step_counter % self.batch_size == 0:
            self.INITIAL_EPSILON = self.INITIAL_EPSILON - (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.batch_size
        return loss
