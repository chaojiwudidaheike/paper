import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, n_agents, state_shape, qmix_hidden, hyper_hidden_dim):
        super(QMixer, self).__init__()

        self.n_agents = n_agents  # 智能体数目
        self.state_shape = state_shape  # 全局观察的size
        self.qmix_hidden = qmix_hidden  # Mix函数的隐藏层节点数
        self.hyper_hidden_dim = hyper_hidden_dim  # 参数估计网络的隐藏层节点数

        # 因为生成的 hyper_w1 需要是一个矩阵，而 pytorch 神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # hyper_w1 网络用于输出推理网络中的第一层神经元所需的 weights，
        # 推理网络第一层矩阵为(n_agents,qmix_hidden)，因此 hyper_w1 网络输出维度为 n_agents*qmix_hidden
        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_shape, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim, self.n_agents * self.qmix_hidden))

        # hyper_w2 网络用于输出推理网络中的第二层神经元所需的 weights，
        # 推理网络第二层矩阵为(qmix_hidden,1)，因此 hyper_w2 网络输出维度为 qmix_hidden*1
        self.hyper_w2 = nn.Sequential(nn.Linear(self.state_shape, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim, self.qmix_hidden * 1))

        # hyper_b1 生成第一层网络对应维度的偏差 bias，
        # 维度为(1,qmix_hidden)，因此，hyper_b1网络的输出维度为 1*qmix_hidden
        self.hyper_b1 = nn.Sequential(nn.Linear(self.state_shape, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim, self.qmix_hidden)
                                      )
        # hyper_b2 生成对应从隐层到输出 Q 值层的 bias
        # 维度为(1,1)，因此，hyper_b2网络的输出维度为 1*1
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_shape, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim, 1)
                                      )

    def forward(self, states, q_values):
        q_values = q_values.view(-1, 1, self.n_agents)

        # 获取Mix网络的第一层权重和偏置
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        # 将权重和偏置变换成需要的矩阵模式
        w1 = w1.view(-1, self.n_agents, self.qmix_hidden)
        b1 = b1.view(-1, 1, self.qmix_hidden)

        # 计算隐藏层输出
        hidden = F.relu(torch.bmm(q_values, w1) + b1)  # torch.bmm(a, b) 计算矩阵 a 和矩阵 b 相乘

        # 获取Mix网络的第二层权重和偏置
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        # 将权重和偏置变换成需要的矩阵模式
        w2 = w2.view(-1, self.qmix_hidden, 1)
        b2 = b2.view(-1, 1, 1)

        # 计算最终Mix网络的输出，并转换成相应的格式
        q_total = torch.bmm(hidden, w2) + b2
        q_total = torch.squeeze(q_total)
        return q_total

    def update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)