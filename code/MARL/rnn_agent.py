import torch.nn as nn


class RNNAgent(nn.Module):

    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(RNNAgent, self).__init__()

        self.input_shape = input_shape  # input_shape：智能体网络输入size，包含局部观察和位置信息等等
        self.rnn_hidden_dim = rnn_hidden_dim  # rnn_hidden_dim：隐藏层节点数
        self.n_actions = n_actions  # n_actions：智能体行为数，也是智能体的输出size

        self.fc2 = nn.Sequential(nn.Linear(self.input_shape, self.rnn_hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.rnn_hidden_dim, self.n_actions))

    def forward(self, inputs):
        # inputs是个二维张量，大小为(episode_num,input_shape)
        q = self.fc2(inputs)
        return q

    def update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
