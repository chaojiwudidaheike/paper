import torch.nn as nn


# 建立pytorch神经网络
class cnn(nn.Module):
    def __init__(self, in_channels_, out_channels_, kernel_size_, stride_, padding_, dilation_, Pool_size_):
        super(cnn, self).__init__()
        # 输入张量的通道数
        self.in_channels_ = in_channels_
        # 输出张量的通道数
        self.out_channels_ = out_channels_
        # 卷积核的大小，一般我们会使用5x5、3x3这种左右两个数相同的卷积核
        self.kernel_size_ = kernel_size_
        # 卷积核在图像窗口上每次平移的间隔，即所谓的步长
        self.stride_ = stride_
        # Padding即所谓的图像填充，以padding = 1为例，
        # 若原始图像大小为32x32，那么padding后的图像大小就变成了34x34，而不是33x33
        self.padding_ = padding_
        # 这个参数决定了是否采用空洞卷积，默认为1（不采用）
        self.dilation_ = dilation_
        # 池化尺寸
        self.Pool_size_ = Pool_size_
        # ----------------------------#
        #   第一部分卷积
        # ----------------------------#
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels_,
                out_channels=self.out_channels_,
                kernel_size=self.kernel_size_,
                stride=self.stride_,
                padding=self.padding_,
                dilation=self.dilation_
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size = self.Pool_size_)
            nn.AvgPool2d(kernel_size=self.Pool_size_)
        )

    # ----------------------------#

    #   前向传播
    # ----------------------------#
    def forward(self, x):
        output = self.conv1(x)
        return output

    def update(self, cnn):
        for param, target_param in zip(cnn.parameters(), self.parameters()):
            target_param.data.copy_(param.data)