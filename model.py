import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
# 引入自定义的模块，包括条件残差块（CResnetBlockConv1d）和条件归一化层（CBatchNorm1d）
from layers import (CResnetBlockConv1d, CBatchNorm1d, CBatchNorm1d_legacy)
import resnet

"""
定义一个 Decoder，用于根据条件特征解码点的占据概率logits。
"""
class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c 条件特征的维度（即由编码器生成的特征）
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure 是否启用旧的归一化方法。
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super(DecoderCBatchNorm, self).__init__()

        self.fc_p = nn.Conv1d(dim, hidden_size, 1) # 输入的点p通过一维卷积映射到隐藏层大小
        # 定义多个条件残差块，将条件特征c融入点的隐藏表示
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        
        # 根据参数选择新式或旧式的条件批归一化层
        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)
        
        # Ensure parameters are learnable
        for param in self.parameters():
            param.requires_grad = True
        self.fc_out = nn.Conv1d(hidden_size, 1, 1) # 最后一层卷积将隐藏表示映射到输出维度（占据概率 logits）

        # 根据参数选择 ReLU 或 Leaky ReLU 作为激活函数
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c, **kwargs):
        p = p.transpose(1, 2)    # 将点p的维度从 (B, N, D) 转置为 (B, D, N) 以匹配卷积层要求
        # batch_size, D, T = p.size()
        net = self.fc_p(p)       # 将输入点通过卷积层映射到隐藏空间

        # 依次通过多个条件残差块，将条件特征c注入隐藏表示
        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        # 对隐藏表示归一化（结合条件特征），应用激活函数，再映射到输出logits
        out = self.fc_out(self.actvn(self.bn(net, c)))

        # 移除logits的通道维度，返回结果
        out = out.squeeze(1)
        return out


"""
##########################################define network##########################################
"""

class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
    '''

    def __init__(self, z_dim=256):
        super(OccupancyNetwork, self).__init__()
        # encoder: 使用 ResNet-18 提取图像特征，特征维度为 z_dim=256
        self.encoder = resnet.Resnet18(z_dim)
        # decoder: 条件解码器，将特征 c 和点云 p 解码为占据概率
        self.decoder = DecoderCBatchNorm(dim=3, c_dim=256, hidden_size=256)
        # sigmoid: 用于将 logits 转换为概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, p):
        ''' Performs a forward pass through the network.
        1、提取图像特征：保留前3个通道的图像数据（RGB），通过编码器生成条件特征 c。
        2、解码点云：将点云p和特征c输入解码器，得到logits，使用伯努利分布建模，占据概率的 logits 即为 p_occ。
        3、生成占据概率：用sigmoid将logits转换为[0, 1]概率。
        Args:
            img (tensor): input image
            p (tensor): sampled points
        '''
        img = img[:, :3, :, :].contiguous()

        c = self.encoder(img)

        logits = self.decoder(p, c)
        p_occ = dist.Bernoulli(logits=logits).logits
        p_occ_sigmoid = self.sigmoid(p_occ)

        return p_occ, p_occ_sigmoid

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs) # 编码输入inputs为条件特征c
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0) # 如果编码器为空，返回空张量

        return c

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        # 解码点云 p 和条件特征 c，返回一个伯努利分布对象
        logits = self.decoder(p, c, **kwargs)
        # 检查并确保 decode 过程中的操作都支持反向传播
        print(f"Logits requires_grad: {logits.requires_grad}")
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def predict(self, img, pts):
        c = self.encoder(img)  # 提取图像特征 c

        # print('p_shape', p.size())
        pts_occ = self.decode(pts, c).logits     # 解码点 pts 的占据概率 logits
        pts_occ_sigmoid = self.sigmoid(pts_occ)  # 转换为概率

        return pts_occ_sigmoid