import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
# 引入自定义的模块，包括条件残差块（CResnetBlockConv1d）和条件归一化层（CBatchNorm1d）
from layers import (CResnetBlockConv1d, CBatchNorm1d, CBatchNorm1d_legacy)
import resnet

"""
池化层，通过最大值池化将点云降维
"""
class PointPooling(nn.Module):
    def __init__(self, input_dim, output_dim, pooling_factor=2):
        super(PointPooling, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, 1)  # 卷积层
        self.pooling_factor = pooling_factor

    def forward(self, x):
        # 输入形状 (B, D, N)，D 为维度，N 为点数
        x = self.conv(x)  # 维度映射
        x = F.max_pool1d(x, self.pooling_factor)  # 池化降维
        return x

"""
通过 Farthest Point Sampling (FPS) 对点云进行聚类降维
"""
def farthest_point_sampling(points, num_samples):
    """
    最远点采样（Farthest Point Sampling）。
    
    Args:
        points (torch.Tensor): 输入点云，形状为 (B, N, D)。
        num_samples (int): 采样的点数量。
    
    Returns:
        torch.Tensor: 采样后的点云，形状为 (B, num_samples, D)。
    """
    B, N, D = points.shape
    sampled_points = torch.zeros(B, num_samples, D, device=points.device)
    indices = torch.zeros(B, num_samples, dtype=torch.long, device=points.device)
    
    for b in range(B):
        # 初始化
        distances = torch.ones(N, device=points.device) * 1e10
        farthest = torch.randint(0, N, (1,), device=points.device)
        
        for i in range(num_samples):
            indices[b, i] = farthest
            sampled_points[b, i] = points[b, farthest]
            dist = torch.norm(points[b] - points[b, farthest], dim=1)
            distances = torch.min(distances, dist)
            farthest = torch.argmax(distances)
    
    return sampled_points

"""
对输入点云进行随机采样从而减少序列长度, 减少显存占用
"""
def random_sampling(points, num_samples):
    """
    随机从点云中采样固定数量的点。
    
    Args:
        points (torch.Tensor): 输入点云，形状为 (B, N, D)。
        num_samples (int): 采样的点数量。
    
    Returns:
        torch.Tensor: 采样后的点云，形状为 (B, num_samples, D)。
    """
    batch_size, num_points, dim = points.shape
    if num_points <= num_samples:
        return points  # 如果点数小于采样点数，直接返回原始点云
    
    # 随机生成采样索引
    indices = torch.randperm(num_points)[:num_samples].to(points.device)
    sampled_points = points[:, indices, :]
    return sampled_points

"""

自注意力模块
在 DecoderCBatchNorm 解码器中引入自注意力机制, 残差块之间加入了自注意力模块 attn1 和 attn2
"""
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)

    def forward(self, x):
        # 输入 x 形状: (batch_size, hidden_size, num_points)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, num_points, hidden_size)
        attn_output, _ = self.attention(x, x, x)  # 自注意力
        x = attn_output + x  # 残差连接
        x = x.permute(0, 2, 1)  # 转换回 (batch_size, hidden_size, num_points)
        return x

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
        # self.pooling = PointPooling(hidden_size, hidden_size, pooling_factor=2)  # 添加池化层
        
        # 定义多个条件残差块，将条件特征c融入点的隐藏表示
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.attn1 = SelfAttention(hidden_size)  # 加入自注意力机制
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.attn2 = SelfAttention(hidden_size)  # 再次加入自注意力机制
        
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

    def forward(self, p, c, num_samples=4096, **kwargs):
        # 使用随机采样
        # p = random_sampling(p, num_samples=num_samples)
        
        p = p.transpose(1, 2)    # 将点p的维度从 (B, N, D) 转置为 (B, D, N) 以匹配卷积层要求
        # batch_size, D, T = p.size()
        net = self.fc_p(p)       # 将输入点通过卷积层映射到隐藏空间

        # net = self.pooling(net)  # 池化操作降低维数
        
        # 依次通过多个条件残差块，将条件特征c注入隐藏表示
        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.attn1(net)     # 自注意力
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)
        net = self.attn2(net)     # 自注意力

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
        # self.encoder = resnet.Resnet18(z_dim, input_channels=1)
        self.encoder = resnet.Resnet34(z_dim, input_channels=1)    # 改用更深的resnet
        
        # 添加 Dropout 层，用于在特征编码后随机丢弃一部分特征
        # self.dropout = nn.Dropout(p=dropout_prob)  # Dropout 概率默认为 30%
        
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
        # img = img[:, :3, :, :].contiguous()

        c = self.encoder(img)
        
        # 在特征 c 上应用 Dropout
        # c = self.dropout(c)

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
            # c = self.dropout(c)      # 应用 Dropout
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
        # c = self.dropout(c)    # 应用 Dropout

        # print('p_shape', p.size())
        pts_occ = self.decode(pts, c).logits     # 解码点 pts 的占据概率 logits
        pts_occ_sigmoid = self.sigmoid(pts_occ)  # 转换为概率

        return pts_occ_sigmoid
