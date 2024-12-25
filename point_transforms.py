import numpy as np

# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data. 为点云添加高斯噪声

    Args:
        stddev (int): standard deviation
    '''
    # 初始化函数，接收一个参数 stddev，表示噪声的标准差
    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        # 创建 data 的副本，提取点云数据 points（通常是一个形如 [N, 3] 的数组，N 是点数）
        data_out = data.copy()
        points = data[None]
        # 生成与点云形状匹配的高斯噪声，其中 np.random.randn 生成均值为 0、标准差为 1 的随机数
        noise = self.stddev * np.random.randn(*points.shape)
        # 转换噪声为 float32 类型，并将其添加到原点云数据中
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out # 返回带有噪声的点云数据


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data. 对点云数据进行随机子采样

    Args:
        N (int): number of points to be subsampled
    '''
    # 初始化函数，接收 N，表示目标采样点的数量
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.
        
        Args:
            data (dict): data dictionary
        '''
        # 创建数据副本 data_out，提取点云数据 points 和法线数据 normals
        data_out = data.copy()
        points = data[None]
        normals = data['normals']
        # 随机生成 N 个索引值，用于从点云中采样
        indices = np.random.randint(points.shape[0], size=self.N)
        # 根据随机索引提取对应的点云和法线，并更新到输出数据中
        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.对占据点数据进行随机子采样，支持非均匀采样（根据占据值分为内部点和外部点）

    Args:
        N (int): number of points to be subsampled
    '''
    # 初始化函数，接收 N。可以是单个整数（表示总采样点数），
    # 也可以是二元组 (Nt_out, Nt_in)（分别表示采样外部点和内部点的数量）
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        # 提取点的位置 points 和占据值 occ（形如 [N, 1] 的数组，表示点是否被占据）
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        # 随机采样 (单整数 N)：随机从 points 中采样 N 个点，同时更新采样点的占据值 occ
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        # 非均匀采样 (二元组 N)
        else:
            # 根据 occ 的值分为外部点 points0（occ < 0.5）和内部点 points1（occ >= 0.5）
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            # 分别从 points0 和 points1 中随机采样 Nt_out 和 Nt_in 个点
            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)
            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            # 将采样的外部点和内部点拼接，同时生成对应的占据值（外部点为 0，内部点为 1）
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            # 计算被占据点的比例（作为额外的统计信息）
            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            # 更新子采样后的点云数据、占据值和占据比例
            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out
