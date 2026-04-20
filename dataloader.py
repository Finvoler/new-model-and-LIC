import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TaobaoDataset(Dataset):
    def __init__(self, config, path="./data/taobao"):
        """
        淘宝数据集加载器。
        核心处理逻辑：保留 'pv' 行为，重映射 ID，将时间戳相位化，构建归一化拉普拉斯矩阵。
        """
        self.config = config
        self.path = path
        
        # 1. 读取与过滤
        # 淘宝 UserBehavior.csv 默认无表头，指定列名
        csv_file = os.path.join(path, "UserBehavior.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"未找到数据集文件: {csv_file}")
            
        df = pd.read_csv(csv_file, header=None, 
                         names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])
        
        # 仅保留 pv 行为
        df = df[df['behavior_type'] == 'pv'].copy()
        user_counts = df['user_id'].value_counts()
        active_users = user_counts[user_counts > 200].index
        df = df[df['user_id'].isin(active_users)].copy()
        print(f"筛选后保留用户数: {len(active_users)}, 剩余交互数: {len(df)}")
        # 2. ID 序列化
        # 使用 Pandas 的 category 类型进行快速的连续整数映射 (0 到 N-1)
        df['user_id'] = df['user_id'].astype('category').cat.codes
        df['item_id'] = df['item_id'].astype('category').cat.codes
        
        self.n_users = df['user_id'].nunique()
        self.m_items = df['item_id'].nunique()
        
        # 3. 时间戳相位化 (天内绝对时间归一化到 [0, 2π))
        df['theta'] = (df['timestamp'] % 86400) / 86400.0 * 2 * np.pi
        
        # 4. 打乱与切分 (按用户级别 8:2)
        self.train_users, self.train_items, self.train_thetas = [], [], []
        self.test_users, self.test_items, self.test_thetas = [], [], []
        
        # 字典结构：用于加速 utils.py 中的负采样验证与测试集过滤
        self.train_dict = {} 
        self.all_dict = {}   
        
        # 存在更优解：此处不使用 iterrows()，通过 groupby 提取底层 numpy array 提升 10 倍以上速度
        df_grouped = df.groupby('user_id')
        for user, group in tqdm(df_grouped, desc="[数据准备] 切分与构建字典", leave=True):
            group_items = group['item_id'].values
            group_thetas = group['theta'].values
            
            # 对当前用户历史进行随机打乱
            indices = np.random.permutation(len(group_items))
            group_items = group_items[indices]
            group_thetas = group_thetas[indices]
            
            # 8:2 静态切分
            split_idx = int(len(group_items) * 0.8)
            
            tr_items = group_items[:split_idx]
            tr_thetas = group_thetas[:split_idx]
            te_items = group_items[split_idx:]
            te_thetas = group_thetas[split_idx:]
            
            if len(tr_items) > 0:
                self.train_users.extend([user] * len(tr_items))
                self.train_items.extend(tr_items)
                self.train_thetas.extend(tr_thetas)
                self.train_dict[user] = tr_items
            
            if len(te_items) > 0:
                self.test_users.extend([user] * len(te_items))
                self.test_items.extend(te_items)
                self.test_thetas.extend(te_thetas)
                
            # 全量历史交互，负采样时必须确保负样本不在 all_dict 中
            self.all_dict[user] = set(group_items.tolist())
            
        self.train_users = np.array(self.train_users)
        self.train_items = np.array(self.train_items)
        self.train_thetas = np.array(self.train_thetas)
        self.trainDataSize = len(self.train_users)
        
        self.test_users = np.array(self.test_users)
        self.test_items = np.array(self.test_items)
        self.test_thetas = np.array(self.test_thetas)
        
        # 5. 延迟构建稀疏图
        self.Graph = None

    def getSparseGraph(self):
        """
        使用分块矩阵拼接 (bmat) 代替切片赋值，彻底解决 29TiB 内存报错问题
        """
        if self.Graph is None:
            import scipy.sparse as sp
            print("正在构建稀疏图结构 (使用 bmat 优化)...")
            
            # 构建交互矩阵 R (Sparse)
            R = sp.csr_matrix(
                (np.ones(len(self.train_users)), (self.train_users, self.train_items)), 
                shape=(self.n_users, self.m_items), dtype=np.float32
            )
            
            # 使用 bmat 构建对称邻接矩阵:
            # [[ 0, R ],
            #  [ R^T, 0 ]]
            adj_mat = sp.bmat([
                [None, R],
                [R.T, None]
            ], format='csr')
            
            # 计算 D^{-0.5} * A * D^{-0.5}
            rowsum = np.array(adj_mat.sum(axis=1)).flatten()
            d_inv = np.power(rowsum, -0.5)
            with np.errstate(divide='ignore'):   # 新增上下文管理器压制警告
                d_inv = np.power(rowsum, -0.5)
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat).dot(d_mat)
            
            # 转换为 PyTorch 稀疏张量
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.config['device'])
            
        return self.Graph
        
    def _convert_sp_mat_to_sp_tensor(self, X):
        """将 scipy 稀疏矩阵转换为 PyTorch 稀疏张量"""
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        
        # 替换原本废弃的 torch.sparse.FloatTensor
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float32)