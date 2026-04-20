import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeAwareLightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(TimeAwareLightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = config['device']
        
        # 基础参数
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']
        
        # 时间与聚类超参数
        self.n_clusters = config['n_clusters']  # n 类物品
        self.fourier_k = config['fourier_k']    # 傅里叶截断阶数 K
        self.tau = config['tau']                # Softmax 温度系数
        
        self.Graph = dataset.getSparseGraph()
        self.__init_weight()

    def __init_weight(self):
        """参数初始化：包含底层 Embedding、类别原型向量及傅里叶参数"""
        # 1. 静态底层 Embedding
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        # 2. 软聚类原型向量 C (n_clusters x d)
        self.cluster_prototypes = nn.Parameter(torch.randn(self.n_clusters, self.latent_dim) * 0.1)
        
        # 3. 傅里叶级数参数 (对应 n 个类别)
        # 直流分量 A_0: (n_clusters, d)
        self.fourier_a0 = nn.Parameter(torch.randn(self.n_clusters, self.latent_dim) * 0.1)
        # 余弦系数 A_k: (n_clusters, K, d)
        self.fourier_a = nn.Parameter(torch.randn(self.n_clusters, self.fourier_k, self.latent_dim) * 0.1)
        # 正弦系数 B_k: (n_clusters, K, d)
        self.fourier_b = nn.Parameter(torch.randn(self.n_clusters, self.fourier_k, self.latent_dim) * 0.1)
        
        # 预先生成频率序列 [1, 2, ..., K]，形状 (K, 1)，用于广播计算
        self.k_seq = torch.arange(1, self.fourier_k + 1, dtype=torch.float32, device=self.device).view(-1, 1)

        # 1. 生成一天 1440 分钟对应的标准 theta 角度 [0, 2π)
        minute_thetas = torch.linspace(0, 2 * np.pi, 1440, device=self.device).view(-1, 1) # (1440, 1)
        
        # 2. 预先乘以频率阶数 K
        k_thetas_table = minute_thetas * self.k_seq.view(1, -1) # (1440, K)
        
        # 3. 计算 cos 和 sin 并注册为 buffer (不参与梯度更新，但随模型保存在设备上)
        self.register_buffer('cos_table', torch.cos(k_thetas_table).unsqueeze(-1)) # (1440, K, 1)
        self.register_buffer('sin_table', torch.sin(k_thetas_table).unsqueeze(-1)) # (1440, K, 1)
        
    def computer(self):
        """标准 LightGCN 静态前向传播，计算高阶拓扑特征"""
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        g_droped = self.Graph
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def get_dynamic_embedding(self, item_static_emb, thetas):
        """
        计算特定时刻 theta 下的物品动态特征
        item_static_emb: (B, d) 传入的物品高阶静态特征
        thetas: (B,) 对应的绝对时间相位角
        """
        B = item_static_emb.size(0)
        
        # --- 1. 计算软聚类概率分布 P ---
        # 相似度打分 = 物品静态特征与原型向量的内积 / 温度系数 tau
        # scores 形状: (B, n_clusters)
        scores = torch.matmul(item_static_emb, self.cluster_prototypes.T) / self.tau
        # P 形状: (B, n_clusters)
        P = F.softmax(scores, dim=1)
        
        # --- 2. 基于查表的高效傅里叶基底计算 F(theta) ---
        # 将传入的连续 theta (0~2π) 映射为 0~1439 的分钟级索引
        # (thetas / (2 * np.pi) * 1440) 得到浮点索引，使用 .long() 转整数并 clamp 防止越界
        minute_indices = ((thetas / (2 * np.pi)) * 1440).long().clamp(0, 1439)
        
        # 直接从预计算好的表中通过索引取出当前 batch 对应的三角函数值，彻底消灭超越函数计算
        cos_vals = self.cos_table[minute_indices] # (B, K, 1)
        sin_vals = self.sin_table[minute_indices] # (B, K, 1)
        
        # 广播机制计算傅里叶级数: 
        # a0: (n, d) -> (1, n, d)
        # A_k: (n, K, d) -> (1, n, K, d)
        # cos_vals: (B, K, 1) -> (B, 1, K, 1) 
        # fourier_features 形状: (B, n_clusters, d)
        a_term = (self.fourier_a.unsqueeze(0) * cos_vals.unsqueeze(1)).sum(dim=2) # (B, n, d)
        b_term = (self.fourier_b.unsqueeze(0) * sin_vals.unsqueeze(1)).sum(dim=2) # (B, n, d)
        
        F_theta = self.fourier_a0.unsqueeze(0) + a_term + b_term
        
        # --- 3. 聚类概率与动态基底的降维融合 ---
        # 使用 einsum 将 P(B, n) 与 F_theta(B, n, d) 按 n 维度加权求和，输出 (B, d)
        item_dynamic_emb = torch.einsum('bn,bnd->bd', P, F_theta)
        
        return item_dynamic_emb, P

    def bpr_loss(self, users, pos, neg, thetas):
        """计算带时间戳的 BPR 损失及正则化项"""
        # 获取全图高阶静态特征
        all_users, all_items = self.computer()
        
        # 切片提取当前 batch 的静态特征 (B, d)
        users_emb = all_users[users]
        pos_emb_static = all_items[pos]
        neg_emb_static = all_items[neg]
        
        # 第 0 层原始特征，用于 L2 正则化
        userEmb0 = self.embedding_user(users)
        posEmb0 = self.embedding_item(pos)
        negEmb0 = self.embedding_item(neg)
        
        # 获取正负样本在 theta 时刻的动态特征及聚类分布
        # pos_emb_dyn: (B, d), P_pos: (B, n)
        pos_emb_dyn, P_pos = self.get_dynamic_embedding(pos_emb_static, thetas)
        neg_emb_dyn, P_neg = self.get_dynamic_embedding(neg_emb_static, thetas)
        
        # 静态与动态特征对齐相加
        pos_emb_final = pos_emb_static + pos_emb_dyn
        neg_emb_final = neg_emb_static + neg_emb_dyn
        
        # 计算偏好得分 (点积后按行求和)
        pos_scores = torch.mul(users_emb, pos_emb_final).sum(dim=1)
        neg_scores = torch.mul(users_emb, neg_emb_final).sum(dim=1)
        
        # 1. BPR Loss
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # 2. 聚类熵正则化 (防止聚类坍缩)
        # 计算当前 batch 所有正负样本在各类别上的平均概率分布 (n_clusters,)
        P_all = torch.cat([P_pos, P_neg], dim=0)
        p_mean = P_all.mean(dim=0)
        # 我们希望 p_mean 尽量接近均匀分布，即最大化 p_mean 的信息熵
        # 因此在 loss 中减去信息熵 (相当于加上负熵)
        entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
        entropy_loss = -entropy
        
        # 3. L2 正则化 (包含第 0 层 Embedding 与原型/傅里叶参数)
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                          posEmb0.norm(2).pow(2) + 
                          negEmb0.norm(2).pow(2)) / float(len(users))
        
        reg_loss += (1/2)*(self.cluster_prototypes.norm(2).pow(2) + 
                           self.fourier_a0.norm(2).pow(2) + 
                           self.fourier_a.norm(2).pow(2) + 
                           self.fourier_b.norm(2).pow(2)) / float(len(users))

        return bpr_loss, reg_loss, entropy_loss

    def predict(self, users, items, thetas):
        """
        测试阶段调用：计算用户在特定时刻 theta 对指定物品的偏好得分
        """
        all_users, all_items = self.computer()
        
        users_emb = all_users[users]      # (B, d)
        items_emb_static = all_items[items] # (B, d)
        
        items_emb_dyn, _ = self.get_dynamic_embedding(items_emb_static, thetas)
        items_emb_final = items_emb_static + items_emb_dyn
        
        scores = torch.mul(users_emb, items_emb_final).sum(dim=1)
        return scores