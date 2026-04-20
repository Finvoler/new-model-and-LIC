import numpy as np
import torch
from tqdm import tqdm

def UniformSample_time(dataset):
    """
    带有绝对时间戳相位的 1:1 均匀负采样器。
    采用基于边（Edge）的采样策略以提升检索 $(u, i, \theta)$ 的速度。
    """
    dataset_size = dataset.trainDataSize
    
    # 1. 随机抽取交互边索引
    sampled_indices = np.random.randint(0, dataset_size, dataset_size)
    
    # 2. 向量化提取当前批次的用户、正样本及其对应的交互时间相位
    users = dataset.train_users[sampled_indices]
    pos_items = dataset.train_items[sampled_indices]
    thetas = dataset.train_thetas[sampled_indices]
    
    # 3. 负样本采样（需经过全局历史过滤）
    neg_items = np.zeros(dataset_size, dtype=np.int32)
    
    # 依然需要对每个采样事件生成独立的不重合负样本
    for i, u in enumerate(tqdm(users, desc="[采样阶段] 生成全局负样本", leave=False)):
        while True:
            # 从全局物品池中随机抽取
            neg_id = np.random.randint(0, dataset.m_items)
            # 过滤逻辑：必须使用 all_dict，防止测试集数据泄露为负样本
            if neg_id not in dataset.all_dict[u]:
                neg_items[i] = neg_id
                break
                
    return users, pos_items, neg_items, thetas

def minibatch(*tensors, batch_size):
    """
    通用的批次切分生成器
    """
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

# ==================== 评价指标计算 ====================
def RecallPrecision_ATk(test_data, r, k):
    """
    计算单个用户的 Recall 和 Precision
    test_data: 该用户在测试集中的真实物品数量
    r: 长度为 K 的布尔数组，表示推荐的 top-k 物品是否命中测试集
    """
    right_pred = r.sum()
    recall = right_pred / len(test_data)
    precision = right_pred / k
    return recall, precision

def NDCGatK_r(test_data, r, k):
    """
    计算单个用户的 NDCG
    """
    assert len(r) == k
    pred_data = r
    test_matrix = np.zeros(k)
    # 计算理想情况下的 IDCG
    test_matrix[:min(len(test_data), k)] = 1
    
    # 折损权重：1 / log2(rank + 2)
    weights = 1. / np.log2(np.arange(2, k + 2))
    
    DCG = np.sum(pred_data * weights)
    IDCG = np.sum(test_matrix * weights)
    
    if IDCG == 0:
        return 0.
    return DCG / IDCG

def HitRatio_ATk(r):
    """
    计算单个用户的命中率 (Hit Ratio)
    """
    return 1. if r.sum() > 0 else 0.