import numpy as np
import torch
import torch.nn.functional as F
import utils
import time
from tqdm import tqdm

def BPR_train_time(dataset, recommend_model, optimizer, config):
    """
    带有绝对时间相位的动态模型训练流程
    """
    recommend_model.train()
    bpr_loss_total = 0.
    reg_loss_total = 0.
    entropy_loss_total = 0.
    
    # 1. 执行全局交互边的均匀负采样
    S = utils.UniformSample_time(dataset)
    users = torch.Tensor(S[0]).long().to(config['device'])
    posItems = torch.Tensor(S[1]).long().to(config['device'])
    negItems = torch.Tensor(S[2]).long().to(config['device'])
    thetas = torch.Tensor(S[3]).float().to(config['device'])
    
    # 2. 批次化训练
    batch_size = config['bpr_batch_size']
    n_batches = len(users) // batch_size + 1
    
    # 构建带进度条的迭代器
    pbar = tqdm(utils.minibatch(users, posItems, negItems, thetas, batch_size=batch_size), 
                total=n_batches, desc="[模型训练] BPR Forward & Backward", leave=False)
    
    for (batch_users, batch_pos, batch_neg, batch_thetas) in pbar:
        optimizer.zero_grad()
        bpr_loss, reg_loss, entropy_loss = recommend_model.bpr_loss(
            batch_users, batch_pos, batch_neg, batch_thetas
        )
        
        # 2. 计算出 bpr_loss 后，再更新进度条显示
        pbar.set_postfix({'bpr': f'{bpr_loss.item():.4f}'})
        
        # 提取各项权重
        weight_decay = config['decay']
        entropy_weight = config['entropy_weight']
        
        # 组合总 Loss
        loss = bpr_loss + weight_decay * reg_loss + entropy_weight * entropy_loss
        
        loss.backward()
        optimizer.step()
        
        bpr_loss_total += bpr_loss.item()
        reg_loss_total += reg_loss.item()
        entropy_loss_total += entropy_loss.item()
        
    avg_bpr = bpr_loss_total / n_batches
    avg_reg = reg_loss_total / n_batches
    avg_entropy = entropy_loss_total / n_batches
    
    return f"BPR: {avg_bpr:.4f} | Reg: {avg_reg:.4f} | Entropy: {avg_entropy:.4f}"

def Test(dataset, recommend_model, config):
    """
    实例级别的动态时间全量排名评估
    由于物品特征随测试用例的时间 theta 动态变化，必须进行三维张量的批处理计算
    """
    recommend_model.eval()
    
    # 提取所有测试实例
    test_users = torch.Tensor(dataset.test_users).long().to(config['device'])
    test_items = torch.Tensor(dataset.test_items).long().to(config['device'])
    test_thetas = torch.Tensor(dataset.test_thetas).float().to(config['device'])
    
    total_instances = len(test_users)
    batch_size = config['test_u_batch_size']
    
    # --- 工程修改：计算 Batch 总数用于显示进度条 ---
    # 使用向上取整除法计算总 batch 数
    num_batches = (total_instances + batch_size - 1) // batch_size
    
    top_k = config['topks'][0] # 默认取列表中第一个，如 20
    
    HR_list = []
    NDCG_list = []
    
    with torch.no_grad():
        # 获取基础的全局静态特征
        all_users_static, all_items_static = recommend_model.computer()
        
        # 聚类分布 P 对所有物品是全局静态的，仅需计算一次
        # P 形状: (m_items, n_clusters)
        scores_P = torch.matmul(all_items_static, recommend_model.cluster_prototypes.T) / recommend_model.tau
        P = F.softmax(scores_P, dim=1)
        
        # --- 工程修改：给 minibatch 迭代器加上 tqdm 进度条 ---
        # desc 设置左侧文本，total 设置总 batch 数，leave=False 结束后自动清除
        test_loader = utils.minibatch(test_users, test_items, test_thetas, batch_size=batch_size)
        pbar = tqdm(test_loader, desc="[模型评估] 全量动态排序", total=num_batches, leave=False)
        
        for (batch_u, batch_i_target, batch_theta) in pbar:
            
            B = len(batch_u)
            u_emb = all_users_static[batch_u] # (B, d)
            
            # --- 批量计算当前 batch 时间对应的动态基底 F(theta) ---
            k_seq = recommend_model.k_seq.view(1, -1) # (1, K)
            k_theta = batch_theta.view(-1, 1) * k_seq # (B, K)
            
            cos_vals = torch.cos(k_theta).unsqueeze(-1) # (B, K, 1)
            sin_vals = torch.sin(k_theta).unsqueeze(-1) # (B, K, 1)
            
            # 广播计算傅里叶展开，输出 F_theta: (B, n_clusters, d)
            a_term = (recommend_model.fourier_a.unsqueeze(0) * cos_vals.unsqueeze(1)).sum(dim=2)
            b_term = (recommend_model.fourier_b.unsqueeze(0) * sin_vals.unsqueeze(1)).sum(dim=2)
            F_theta = recommend_model.fourier_a0.unsqueeze(0) + a_term + b_term
            
            # --- 计算全量物品的动态特征并打分 ---
            # 使用 einsum 将 P (M, n) 与 F_theta (B, n, d) 结合
            # 输出 item_dynamic_emb: (B, M, d)
            item_dynamic_emb = torch.einsum('mn,bnd->bmd', P, F_theta)
            
            # 物品最终特征 = 静态特征 (扩展为 1, M, d) + 动态特征 (B, M, d)
            item_final_emb = all_items_static.unsqueeze(0) + item_dynamic_emb
            
            # 预测得分矩阵 ratings: (B, M)
            ratings = torch.einsum('bd,bmd->bm', u_emb, item_final_emb)
            
            # --- 训练集 Mask 过滤 ---
            # 必须过滤掉用户在训练集中已经交互过的物品
            for i, user in enumerate(batch_u):
                user_id = user.item()
                if user_id in dataset.train_dict:
                    train_items_idx = dataset.train_dict[user_id]
                    ratings[i, train_items_idx] = -np.inf
            
            # --- 获取 Top-K 并计算指标 ---
            # 获取排名最高的前 K 个物品的索引: (B, K)
            _, top_k_indices = torch.topk(ratings, k=top_k, dim=1)
            
            # 验证目标测试物品 target_i 是否存在于 top_k_indices 中
            # targets_expanded: (B, 1)
            targets_expanded = batch_i_target.unsqueeze(1)
            # hits: (B, K) 的布尔矩阵
            hits = (top_k_indices == targets_expanded)
            
            for i in range(B):
                hit_array = hits[i].cpu().numpy()
                HR_list.append(utils.HitRatio_ATk(hit_array))
                NDCG_list.append(utils.NDCGatK_r([batch_i_target[i].item()], hit_array, top_k))
                
    final_HR = np.mean(HR_list)
    final_NDCG = np.mean(NDCG_list)
    
    return {'HR': final_HR, 'NDCG': final_NDCG}