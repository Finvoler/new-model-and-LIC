import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import time

# 导入你现有的模块
from dataloader import TaobaoDataset
from model import TimeAwareLightGCN
from Procedure import BPR_train_time, Test

def generate_mock_taobao_data(path="./data/taobao_mock"):
    """生成用于快速跑通流程的虚拟淘宝数据集"""
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, "UserBehavior.csv")
    
    print(">>> 正在生成虚拟数据集...")
    num_users = 500
    num_items = 1000
    num_interactions = 15000
    
    users = np.random.randint(1, num_users, num_interactions)
    items = np.random.randint(1, num_items, num_interactions)
    categories = np.random.randint(1, 50, num_interactions)
    behaviors = ['pv'] * num_interactions
    
    # 随机生成时间戳 (模拟一天内的分布)
    base_time = int(time.time())
    timestamps = [base_time + random.randint(0, 86400) for _ in range(num_interactions)]
    
    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'category_id': categories,
        'behavior_type': behaviors,
        'timestamp': timestamps
    })
    
    df.to_csv(csv_path, header=False, index=False)
    print(f">>> 虚拟数据集已保存至 {csv_path}")
    return path

def run_quick_test():
    # 1. 极简参数配置
    mock_path = generate_mock_taobao_data()
    
    config = {
        'dataset': 'taobao_mock',
        'bpr_batch_size': 256,
        'test_u_batch_size': 50,
        'latent_dim_rec': 32,          # 缩小维度加快测试
        'lightGCN_n_layers': 2,        # 减少图层数
        'lr': 0.005,
        'decay': 1e-4,
        'epochs': 2,                   # 只跑 2 个 Epoch
        'topks': [20],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_clusters': 3,
        'fourier_k': 2,
        'tau': 0.1,
        'entropy_weight': 0.01
    }
    
    print("\n" + "="*40)
    print("启动快速验证测试 (Quick Test)")
    print("使用设备:", config['device'])
    print("="*40)

    # 2. 实例化组件
    dataset = TaobaoDataset(config=config, path=mock_path)
    model = TimeAwareLightGCN(config=config, dataset=dataset).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 3. 执行流程验证
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n--- Epoch {epoch} ---")
        train_log = BPR_train_time(dataset, model, optimizer, config)
        print(f"训练输出: {train_log}")
        
        test_results = Test(dataset, model, config)
        print(f"测试输出: HR@20: {test_results['HR']:.4f} | NDCG@20: {test_results['NDCG']:.4f}")
        
    print("\n>>> 快速验证通过，代码逻辑无中断错误。")

if __name__ == '__main__':
    run_quick_test()