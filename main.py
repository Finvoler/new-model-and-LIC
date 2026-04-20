import torch
import torch.optim as optim
import numpy as np
import ast
from parse import parse_args
from dataloader import TaobaoDataset
from model import TimeAwareLightGCN
from Procedure import BPR_train_time, Test

def set_seed(seed=2026):
    """固定随机种子以保证实验的可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    # 1. 解析全局参数
    args = parse_args()
    config = vars(args)
    # 将字符串形式的 topks 转换为列表
    config['topks'] = ast.literal_eval(config['topks'])
    
    # 检查硬件并固定随机种子
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        config['device'] = 'cpu'
    set_seed(2026)
    
    print("="*50)
    print("配置清单:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*50)

    # 2. 实例化数据加载器
    print("[1/3] 正在加载并处理淘宝数据集...")
    dataset = TaobaoDataset(config=config, path="./")
    print(f"数据加载完成。用户数: {dataset.n_users}, 物品数: {dataset.m_items}")
    print(f"训练集交互数: {dataset.trainDataSize}, 图稀疏度: {dataset.trainDataSize / (dataset.n_users * dataset.m_items):.6f}")

    # 3. 实例化网络模型与优化器
    print("[2/3] 正在构建 Time-Aware LightGCN 神经网络结构...")
    model = TimeAwareLightGCN(config=config, dataset=dataset).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 4. 主循环：训练与测试
    print("[3/3] 开始训练...")
    best_ndcg = 0.0
    
    for epoch in range(1, config['epochs'] + 1):
        # 执行单轮训练
        train_log = BPR_train_time(dataset, model, optimizer, config)
        
        # 定期输出训练损失日志
        if epoch % 10 == 0 or epoch == 1:
            print(f"EPOCH[{epoch}/{config['epochs']}] {train_log}")
            
        # 定期进行全量排序测试
        if epoch % 20 == 0:
            test_results = Test(dataset, model, config)
            hr = test_results['HR']
            ndcg = test_results['NDCG']
            k = config['topks'][0]
            
            log_str = f">>> [TEST] EPOCH[{epoch}] HR@{k}: {hr:.4f} | NDCG@{k}: {ndcg:.4f}"
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                log_str += " (New Best!)"
                # 保存模型权重的逻辑可在此处添加
                # torch.save(model.state_dict(), './checkpoints/best_model.pth')
                
            print(log_str)

if __name__ == '__main__':
    main()