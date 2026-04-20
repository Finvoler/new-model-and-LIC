import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Time-Aware LightGCN for Taobao Dataset.")
    
    # --- 基础配置参数 ---
    parser.add_argument('--dataset', nargs='?', default='taobao', help='Dataset name')
    parser.add_argument('--bpr_batch_size', type=int, default=2048, help='BPR Training batch size')
    parser.add_argument('--test_u_batch_size', type=int, default=100, help='Testing batch size')
    parser.add_argument('--latent_dim_rec', type=int, default=64, help='Latent dimension for embeddings')
    parser.add_argument('--lightGCN_n_layers', type=int, default=3, help='Number of LightGCN message passing layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--decay', type=float, default=1e-4, help='L2 regularization weight')
    parser.add_argument('--epochs', type=int, default=1000, help='Total training epochs')
    parser.add_argument('--topks', nargs='?', default='[20]', help='Top k items for evaluation metrics')
    parser.add_argument('--device', type=str, default='cuda', help='Hardware device (cuda/cpu)')
    
    # --- 新增：动态时间与软聚类参数 ---
    parser.add_argument('--n_clusters', type=int, default=4, help='Number of soft clusters (item time-behavior categories)')
    parser.add_argument('--fourier_k', type=int, default=3, help='Truncation order of Fourier series (K)')
    parser.add_argument('--tau', type=float, default=0.1, help='Temperature coefficient for soft clustering Softmax')
    parser.add_argument('--entropy_weight', type=float, default=0.01, help='Weight for entropy regularization to prevent cluster collapse')
    
    return parser.parse_args()