"""
鏈湴 3070 (16GB) 杞婚噺璁粌鑴氭湰
浣跨敤鍓嶈鍏堣繍琛?make_subset.py 鐢熸垚瀛愰泦鏁版嵁锛?
  python make_subset.py --target_users 2000 --min_inter 20
"""
import subprocess
import sys

# ============= 鍏叡鍩虹鍙傛暟 =============
BASE = [
    sys.executable, "main.py",
    "--data_path", "./data/taobao_subset",
    "--min_interactions", "20",
    "--latent_dim_rec", "32",
    "--lightGCN_n_layers", "2",
    "--bpr_batch_size", "2048",
    "--test_u_batch_size", "256",
    "--lr", "0.001",
    "--decay", "1e-4",
    "--epochs", "200",
    "--topks", "[10,20]",
    "--device", "cuda",
]

# ============= 涓夌粍瀹為獙閰嶇疆 =============
configs = {
    "fourier": BASE + [
        "--temporal_model", "fourier",
        "--n_clusters", "4",
        "--fourier_k", "3",
        "--tau", "0.5",
        "--entropy_weight", "0.01",
        "--fusion_mode", "add",
    ],
    "gaussian": BASE + [
        "--temporal_model", "gaussian",
        "--clock_emb_dim", "32",
        "--time_diff_alpha", "8.0",
        "--clock_gaussian_sigma", "1.0",
    ],
    "lic": BASE + [
        "--temporal_model", "lic",
        "--lic_top_k", "50",
        "--lic_n_heads", "2",
        "--lic_d", "16",
        "--lic_alpha", "1.0",
    ],
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(configs.keys()), help="閫夋嫨瑕佽繍琛岀殑妯″瀷")
    args = parser.parse_args()

    cmd = configs[args.model]
    print(f">>> 杩愯 {args.model} 妯″瀷")
    print(f">>> 鍛戒护: {' '.join(cmd)}")
    print("=" * 60)
    subprocess.run(cmd, cwd=".")
