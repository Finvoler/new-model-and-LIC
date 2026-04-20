"""
浠庡畬鏁?UserBehavior.csv 涓娊鍙栦竴涓€傚悎鏈湴 GPU (濡?3070 16GB) 蹇€熻皟璇曠殑瀛愰泦銆?绛栫暐锛?  1. 娴佸紡璇诲彇锛屼粎淇濈暀 PV 琛屼负
  2. 缁熻姣忎釜鐢ㄦ埛鐨勪氦浜掓暟锛岀瓫閫?>min_inter 鐨勬椿璺冪敤鎴?  3. 浠庝腑闅忔満閲囨牱 target_users 涓敤鎴?  4. 淇濈暀杩欎簺鐢ㄦ埛鐨勬墍鏈?PV 浜や簰
  5. 杈撳嚭鍒?data/taobao_subset/UserBehavior.csv

棰勬湡瑙勬ā锛堥粯璁ゅ弬鏁帮級锛殈2000 鐢ㄦ埛銆亊200K 浜や簰銆亊50K 鐗╁搧
"""
import os
import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='UserBehavior.csv', help='鍘熷 CSV 璺緞')
    parser.add_argument('--output_dir', default='data/taobao_subset', help='杈撳嚭鐩綍')
    parser.add_argument('--target_users', type=int, default=2000, help='鐩爣鐢ㄦ埛鏁?)
    parser.add_argument('--min_inter', type=int, default=20, help='鐢ㄦ埛鏈€灏戜氦浜掓暟')
    parser.add_argument('--seed', type=int, default=2026, help='闅忔満绉嶅瓙')
    parser.add_argument('--chunksize', type=int, default=5_000_000, help='鍒嗗潡璇诲彇琛屾暟')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"[1/4] 娴佸紡璇诲彇 {args.input}锛岀粺璁＄敤鎴蜂氦浜掓暟...")
    user_counts = {}
    total_rows = 0
    for chunk in pd.read_csv(args.input, header=None,
                              names=['u', 'i', 'c', 'b', 't'],
                              chunksize=args.chunksize):
        pv = chunk[chunk['b'] == 'pv']
        for uid, cnt in pv['u'].value_counts().items():
            user_counts[uid] = user_counts.get(uid, 0) + cnt
        total_rows += len(chunk)
        print(f"  宸插鐞?{total_rows:,} 琛岋紝褰撳墠绱 {len(user_counts):,} 涓?PV 鐢ㄦ埛")

    # 绛涢€夋椿璺冪敤鎴?    active = {u for u, c in user_counts.items() if c > args.min_inter}
    print(f"\n[2/4] 婊¤冻 >{args.min_inter} 浜や簰鐨勭敤鎴? {len(active):,}")

    if len(active) < args.target_users:
        print(f"  娲昏穬鐢ㄦ埛涓嶈冻 {args.target_users}锛屽皢浣跨敤鍏ㄩ儴 {len(active)} 涓敤鎴?)
        sampled = active
    else:
        sampled = set(np.random.choice(list(active), args.target_users, replace=False))
    print(f"  閲囨牱 {len(sampled)} 涓敤鎴?)

    print(f"\n[3/4] 浜屾鎵弿锛屾彁鍙栫洰鏍囩敤鎴风殑 PV 浜や簰...")
    frames = []
    for chunk in pd.read_csv(args.input, header=None,
                              names=['u', 'i', 'c', 'b', 't'],
                              chunksize=args.chunksize):
        pv = chunk[(chunk['b'] == 'pv') & (chunk['u'].isin(sampled))]
        if len(pv) > 0:
            frames.append(pv)

    df = pd.concat(frames, ignore_index=True)
    print(f"  鎻愬彇瀹屾垚: {len(df):,} 琛? {df['u'].nunique()} 鐢ㄦ埛, {df['i'].nunique()} 鐗╁搧, {df['c'].nunique()} 绫诲埆")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'UserBehavior.csv')
    df.to_csv(out_path, header=False, index=False)
    file_size = os.path.getsize(out_path)
    print(f"\n[4/4] 宸蹭繚瀛樿嚦 {out_path} ({file_size / 1e6:.1f} MB)")
    print("鐢ㄦ硶: python main.py --min_interactions 20 (灏?dataloader path 鎸囧悜 data/taobao_subset)")

if __name__ == '__main__':
    main()
