"""
浠?UserBehavior.csv 蹇€熸彁鍙栦竴涓湰鍦拌皟璇曠骇瀛愰泦銆?
鍙鍙栧墠 N_ROWS 琛岋紙榛樿 10M锛夛紝涓嶆壂鎻忓叏鏂囦欢锛岄€熷害杩滃揩浜?make_subset.py銆?

鍏稿瀷鐢ㄦ硶:
  python make_quick_subset.py
  python make_quick_subset.py --nrows 5000000 --target_users 2000 --min_inter 30
"""
import os
import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='UserBehavior.csv')
    parser.add_argument('--output_dir', default='data/taobao_subset')
    parser.add_argument('--nrows', type=int, default=10_000_000,
                        help='鍙鍙栧墠 N 琛屽師濮?CSV')
    parser.add_argument('--target_users', type=int, default=3000)
    parser.add_argument('--min_inter', type=int, default=30,
                        help='鐢ㄦ埛鏈€灏?PV 浜や簰鏁?)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"[1/3] 璇诲彇鍓?{args.nrows:,} 琛?{args.input} ...")
    df = pd.read_csv(args.input, header=None,
                     names=['u', 'i', 'c', 'b', 't'],
                     nrows=args.nrows)
    print(f"  鍘熷琛屾暟: {len(df):,}")

    df = df[df['b'] == 'pv'].copy()
    print(f"  PV 琛屾暟: {len(df):,}, 鐢ㄦ埛: {df['u'].nunique():,}, 鐗╁搧: {df['i'].nunique():,}")

    # 鎸変氦浜掓暟杩囨护
    uc = df['u'].value_counts()
    active = uc[uc > args.min_inter].index
    df = df[df['u'].isin(active)].copy()
    print(f"  >{args.min_inter} 浜や簰鍚? {len(df):,} 琛? {df['u'].nunique():,} 鐢ㄦ埛")

    # 閲囨牱鐩爣鏁伴噺鐢ㄦ埛
    unique_users = df['u'].unique()
    if len(unique_users) > args.target_users:
        sampled = np.random.choice(unique_users, args.target_users, replace=False)
        df = df[df['u'].isin(sampled)].copy()
    print(f"[2/3] 閲囨牱 {df['u'].nunique():,} 鐢ㄦ埛, {len(df):,} 琛? {df['i'].nunique():,} 鐗╁搧")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'UserBehavior.csv')
    df.to_csv(out_path, header=False, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[3/3] 宸蹭繚瀛樿嚦 {out_path} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
