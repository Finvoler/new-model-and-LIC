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
        娣樺疂鏁版嵁闆嗗姞杞藉櫒銆?
        鏍稿績澶勭悊閫昏緫锛氫繚鐣?'pv' 琛屼负锛岄噸鏄犲皠 ID锛屽皢鏃堕棿鎴崇浉浣嶅寲锛屾瀯寤哄綊涓€鍖栨媺鏅媺鏂煩闃点€?
        """
        self.config = config
        self.path = path
        
        # 1. 璇诲彇涓庤繃婊?
        # 娣樺疂 UserBehavior.csv 榛樿鏃犺〃澶达紝鎸囧畾鍒楀悕
        csv_file = os.path.join(path, "UserBehavior.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"鏈壘鍒版暟鎹泦鏂囦欢: {csv_file}")
            
        df = pd.read_csv(csv_file, header=None, 
                         names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])
        
        # 浠呬繚鐣?pv 琛屼负
        df = df[df['behavior_type'] == 'pv'].copy()

        # --- 闈炶凯浠ｅ崟杞繃婊わ紙閬垮厤 k-core 绾ц仈鍧嶅锛?--
        # 鍏堢瓫鐑棬鐗╁搧锛屽啀绛涙椿璺冪敤鎴凤紝鏈€鍚庢竻鐞嗘畫浣欎綆棰戠墿鍝?
        min_user_inter = config.get('min_user_inter', 100)
        min_item_inter = config.get('min_item_inter', 1000)
        
        # 绗?杞細鍩轰簬鍏ㄥ眬浜や簰鏁拌繃婊ょ墿鍝?
        ic = df['item_id'].value_counts()
        df = df[df['item_id'].isin(ic[ic >= min_item_inter].index)]
        # 绗?杞細杩囨护鐢ㄦ埛
        uc = df['user_id'].value_counts()
        df = df[df['user_id'].isin(uc[uc >= min_user_inter].index)]
        # 绗?杞細娓呯悊鍥犵敤鎴峰垹闄よ€屽彉绋€鐤忕殑鐗╁搧锛堜笅闄?10闃叉姝绘暟鎹紝涓嶄細瑙﹀彂绾ц仈锛?
        ic2 = df['item_id'].value_counts()
        df = df[df['item_id'].isin(ic2[ic2 >= 10].index)]

        # --- (user, item) 鍘婚噸锛氭秷闄?train/test 鐗╁搧閲嶅彔 ---
        # 淇濈暀鏈€鍚庝竴娆′氦浜掞紙淇濈暀鏈€鏂扮殑 theta 鍜?timestamp锛?
        before_dedup = len(df)
        df = df.sort_values('timestamp').drop_duplicates(
            subset=['user_id', 'item_id'], keep='last'
        )
        print(f"鍘婚噸: {before_dedup} 鈫?{len(df)} 鏉?"
              f"(鍘婚櫎 {before_dedup - len(df)} 鏉￠噸澶?user-item 瀵?")
        
        # 鍘婚噸鍚庝簩娆℃竻鐞嗭細纭繚鐢ㄦ埛/鐗╁搧浜や簰鏁颁粛瓒冲
        uc2 = df['user_id'].value_counts()
        df = df[df['user_id'].isin(uc2[uc2 >= 10].index)]
        ic3 = df['item_id'].value_counts()
        df = df[df['item_id'].isin(ic3[ic3 >= 5].index)]
        
        df = df.copy()
        print(f"绛涢€夊悗: 鐢ㄦ埛鏁?{df['user_id'].nunique()}, "
              f"鐗╁搧鏁?{df['item_id'].nunique()}, 浜や簰鏁?{len(df)}, "
              f"瀵嗗害 {len(df)/(df['user_id'].nunique()*df['item_id'].nunique()+1e-9):.6f}")

        # 2. ID 搴忓垪鍖?
        # 浣跨敤 Pandas 鐨?category 绫诲瀷杩涜蹇€熺殑杩炵画鏁存暟鏄犲皠 (0 鍒?N-1)
        df['user_id'] = df['user_id'].astype('category').cat.codes
        df['item_id'] = df['item_id'].astype('category').cat.codes
        # category_id 棰勭暀 0 浣滀负 padding锛岀湡瀹炵被鍒粠 1 寮€濮?
        df['category_id'] = df['category_id'].astype('category').cat.codes + 1
        
        self.n_users = df['user_id'].nunique()
        self.m_items = df['item_id'].nunique()
        # 绫诲埆鎬绘暟鍖呭惈 padding 浣?0
        self.n_categories = int(df['category_id'].nunique()) + 1

        # 鐗╁搧鍒扮被鍒槧灏勶紙娣樺疂鍦烘櫙涓?item 閫氬父瀵瑰簲鍗曚竴绫荤洰锛?
        self.item2category = np.zeros(self.m_items, dtype=np.int64)
        item_cat = df[['item_id', 'category_id']].drop_duplicates(subset=['item_id'])
        self.item2category[item_cat['item_id'].to_numpy()] = item_cat['category_id'].to_numpy()
        
        # 3. 鏃堕棿鎴崇浉浣嶅寲 (澶╁唴缁濆鏃堕棿褰掍竴鍖栧埌 [0, 2蟺))
        df['theta'] = (df['timestamp'] % 86400) / 86400.0 * 2 * np.pi
        
        # 4. 鎵撲贡涓庡垏鍒?(鎸夌敤鎴风骇鍒?8:2)
        self.train_users, self.train_items, self.train_thetas = [], [], []
        self.test_users, self.test_items, self.test_thetas = [], [], []
        
        # 瀛楀吀缁撴瀯锛氱敤浜庡姞閫?utils.py 涓殑璐熼噰鏍烽獙璇佷笌娴嬭瘯闆嗚繃婊?
        self.train_dict = {} 
        self.all_dict = {}   
        
        # 瀛樺湪鏇翠紭瑙ｏ細姝ゅ涓嶄娇鐢?iterrows()锛岄€氳繃 groupby 鎻愬彇搴曞眰 numpy array 鎻愬崌 10 鍊嶄互涓婇€熷害
        df_grouped = df.groupby('user_id')
        for user, group in tqdm(df_grouped, desc="[鏁版嵁鍑嗗] 鍒囧垎涓庢瀯寤哄瓧鍏?, leave=True):
            group_items = group['item_id'].values
            group_thetas = group['theta'].values
            
            # 瀵瑰綋鍓嶇敤鎴峰巻鍙茶繘琛岄殢鏈烘墦涔?
            indices = np.random.permutation(len(group_items))
            group_items = group_items[indices]
            group_thetas = group_thetas[indices]
            
            # 8:2 闈欐€佸垏鍒?
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
                
            # 鍏ㄩ噺鍘嗗彶浜や簰锛岃礋閲囨牱鏃跺繀椤荤‘淇濊礋鏍锋湰涓嶅湪 all_dict 涓?
            self.all_dict[user] = set(group_items.tolist())
            
        self.train_users = np.array(self.train_users)
        self.train_items = np.array(self.train_items)
        self.train_thetas = np.array(self.train_thetas)
        self.trainDataSize = len(self.train_users)
        
        self.test_users = np.array(self.test_users)
        self.test_items = np.array(self.test_items)
        self.test_thetas = np.array(self.test_thetas)
        
        # 5. 寤惰繜鏋勫缓绋€鐤忓浘
        self.Graph = None

    def getSparseGraph(self):
        """
        浣跨敤鍒嗗潡鐭╅樀鎷兼帴 (bmat) 浠ｆ浛鍒囩墖璧嬪€硷紝褰诲簳瑙ｅ喅 29TiB 鍐呭瓨鎶ラ敊闂
        """
        if self.Graph is None:
            import scipy.sparse as sp
            print("姝ｅ湪鏋勫缓绋€鐤忓浘缁撴瀯 (浣跨敤 bmat 浼樺寲)...")
            
            # 鏋勫缓浜や簰鐭╅樀 R (Sparse)
            R = sp.csr_matrix(
                (np.ones(len(self.train_users)), (self.train_users, self.train_items)), 
                shape=(self.n_users, self.m_items), dtype=np.float32
            )
            
            # 浣跨敤 bmat 鏋勫缓瀵圭О閭绘帴鐭╅樀:
            # [[ 0, R ],
            #  [ R^T, 0 ]]
            adj_mat = sp.bmat([
                [None, R],
                [R.T, None]
            ], format='csr')
            
            # 璁＄畻 D^{-0.5} * A * D^{-0.5}
            rowsum = np.array(adj_mat.sum(axis=1)).flatten()
            with np.errstate(divide='ignore'):   # 鏂板涓婁笅鏂囩鐞嗗櫒鍘嬪埗璀﹀憡
                d_inv = np.power(rowsum, -0.5)
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat).dot(d_mat)
            
            # 杞崲涓?PyTorch 绋€鐤忓紶閲?
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.config['device'])
            
        return self.Graph
        
    def _convert_sp_mat_to_sp_tensor(self, X):
        """灏?scipy 绋€鐤忕煩闃佃浆鎹负 PyTorch 绋€鐤忓紶閲?""
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        
        # 鏇挎崲鍘熸湰搴熷純鐨?torch.sparse.FloatTensor
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float32)
