"""
绾?LightGCN 鍩虹嚎妯″瀷 鈥斺€?涓嶅惈浠讳綍鏃堕棿 / 鏃堕挓妯″潡銆?
鐢ㄤ簬闅旂娴嬭瘯 LightGCN 搴曞骇鍦ㄦ窐瀹濇暟鎹泦涓婃槸鍚﹁兘瀛﹀埌姝ｅ父鐨勫崗鍚岃繃婊や俊鍙枫€?

鎺ュ彛涓?TimeAwareLightGCN 瀹屽叏鍏煎:
  bpr_loss(users, pos, neg, thetas)   鈥斺€?thetas 浼犲叆浣嗗拷鐣?
  computer()
  score_all_items(users, thetas, precomputed=None)  鈥斺€?thetas 浼犲叆浣嗗拷鐣?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PureLightGCN(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = config['device']

        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']

        self.Graph = dataset.getSparseGraph()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def bpr_loss(self, users, pos, neg, thetas):
        """thetas 浼犲叆浣嗕笉浣跨敤锛屼繚鎸佹帴鍙ｅ吋瀹?""
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos]
        neg_emb = all_items[neg]

        # 绗?0 灞?embedding 鐢ㄤ簬 L2 姝ｅ垯
        userEmb0 = self.embedding_user(users)
        posEmb0 = self.embedding_item(pos)
        negEmb0 = self.embedding_item(neg)

        pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)

        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        reg_loss = (1 / 2) * (
            userEmb0.norm(2).pow(2)
            + posEmb0.norm(2).pow(2)
            + negEmb0.norm(2).pow(2)
        ) / float(len(users))

        # 鏃犺仛绫荤喌椤?
        entropy_loss = torch.zeros((), device=users.device)

        return bpr_loss, reg_loss, entropy_loss

    def score_all_items(self, users, thetas, precomputed=None):
        """杩斿洖 (B, M) 鍏ㄩ噺鎵撳垎鐭╅樀銆傚拷鐣?thetas銆?""
        if precomputed is None:
            all_users, all_items = self.computer()
        else:
            all_users, all_items = precomputed

        u_emb = all_users[users]
        ratings = torch.matmul(u_emb, all_items.T)
        return ratings
