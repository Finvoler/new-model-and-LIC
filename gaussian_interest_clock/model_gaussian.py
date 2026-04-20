import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianClockLightGCN(nn.Module):
    """
    LightGCN backbone + Gaussian interest clock.
    杈撳叆鎺ュ彛涓庣幇鏈?TimeAwareLightGCN 淇濇寔涓€鑷达細
    - bpr_loss(users, pos, neg, thetas)
    - computer()
    - score_all_items(users, thetas, precomputed=None)
    """

    def __init__(self, config, dataset):
        super(GaussianClockLightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = config['device']

        # LightGCN backbone config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']

        # Interest Clock config (paper-aligned hour buckets + Gaussian smoothing)
        self.time_bins = 24
        self.topk_categories = 3
        self.clock_emb_dim = config['clock_emb_dim']
        self.clock_alpha = config['time_diff_alpha']
        self.clock_gaussian_mu = config['clock_gaussian_mu']
        self.clock_gaussian_sigma = max(config['clock_gaussian_sigma'], 1e-6)

        self.Graph = dataset.getSparseGraph()
        self.__init_weight()

    def __init_weight(self):
        # 1) Static embeddings (LightGCN backbone)
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # 2) Category embedding (娣樺疂涓互 category 鏇夸唬 genre/mood/lang)
        self.padding_idx = 0
        self.category_embedding = nn.Embedding(
            num_embeddings=self.dataset.n_categories,
            embedding_dim=self.clock_emb_dim,
            padding_idx=self.padding_idx,
        )
        nn.init.normal_(self.category_embedding.weight, std=0.1)
        with torch.no_grad():
            self.category_embedding.weight[self.padding_idx].fill_(0.0)

        # 3) 棰勫瓨鐗╁搧绫诲埆绱㈠紩涓庣敤鎴?灏忔椂 Top-3 绫诲埆鐗瑰緛
        self.register_buffer('item_category_ids', torch.from_numpy(self.dataset.item2category).long())
        top3 = self._build_user_hour_top3_categories()
        self.register_buffer('user_hour_top3_categories', top3)
        self.register_buffer('hour_indices', torch.arange(self.time_bins, dtype=torch.float32))

    def _build_user_hour_top3_categories(self):
        """
        绂荤嚎缁熻锛氭寜璁粌鏁版嵁鏋勫缓 [num_users, 24, 3] 鐨勭被鍒储寮曞紶閲忋€?
        璇勫垎绠€鍖栦负鐐瑰嚮娆℃暟锛堥娆★級锛屼笉鍦?forward 涓仛缁熻銆?
        """
        top3 = np.zeros((self.num_users, self.time_bins, self.topk_categories), dtype=np.int64)

        train_users = self.dataset.train_users
        train_items = self.dataset.train_items
        train_thetas = self.dataset.train_thetas
        item2cat = self.dataset.item2category

        hour_bins = np.floor((train_thetas / (2.0 * math.pi)) * self.time_bins).astype(np.int64)
        hour_bins = np.clip(hour_bins, 0, self.time_bins - 1)

        # key: (user_id, hour_bin) -> {category_id: click_count}
        counts = {}
        for u, i, h in zip(train_users, train_items, hour_bins):
            c = int(item2cat[i])
            key = (int(u), int(h))
            if key not in counts:
                counts[key] = {}
            counts[key][c] = counts[key].get(c, 0) + 1

        for (u, h), cat_counter in counts.items():
            ranked = sorted(cat_counter.items(), key=lambda x: (-x[1], x[0]))
            best_cats = [cid for cid, _ in ranked[:self.topk_categories]]
            for idx, cid in enumerate(best_cats):
                top3[u, h, idx] = cid

        return torch.from_numpy(top3).long()

    def computer(self):
        # 绋€鐤忕煩闃典箻娉曚笉鏀寔 FP16锛屽己鍒跺湪 float32 涓嬫墽琛?
        with torch.amp.autocast('cuda', enabled=False):
            users_emb = self.embedding_user.weight.float()
            items_emb = self.embedding_item.weight.float()
            all_emb = torch.cat([users_emb, items_emb], dim=0)

            embs = [all_emb]
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(self.Graph, all_emb)
                embs.append(all_emb)

            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items], dim=0)
        return users, items

    def _circular_hour_distance(self, cur_hours, hour_bins):
        """
        cur_hours: (B, 1)
        hour_bins: (1, 24)
        return: (B, 24)
        """
        diff = torch.abs(cur_hours - hour_bins)
        return torch.minimum(diff, 24.0 - diff)

    def _gaussian_hour_weights(self, thetas):
        """
        鏍规嵁璇锋眰鏃堕棿鐢熸垚 24 灏忔椂楂樻柉骞虫粦鏉冮噸銆?
        thetas: (B,)
        return: (B, 24)
        """
        # 鐩镐綅杞皬鏃讹紝鑼冨洿 [0, 24)
        cur_hours = (thetas / (2.0 * math.pi)) * 24.0
        cur_hours = cur_hours.view(-1, 1)
        hour_bins = self.hour_indices.view(1, -1)

        delta = self._circular_hour_distance(cur_hours, hour_bins)
        # g(delta) = exp(- ((delta - mu)^2) / (2 * sigma^2))
        weights = torch.exp(-((delta - self.clock_gaussian_mu).pow(2)) / (2.0 * (self.clock_gaussian_sigma ** 2)))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def _interest_clock_embedding(self, users, thetas):
        """
        users: (B,)
        thetas: (B,)
        return: v_clock (B, clock_emb_dim)
        """
        # [B,24,3]
        user_hour_top3 = self.user_hour_top3_categories[users]
        # [B,24,3,D]
        hour_cat_emb = self.category_embedding(user_hour_top3)
        # [B,24,D]锛屾瘡灏忔椂瀵?top3 鍋?mean-pooling
        hour_emb = hour_cat_emb.mean(dim=2)

        # [B,24]
        weights = self._gaussian_hour_weights(thetas)
        # [B,D]
        v_clock = torch.einsum('bh,bhd->bd', weights, hour_emb)
        return v_clock

    def _clock_score_for_items(self, v_clock, items):
        """
        v_clock: (B,D)
        items: (B,)
        return: (B,)
        """
        cat_ids = self.item_category_ids[items]
        target_cat_emb = self.category_embedding(cat_ids)
        return torch.mul(v_clock, target_cat_emb).sum(dim=1)

    def bpr_loss(self, users, pos, neg, thetas):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb_static = all_items[pos]
        neg_emb_static = all_items[neg]

        userEmb0 = self.embedding_user(users)
        posEmb0 = self.embedding_item(pos)
        negEmb0 = self.embedding_item(neg)

        base_pos_scores = torch.mul(users_emb, pos_emb_static).sum(dim=1)
        base_neg_scores = torch.mul(users_emb, neg_emb_static).sum(dim=1)

        v_clock = self._interest_clock_embedding(users, thetas)
        clock_pos_scores = self._clock_score_for_items(v_clock, pos)
        clock_neg_scores = self._clock_score_for_items(v_clock, neg)

        # score = dot(user, item) + alpha * dot(v_clock, v_target_category)
        pos_scores = base_pos_scores + self.clock_alpha * clock_pos_scores
        neg_scores = base_neg_scores + self.clock_alpha * clock_neg_scores

        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # Interest Clock 涓嶄娇鐢ㄨ仛绫荤喌椤?
        entropy_loss = torch.zeros((), device=users.device)

        reg_loss = (0.5 / float(len(users))) * (
            userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)
        )
        reg_loss += (0.5 / float(len(users))) * self.category_embedding.weight.norm(2).pow(2)
        return bpr_loss, reg_loss, entropy_loss

    def score_all_items(self, users, thetas, precomputed=None):
        """
        缁熶竴娴嬭瘯鎺ュ彛锛氳繑鍥?(B, m_items) 鐨勫叏閲忔墦鍒嗙煩闃点€?
        """
        if precomputed is None:
            all_users_static, all_items_static = self.computer()
        else:
            all_users_static, all_items_static = precomputed

        u_emb = all_users_static[users]

        base_scores = torch.matmul(u_emb, all_items_static.T)  # (B, M)

        v_clock = self._interest_clock_embedding(users, thetas)  # (B, D)
        all_target_cat_emb = self.category_embedding(self.item_category_ids)  # (M, D)
        clock_scores = torch.matmul(v_clock, all_target_cat_emb.T)  # (B, M)

        ratings = base_scores + self.clock_alpha * clock_scores
        return ratings
