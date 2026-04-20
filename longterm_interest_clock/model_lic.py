"""
Long-term Interest Clock (LIC) on top of LightGCN.

璁烘枃: Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System

鏈粨搴撶殑瀹為獙璁惧畾涓庡師璁烘枃涓嶅悓锛?
1. 浜や簰琚寜鐢ㄦ埛鍐呴殢鏈烘墦涔憋紝鐢ㄤ簬鐮旂┒澶╁唴鏃堕棿瑙勫緥锛岃€屼笉鏄祦寮忛『搴忛娴嬶紱
2. 绂荤嚎 BPR 璁粌涓笉瀛樺湪棰濆鐨?request/query 鐗瑰緛锛屽彧鏈?user / item / theta銆?

鍥犳杩欓噷涓嶈兘鎶婄洰鏍?item 鐩存帴褰撲綔璁烘枃涓殑 query锛屽惁鍒欎細鎶婃爣绛炬硠婕忓埌
Clock-GSU / Clock-ESU 涓紝璁粌鏃惰〃鐜颁负蹇€熻繃鎷熷悎銆佹祴璇曟寚鏍囧闄枫€?

鏈疄鐜颁繚鐣?LIC 鐨勪袱涓牳蹇冩ā鍧楋細
    - Clock-GSU: 浠庣敤鎴烽暱鏈熻涓烘睜涓绱⑩€滀笌褰撳墠鐢ㄦ埛鐘舵€佺浉鍏充笖鏃堕棿鎺ヨ繎鈥濈殑 Top-K 琛屼负
    - Clock-ESU: 浠ュ澶存椂闂存劅鐭ユ敞鎰忓姏鎻愬彇褰撳墠鏃跺埢鐨勭敤鎴峰叴瓒ｈ〃绀?v_cur

閫傞厤鍚庣殑鏈€缁堟墦鍒嗭細
    score(u, i, theta) = <u_static, i_static> + alpha * <v_cur(u, theta), i_static>
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeGapEncoder(nn.Module):
    """
    璁烘枃鍏紡 s(螖): 灏嗘椂闂村樊 螖 缂栫爜涓烘爣閲忓亸缃緱鍒?
    杈撳叆鐗瑰緛: [螖, 鈭毼? 螖虏, log(螖+1)]
    涓ゅ眰 MLP: 4 -> 8 -> 1
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta: (...) 浠绘剰褰㈢姸鐨勬椂闂村樊 (灏忔椂鍒? [0, 12])
        return: (...) 涓?delta 鍚屽舰鐘剁殑鏍囬噺鍋忕疆
        """
        delta = delta.clamp(min=0.0)
        features = torch.stack(
            [delta, delta.sqrt(), delta.pow(2), torch.log(delta + 1.0)], dim=-1
        )  # (..., 4)
        return self.net(features).squeeze(-1)  # (...)


class ClockGSU(nn.Module):
    """
    Clock-based General Search Unit.
    浠庣敤鎴峰叏灞€琛屼负闆嗗悎涓寜 (涓婁笅鏂囩浉浼煎害 + 鏃堕棿宸瘎鍒? 妫€绱?Top-K 瀛愬簭鍒椼€?

    璁烘枃鍏紡 (2):
      伪(b_m, q) = (W_b路b_m 鈯?W_q路q)^T / 鈭歞 + s(螖(t^{b_m}, t^{cur}))

    娉ㄦ剰: 璁烘枃鍘熸枃浣跨敤閫愬厓绱犵偣绉啀姹傚拰(绛変环浜庡唴绉?, 姝ゅ瀹炵幇涓哄悜閲忓唴绉€?
    """

    def __init__(self, behavior_dim: int, query_dim: int, d: int):
        super().__init__()
        self.d = d
        self.W_b = nn.Linear(behavior_dim, d, bias=False)
        self.W_q = nn.Linear(query_dim, d, bias=False)
        self.time_encoder = TimeGapEncoder()

    def forward(
        self,
        behaviors: torch.Tensor,
        query: torch.Tensor,
        behavior_thetas: torch.Tensor,
        cur_theta: torch.Tensor,
        top_k: int,
        behavior_mask: torch.Tensor | None = None,
    ):
        """
        behaviors:       (B, M, L)  鏌愮敤鎴风殑鍏ㄥ眬琛屼负宓屽叆
        query:           (B, H)     褰撳墠涓婁笅鏂囧悜閲忥紙姝ゅ涓虹敤鎴烽潤鎬佸祵鍏ワ級
        behavior_thetas: (B, M)     姣忔潯琛屼负瀵瑰簲鐨?theta
        cur_theta:       (B,)       褰撳墠璇锋眰鏃堕棿 theta
        top_k:           int        妫€绱㈡暟閲?K

        return:
          top_behaviors: (B, K, L)
          top_thetas:    (B, K)
          top_scores:    (B, K)  鐢ㄤ簬 ESU 鐨勭浉鍏冲害鍒嗘暟
        """
        B, M, L = behaviors.shape

        # 涓婁笅鏂囩浉浼煎害: (W_b 路 b) * (W_q 路 q) 姹傚拰 / 鈭歞
        proj_b = self.W_b(behaviors)  # (B, M, d)
        proj_q = self.W_q(query).unsqueeze(1)  # (B, 1, d)
        item_sim = (proj_b * proj_q).sum(dim=-1) / math.sqrt(self.d)  # (B, M)

        # 鐩稿鏃堕棿宸?(寰幆璺濈, 杞负灏忔椂鍒?
        delta = self._circular_hour_delta(behavior_thetas, cur_theta)  # (B, M)
        time_score = self.time_encoder(delta)  # (B, M)

        alpha = item_sim + time_score  # (B, M)

        if behavior_mask is not None:
            neg_inf = torch.finfo(alpha.dtype).min
            alpha = alpha.masked_fill(~behavior_mask, neg_inf)

        # Top-K 妫€绱?
        actual_k = min(top_k, M)
        top_scores, top_idx = torch.topk(alpha, k=actual_k, dim=1)  # (B, K)

        # 鏀堕泦 Top-K 琛屼负涓庢椂闂?
        top_behaviors = torch.gather(
            behaviors, 1, top_idx.unsqueeze(-1).expand(-1, -1, L)
        )  # (B, K, L)
        top_thetas = torch.gather(behavior_thetas, 1, top_idx)  # (B, K)

        return top_behaviors, top_thetas, top_scores

    @staticmethod
    def _circular_hour_delta(thetas_a: torch.Tensor, thetas_b: torch.Tensor) -> torch.Tensor:
        """
        璁＄畻寰幆灏忔椂璺濈 (璁烘枃涓殑鐩稿鏃堕棿宸?
        thetas_a: (...) 寮у害鍒?[0, 2蟺)
        thetas_b: (...) 鎴?(B,) 寮у害鍒?
        return: (...) 灏忔椂鍒惰窛绂?[0, 12]
        """
        hours_a = thetas_a / (2.0 * math.pi) * 24.0
        if thetas_b.dim() < thetas_a.dim():
            thetas_b = thetas_b.unsqueeze(-1)
        hours_b = thetas_b / (2.0 * math.pi) * 24.0
        diff = torch.abs(hours_a - hours_b)
        return torch.minimum(diff, 24.0 - diff)


class ClockESU(nn.Module):
    """
    Clock-based Exact Search Unit.
    澶氬ご鏃堕棿宸劅鐭ユ敞鎰忓姏: 浠?Top-K 瀛愬簭鍒椾腑鎻愬彇褰撳墠鍏磋叮琛ㄥ緛 v_cur.

    璁烘枃鍏紡 (4):
      r_i = Softmax(伪_i)^T 路 Z 路 W_vi

    姣忎釜澶存湁鐙珛鐨?W_b, W_q, W_v 鍜?TimeGapEncoder.
    鏈€缁? v_cur = h([r_1, ..., r_n_heads])
    """

    def __init__(self, behavior_dim: int, query_dim: int, d: int, n_heads: int, output_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.d = d

        # 姣忎釜澶寸殑鍙傛暟
        self.W_bs = nn.ModuleList([nn.Linear(behavior_dim + 4, d, bias=False) for _ in range(n_heads)])
        self.W_qs = nn.ModuleList([nn.Linear(query_dim, d, bias=False) for _ in range(n_heads)])
        self.W_vs = nn.ModuleList([nn.Linear(behavior_dim + 4, d, bias=False) for _ in range(n_heads)])
        self.time_encoders = nn.ModuleList([TimeGapEncoder() for _ in range(n_heads)])

        # 鏈€缁堣瀺鍚堢綉缁?h(路)
        self.h = nn.Sequential(
            nn.Linear(n_heads * d, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        top_behaviors: torch.Tensor,
        query: torch.Tensor,
        top_thetas: torch.Tensor,
        cur_theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        top_behaviors: (B, K, L)
        query:         (B, H)
        top_thetas:    (B, K)
        cur_theta:     (B,)
        return: v_cur  (B, output_dim)
        """
        B, K, L = top_behaviors.shape

        # 鏃堕棿宸壒寰? [螖, 鈭毼? 螖虏, log(螖+1)]
        delta = ClockGSU._circular_hour_delta(top_thetas, cur_theta)  # (B, K)
        delta_c = delta.clamp(min=0.0)
        time_features = torch.stack(
            [delta_c, delta_c.sqrt(), delta_c.pow(2), torch.log(delta_c + 1.0)], dim=-1
        )  # (B, K, 4)

        # 灏嗘椂闂村樊鐗瑰緛鎷兼帴鍒拌涓哄祵鍏?(璁烘枃 Section 3.3: "concentrate [...] into representations")
        z_aug = torch.cat([top_behaviors, time_features], dim=-1)  # (B, K, L+4)

        heads = []
        for i in range(self.n_heads):
            proj_b = self.W_bs[i](z_aug)  # (B, K, d)
            proj_q = self.W_qs[i](query).unsqueeze(1)  # (B, 1, d)
            item_sim = (proj_b * proj_q).sum(dim=-1) / math.sqrt(self.d)  # (B, K)

            time_score = self.time_encoders[i](delta)  # (B, K)
            alpha = item_sim + time_score  # (B, K)

            attn_weights = F.softmax(alpha, dim=-1)  # (B, K)

            # r_i = Softmax(伪)^T 路 Z 路 W_vi
            z_proj = self.W_vs[i](z_aug)  # (B, K, d)
            r_i = torch.einsum('bk,bkd->bd', attn_weights, z_proj)  # (B, d)
            heads.append(r_i)

        # v_cur = h([r_1, ..., r_n_heads])
        v_cur = self.h(torch.cat(heads, dim=-1))  # (B, output_dim)
        return v_cur


class LongTermInterestClockLightGCN(nn.Module):
    """
    LightGCN backbone + Long-term Interest Clock (Clock-GSU + Clock-ESU).

    鎺ュ彛涓?TimeAwareLightGCN / GaussianClockLightGCN 瀹屽叏涓€鑷?
      - bpr_loss(users, pos, neg, thetas)
      - computer()
      - score_all_items(users, thetas, precomputed=None)
    """

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = config['device']

        # LightGCN backbone 鍙傛暟
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']

        # LIC 鍙傛暟
        self.top_k = config.get('lic_top_k', 100)
        self.n_heads = config.get('lic_n_heads', 4)
        self.lic_d = config.get('lic_d', 32)
        self.lic_alpha = config.get('lic_alpha', 1.0)
        self.Graph = dataset.getSparseGraph()
        self.__init_weight()

    def __init_weight(self):
        # 1) LightGCN 闈欐€佸祵鍏?
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # 2) Clock-GSU
        self.clock_gsu = ClockGSU(
            behavior_dim=self.latent_dim,
            query_dim=self.latent_dim,
            d=self.lic_d,
        )

        # 3) Clock-ESU
        self.clock_esu = ClockESU(
            behavior_dim=self.latent_dim,
            query_dim=self.latent_dim,
            d=self.lic_d,
            n_heads=self.n_heads,
            output_dim=self.latent_dim,
        )

        # 4) 棰勬瀯寤虹敤鎴疯涓虹储寮?(鏃犲簭鍏ㄥ眬琛屼负闆嗗悎)
        self._build_user_behavior_index()

    def _build_user_behavior_index(self):
        """
        绂荤嚎鏋勫缓姣忎釜鐢ㄦ埛鐨勫叏閮ㄨ缁冭涓?(鐗╁搧ID, theta) 瀵广€?
        瀛樺偍涓?padded tensor 鏂逛究 batch 妫€绱€傛暟鎹凡 shuffle锛屾澶勪粎鍋氶泦鍚堝瓨鍌ㄣ€?
        """
        train_users = self.dataset.train_users
        train_items = self.dataset.train_items
        train_thetas = self.dataset.train_thetas

        # 缁熻姣忎釜鐢ㄦ埛鏈€澶ц涓烘暟锛岀敤浜?padding
        from collections import Counter
        user_counts = Counter(train_users.tolist())
        max_behaviors = min(max(user_counts.values()) if user_counts else 0, 
                            self.config.get('lic_max_behaviors', 300))
        # 闄愬埗鏈€澶ц涓哄簭鍒楅暱搴︼細瑕嗙洊 99%+ 鐢ㄦ埛锛屾帶鍒舵瘡 batch 鐨?GPU 鍐呭瓨鍗犵敤
        self.max_behaviors = max_behaviors

        # 鏋勫缓 padded 鏁扮粍
        user_items = np.zeros((self.num_users, max_behaviors), dtype=np.int64)
        user_thetas = np.zeros((self.num_users, max_behaviors), dtype=np.float32)
        user_lengths = np.zeros(self.num_users, dtype=np.int64)

        # 閫愮敤鎴峰～鍏?
        user_positions = {}
        for u, item, theta in zip(train_users, train_items, train_thetas):
            u = int(u)
            pos = user_positions.get(u, 0)
            if pos < max_behaviors:
                user_items[u, pos] = item
                user_thetas[u, pos] = theta
                user_positions[u] = pos + 1

        for u, pos in user_positions.items():
            user_lengths[u] = pos

        self.register_buffer('user_behavior_lengths', torch.from_numpy(user_lengths).long())
        # 琛屼负鐗╁搧鍜屾椂闂翠繚鐣欏湪 CPU 鍐呭瓨涓紝閬垮厤澶у紶閲忓父椹绘樉瀛?
        self.user_behavior_items_cpu = torch.from_numpy(user_items).long()
        self.user_behavior_thetas_cpu = torch.from_numpy(user_thetas).float()

    def computer(self):
        """鏍囧噯 LightGCN 闈欐€佸墠鍚戜紶鎾?""
        # 绋€鐤忕煩闃典箻娉曚笉鏀寔 FP16锛屽己鍒跺湪 float32 涓嬫墽琛?
        with torch.amp.autocast('cuda', enabled=False):
            users_emb = self.embedding_user.weight.float()
            items_emb = self.embedding_item.weight.float()
            all_emb = torch.cat([users_emb, items_emb])

            embs = [all_emb]
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(self.Graph, all_emb)
                embs.append(all_emb)

            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def _get_user_behaviors(self, users, all_items):
        """
        鑾峰彇涓€涓?batch 鐢ㄦ埛鐨勮涓哄祵鍏ュ拰鏃堕棿銆?
        users: (B,)
        all_items: (M, d)
        return:
                    behavior_item_ids: (B, max_M)
          behaviors: (B, max_M, d)  鈥?padded, 鏃犳晥浣嶇疆涓?0
          thetas:    (B, max_M)
          masks:     (B, max_M) bool, True 涓烘湁鏁堜綅缃?
        """
        users_cpu = users.cpu()
        # 鍦?CPU 涓婂垏鐗囷紝鍐嶆惉鍒?GPU锛岄伩鍏嶅叏閲忚涓烘暟鎹父椹绘樉瀛?
        behavior_item_ids = self.user_behavior_items_cpu[users_cpu].to(self.device)  # (B, max_M)
        behavior_thetas = self.user_behavior_thetas_cpu[users_cpu].to(self.device)   # (B, max_M)
        lengths = self.user_behavior_lengths[users]          # (B,)

        # 宓屽叆鏌ヨ〃
        behaviors = all_items[behavior_item_ids]  # (B, max_M, d)

        # 鏋勫缓 mask
        B = users.size(0)
        max_M = behavior_item_ids.size(1)
        range_idx = torch.arange(max_M, device=users.device).unsqueeze(0)  # (1, max_M)
        masks = range_idx < lengths.unsqueeze(1)  # (B, max_M)

        # 瀵规棤鏁堜綅缃竻闆?
        behaviors = behaviors * masks.unsqueeze(-1).float()
        behavior_thetas = behavior_thetas * masks.float()

        return behavior_item_ids, behaviors, behavior_thetas, masks

    def _prepare_behavior_pool(self, users, all_items, candidate_ids=None):
        behavior_item_ids, behaviors, behavior_thetas, masks = self._get_user_behaviors(users, all_items)

        if candidate_ids is not None:
            masks = masks & (behavior_item_ids != candidate_ids.unsqueeze(1))

        empty_mask = masks.sum(dim=1) == 0
        if empty_mask.any():
            behaviors = behaviors.clone()
            behavior_thetas = behavior_thetas.clone()
            masks = masks.clone()
            behaviors[empty_mask, 0] = 0.0
            behavior_thetas[empty_mask, 0] = 0.0
            masks[empty_mask, 0] = True

        return behaviors, behavior_thetas, masks, empty_mask

    def _lic_embedding(self, users, query_emb, thetas, all_items, exclude_item_ids=None):
        """
        璁＄畻 LIC 褰撳墠鍏磋叮宓屽叆 v_cur銆?
        users:         (B,)
        query_emb:     (B, d)  褰撳墠涓婁笅鏂囧悜閲忥紙姝ゅ浣跨敤鐢ㄦ埛闈欐€佸祵鍏ワ級
        thetas:        (B,)    褰撳墠鏃堕棿
        all_items:     (M, d)
        return: v_cur  (B, d)
        """
        behaviors, behavior_thetas, masks, empty_mask = self._prepare_behavior_pool(
            users, all_items, candidate_ids=exclude_item_ids
        )

        top_behaviors, top_thetas, _ = self.clock_gsu(
            behaviors,
            query_emb,
            behavior_thetas,
            thetas,
            self.top_k,
            behavior_mask=masks,
        )

        # Clock-ESU
        v_cur = self.clock_esu(top_behaviors, query_emb, top_thetas, thetas)
        if empty_mask.any():
            v_cur = v_cur.clone()
            v_cur[empty_mask] = 0.0
        return v_cur

    def bpr_loss(self, users, pos, neg, thetas):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb_static = all_items[pos]
        neg_emb_static = all_items[neg]

        userEmb0 = self.embedding_user(users)
        posEmb0 = self.embedding_item(pos)
        negEmb0 = self.embedding_item(neg)

        # 鍩虹 CF 寰楀垎
        base_pos_scores = torch.mul(users_emb, pos_emb_static).sum(dim=1)
        base_neg_scores = torch.mul(users_emb, neg_emb_static).sum(dim=1)

        # 鍦ㄥ綋鍓嶅疄楠岃瀹氫笅锛孡IC 搴旂敓鎴愮粺涓€鐨?user-time 鍏磋叮琛ㄥ緛锛?
        # 涓嶈兘灏嗙洰鏍?item 鐩存帴浣滀负 query 浣跨敤銆?
        v_cur = self._lic_embedding(users, users_emb, thetas, all_items, exclude_item_ids=pos)

        lic_pos_scores = torch.mul(v_cur, pos_emb_static).sum(dim=1)
        lic_neg_scores = torch.mul(v_cur, neg_emb_static).sum(dim=1)

        pos_scores = base_pos_scores + self.lic_alpha * lic_pos_scores
        neg_scores = base_neg_scores + self.lic_alpha * lic_neg_scores

        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # LIC 涓嶄娇鐢ㄨ仛绫荤喌椤?
        entropy_loss = torch.zeros((), device=users.device)

        # L2 姝ｅ垯鍖?
        reg_loss = (0.5 / float(len(users))) * (
            userEmb0.norm(2).pow(2)
            + posEmb0.norm(2).pow(2)
            + negEmb0.norm(2).pow(2)
        )
        lic_param_norm = torch.zeros((), device=users.device)
        for parameter in self.clock_gsu.parameters():
            lic_param_norm = lic_param_norm + parameter.norm(2).pow(2)
        for parameter in self.clock_esu.parameters():
            lic_param_norm = lic_param_norm + parameter.norm(2).pow(2)
        reg_loss = reg_loss + (0.5 / float(len(users))) * lic_param_norm

        return bpr_loss, reg_loss, entropy_loss

    def score_all_items(self, users, thetas, precomputed=None):
        """
        缁熶竴娴嬭瘯鎺ュ彛: 杩斿洖 (B, m_items) 鐨勫叏閲忔墦鍒嗙煩闃点€?
        璇勪及闃舵鍙渶璁＄畻涓€娆?user-time 鍏磋叮琛ㄥ緛锛屽啀涓庡叏閲忕墿鍝佸仛鐐圭Н銆?
        """
        if precomputed is None:
            all_users_static, all_items_static = self.computer()
        else:
            all_users_static, all_items_static = precomputed

        u_emb = all_users_static[users]  # (B, d)
        v_cur = self._lic_embedding(users, u_emb, thetas, all_items_static)

        base_scores = torch.matmul(u_emb, all_items_static.T)
        lic_scores = torch.matmul(v_cur, all_items_static.T)
        return base_scores + self.lic_alpha * lic_scores
