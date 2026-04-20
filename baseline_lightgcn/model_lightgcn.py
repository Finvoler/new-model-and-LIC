import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCNBaseline(nn.Module):
    """
    浠呬繚鐣?LightGCN 涓诲共锛岀敤浜庝笌鏃堕棿寤烘ā鏂规硶鍋氫弗鏍煎弬鏁板榻愮殑鍩虹嚎銆?    鎺ュ彛涓庡叾浠栨ā鍨嬩繚鎸佷竴鑷淬€?    """

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

    def bpr_loss(self, users, pos, neg, thetas):
        del thetas
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos]
        neg_emb = all_items[neg]

        user_emb_0 = self.embedding_user(users)
        pos_emb_0 = self.embedding_item(pos)
        neg_emb_0 = self.embedding_item(neg)

        pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        reg_loss = (0.5 / float(len(users))) * (
            user_emb_0.norm(2).pow(2) + pos_emb_0.norm(2).pow(2) + neg_emb_0.norm(2).pow(2)
        )
        entropy_loss = torch.zeros((), device=users.device)
        return bpr_loss, reg_loss, entropy_loss

    def score_all_items(self, users, thetas, precomputed=None):
        del thetas
        if precomputed is None:
            all_users, all_items = self.computer()
        else:
            all_users, all_items = precomputed
        return torch.matmul(all_users[users], all_items.T)
