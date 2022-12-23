
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from dtaidistance import dtw
from tslearn.clustering import TimeSeriesKMeans
from tslearn.neighbors import KNeighborsTimeSeries
from einops import repeat


# def forward_infonceloss(self, query, keys, temperature=0.07):
#     def transpose(x):
#         return x.transpose(-2, -1)
#
#     def normalize(*xs):
#         return [None if x is None else F.normalize(x, dim=-1) for x in xs]
#
#     positive_key = keys
#     query, positive_key = normalize(query, positive_key)
#     logits = query @ transpose(positive_key)
#     # Positive keys are the entries on the diagonal
#     labels = torch.arange(len(query), device=query.device)
#     loss = F.cross_entropy(logits / temperature, labels, reduction='mean')
#     return loss

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        # self.losstype = losstype

        self.args = args
        P = "LOSS"

    def forward_reconstructloss_kurtosis(self, x, pred_list, type="last", kurtosis=False):
        B, L, D = x.shape
        def calculate_kur(x):
            meanv = torch.mean(x, dim=1).unsqueeze(dim=1)
            m4 = torch.mean((x - meanv) ** 4, dim=1)
            m2 = torch.mean((x - meanv) ** 2, dim=1) + 0.0001
            kurv  =m4 / (m2**2)
            kurv = kurv.unsqueeze(dim=1)
            kurv = repeat(kurv, "B L D -> B (repeat L) D", repeat=L)
            return kurv
        if type == "last":
            pred = pred_list[-1]
            if kurtosis:
                kurv = calculate_kur(x)
                loss = (x - pred) ** 2 * kurv
            else:
                loss = (x - pred) ** 2
            loss = torch.mean(loss)
        elif type == "sum":
            N = len(pred_list)
            e = x
            loss = 0
            for k in range(N):
                rec = pred_list[N - 1 - k]
                l = (e - rec) ** 2
                loss = loss + torch.mean(l)
                e = e - rec
        return loss

    def forward_simsiam_point(self, Z_list, P_list):
        step = 8
        crit = nn.CosineSimilarity(dim=-1).to(self.args.device)
        N = len(Z_list)
        assert len(Z_list) == len(P_list)
        z = torch.stack(Z_list, dim=2)[:, -1, :, :]
        p = torch.stack(P_list, dim=2)
        p = torch.flip(torch.flip(p, dims=[2])[:,step::step,:,:], dims=[2])
        ext_loss = 0
        for k in range(N):
            z_ = z[:,k,:].unsqueeze(dim=1)
            p_ = p[:,:,k,:]
            z_ = repeat(z_, "B L D -> B (repeat L) D", repeat=p_.shape[1])
            ext_loss = ext_loss + -(crit(z_, p_).mean())
        ext_loss = ext_loss / len(Z_list)
        return ext_loss


    def forward_simsiam_extend(self, Z_list, P_list):
        step = 6
        crit = nn.CosineSimilarity(dim=-1).to(self.args.device)
        N = len(Z_list)
        ext_loss = 0
        for i in range(len(Z_list)):
            z = Z_list[i]
            p = P_list
            z = z.unsqueeze(0)
            z = repeat(z, "N B L D -> (repeat N) B L D", repeat=len(p))
            p = torch.stack(p, dim=0)
            p = torch.cat((p[:, :,-step:, :], p[:, :, :-step, :]), dim=2) # skip-step
            # z = repeat(z, "B L D -> (repeat B) L D", repeat=len(p))
            # p = torch.cat(p, dim=0).detach()
            # p = p[:, torch.randperm(p.shape[1]), :] # random shuffle time-steps

            ext_loss = ext_loss + -(crit(z, p).mean())
        # ext_loss = ext_loss / len(Z_list)
        loss = ext_loss / len(Z_list)
        return loss

    def forward_simsiamloss(self, z1, z2, p1, p2):
        crit = nn.CosineSimilarity(dim=-1).to(self.args.device)
        loss = -(crit(z1, p2).mean() + crit(z2, p1).mean())*0.5
        return loss

    def forward_contrastiveloss(self, query, keys, temperature=0.07):
        def get_negative_mask(batch_size):
            negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
            for i in range(batch_size):
                negative_mask[i, i] = 0
                negative_mask[i, i + batch_size] = 0

            negative_mask = torch.cat((negative_mask, negative_mask), 0)
            return negative_mask

        device = self.args.device
        positive_key = keys
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        out_1 = query
        out_2 = positive_key
        batch_size = query.shape[0]
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        old_neg = neg.clone()
        mask = get_negative_mask(batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)
        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        # negative samples similarity scoring
        Ng = neg.sum(dim=-1)
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()
        return loss

    def forward_debiasedcontrastiveloss(self, query, keys, temperature=0.07, tau_plus=0.5):
        '''
        Debiased Contrastive Learning - NIPS2020
        https://github.com/chingyaoc/DCL/blob/master/main.py
        '''

        def get_negative_mask(batch_size):
            negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
            for i in range(batch_size):
                negative_mask[i, i] = 0
                negative_mask[i, i + batch_size] = 0

            negative_mask = torch.cat((negative_mask, negative_mask), 0)
            return negative_mask

        positive_key = keys
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)

        debiased = True
        out_1 = query
        out_2 = positive_key
        batch_size = query.shape[0]
        tau_plus = tau_plus
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim=-1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()
        return loss

    def forward_hardcontrastiveloss(self, query, keys, temperature=0.07, tau_plus=0.1, beta=1.0):
        '''
        Contrastive Learning with Hard Negative Samples - ICLR2021
        https://github.com/joshr17/HCL
        '''

        def get_negative_mask(batch_size):
            negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
            for i in range(batch_size):
                negative_mask[i, i] = 0
                negative_mask[i, i + batch_size] = 0

            negative_mask = torch.cat((negative_mask, negative_mask), 0)
            return negative_mask

        positive_key = keys
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        estimator = "hard"
        out_1 = query
        out_2 = positive_key
        tau_plus = tau_plus
        beta = beta
        batch_size = query.shape[0]
        device = self.args.device
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        old_neg = neg.clone()
        mask = get_negative_mask(batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
        elif estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()

        return loss

    def forward_supervised_contrastive(self, query, keys, labels, temperature=0.07, base_temperature=0.07, contrast_mode="all", mask=None):
        """

        :param query: [B * D]
        :param keys: [B * D]
        :param labels: [B]
        :param temperature:
        :param base_temperature:
        :param contrast_mode:
        :param mask:
        :return: loss
        """

        query = F.normalize(query, dim=-1)
        keys = F.normalize(keys, dim=-1)
        query = query.unsqueeze(dim=1)
        keys = keys.unsqueeze(dim=1)
        features = torch.cat((query, keys), dim=1)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss




if __name__ == "__main__":
    pass
    # query = torch.rand(100, 3)
    # keys = torch.rand(100, 3)
    # raw_ts = torch.rand(100, 50, 2) # B L D
    # feats = torch.rand(100, 5)
    #
    # LOSS = Loss()
    # lossv = LOSS.forward_posadd_dtwloss(query, keys, raw_ts)
    # print(lossv)