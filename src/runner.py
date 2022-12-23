import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from einops import repeat, rearrange
from scipy.signal import butter, lfilter, freqz
import os
import numpy as np
import random
import math
from tqdm import tqdm, trange
from collections import OrderedDict
from scipy import signal
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.metrics import PrecisionRecallDisplay, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('save/tblog')

import logging
logging.basicConfig(format='> %(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRunner(object):

    def __init__(self, traindl, validdl, testdl, model, optimizer, lossf, args):

        self.traindl = traindl
        self.validdl = validdl
        self.testdl = testdl
        self.model = model
        self.optimizer = optimizer
        self.lossf = lossf
        self.args = args
        self.epo_metrics = OrderedDict()

    def train(self):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, trained_model, testdl, args):
        raise NotImplementedError('Please override in child class')

    def keepmoving(self):
        raise NotImplementedError('Do not override in child class')

class Runner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(Runner, self).__init__(*args, **kwargs)

    def threshold_and_predict(self, scores, y_ture, thretype, point_adjust=False, composite_best_f1=False):
        """
        https://github.com/astha-chem/mvts-ano-eval/blob/main/src/evaluation/evaluation_utils.py
        :param scores:
        :param y_ture:
        :param thretype:
        :param point_adjust:
        :param composite_best_f1:
        :return:
        """
        score_t_test = scores.cpu().numpy()
        y_test = y_ture.cpu().numpy()
        true_events = get_events(y_test)
        if thretype == "fixed_thre":
            opt_thres = 0.000
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
        elif thretype == "best_f1_test" and point_adjust:
            prec, rec, thresholds = precision_recall_curve(y_test, score_t_test, pos_label=1)
            fscore_best_time = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
            opt_num = np.squeeze(np.argmax(fscore_best_time))
            opt_thres = thresholds[opt_num]
            thresholds = np.random.choice(thresholds, size=5000) + [opt_thres]
            fscores = []
            for thres in thresholds:
                _, _, _, _, _, fscore = get_point_adjust_scores(y_test, score_t_test > thres, true_events)
                fscores.append(fscore)
            opt_thres = thresholds[np.argmax(fscores)]
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
        elif thretype == "best_f1_test" and composite_best_f1:
            prec, rec, thresholds = precision_recall_curve(y_test, score_t_test, pos_label=1)
            precs_t = prec
            fscores_c = [get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t) for thres, prec_t in
                         zip(thresholds, precs_t)]
            try:
                opt_thres = thresholds[np.nanargmax(fscores_c)]
            except:
                opt_thres = 0.0
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
        elif thretype == "best_f1_test":
            prec, rec, thres = precision_recall_curve(y_test, score_t_test, pos_label=1)
            fscore = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
            opt_num = np.squeeze(np.argmax(fscore))
            opt_thres = thres[opt_num]
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
            # # Output Pre-Rec-Curve
            # disp = PrecisionRecallDisplay(precision=prec, recall=rec)
            # disp.plot()
            # plt.savefig("save/metrics/precision-recall-curve-{}.png".format(self.args.dataname))
            # # Output ROC-Curve
            # fpr, tpr, roc_thres = roc_curve(y_ture, scores, pos_label=1)
            # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
            # roc_display.plot()
            # plt.savefig("save/metrics/roc-curve-{}.png".format(self.args.dataname))
            # pred_labels = np.where(scores > opt_thres, 1, 0)
        elif thretype == "top_k_time":
            test_anom_frac = 0.1
            opt_thres = np.nanpercentile(score_t_test, 100 * (1 - test_anom_frac), interpolation='higher')
            pred_labels = np.where(score_t_test > opt_thres, 1, 0)
        return pred_labels, opt_thres

    def train(self):
        print("\n")
        print(">>>>>>> Start Train <<<<<<<")
        self.model.train()
        for epo in range(self.args.epoch):
            epo_sumloss = 0
            tb_loss = 0
            for i, batch in enumerate(tqdm(self.traindl)):
                self.optimizer.zero_grad()
                x, y = batch[0].to(self.args.device), batch[1].to(self.args.device)

                # add noise into x
                # x_noise = x + torch.normal(mean=0.0, std=0.1, size=x.shape).to(self.args.device)
                # print("x_noise", x_noise.shape)

                out = self.model(x)
                x_raw, z, encoder_out, REC_list, Z_list, P_list = out[0], out[1], out[2], out[3], out[4], out[5]

                # compute recloss
                rec_loss = self.lossf.forward_reconstructloss_kurtosis(x_raw, REC_list, type="last", kurtosis=False)

                """RUN"""
                # contrast_loss = self.lossf.forward_simsiam_point(Z_list, P_list)
                contrast_loss = self.lossf.forward_simsiam_extend(Z_list, P_list)
                # compute contrastive loss
                # z1, z2, p1, p2 = Z_list[0], Z_list[-1], P_list[0].detach(), P_list[-1].detach()
                # contrast_loss = self.lossf.forward_simsiamloss(z1, z2, p1, p2)

                # backward
                batch_loss = rec_loss + contrast_loss
                batch_loss.backward()
                # batch_loss.backward(retain_graph=True)
                self.optimizer.step()
                addloss = rec_loss - contrast_loss
                addloss.backward()
                self.optimizer.step()

                epo_sumloss += batch_loss.item()

            if i == 0:
                i = i+1
            epo_meanloss = epo_sumloss / i
            logger.info('Training Epoch - {} Summary: EpoMeanLoss={}'.format(epo, epo_meanloss))
            # writer.add_scalar("epo_meanloss", epo_meanloss, epo)

        return self.model

    def evaluate(self, trained_model, testdl, args):
        print("\n")
        print("=== Start Evaluating ===")
        testmode = "win"
        if testmode == "win":
            rec_scores, rec_data, sim_scores, gt = self.evaluate_simrec_window(trained_model, testdl, args)
        elif testmode == "point":
            rec_scores, rec_data, sim_scores, gt = self.evaluate_simrec_point(trained_model, testdl, args)
        gtnp = gt.cpu().numpy()
        if self.args.evalmethod == "recsim":
            anomalyscore = rec_scores * sim_scores
        elif self.args.evalmethod == "sim":
            anomalyscore = sim_scores
        elif self.args.evalmethod == "rec":
            anomalyscore = rec_scores
        # Pred and find threshold
        raw_pred, opt_thre = self.threshold_and_predict(anomalyscore, gt, thretype="best_f1_test", point_adjust=True, composite_best_f1=False)
        adjusted_pred = adjust_prediction(gtnp, raw_pred)
        self.show_results(gtnp, adjusted_pred)
        print("Best Threshold:", opt_thre)
        ## save
        # np.save("save/results/anomalyscore-{}.npy".format(args.dataname), anomalyscore.cpu().numpy())
        # np.save("save/results/anomalyscore-rec-{}.npy".format(args.dataname), rec_scores.cpu().numpy())
        # np.save("save/results/anomalyscore-sim-{}.npy".format(args.dataname), sim_scores.cpu().numpy())
        # np.save("save/results/gt-{}.npy".format(args.dataname), gtnp)
        # np.save("save/results/pred-{}.npy".format(args.dataname), adjusted_pred)
        # np.save("save/results/threshold-{}.npy".format(args.dataname), opt_thre)
        # recdata = torch.cat(rec_data, dim=0)
        # np.save("save/results/recdata-{}.npy".format(args.dataname), recdata.cpu().numpy())

    def evaluate_simrec_window(self, trained_model, testdl, args):
        def compute_rec_score_win(raw_x, rec_list, type="last", kurtosis=False):
            B, L, D = raw_x.shape

            def calculate_kur(x):
                meanv = torch.mean(x, dim=1).unsqueeze(dim=1)
                meanv = repeat(meanv, "B L D -> B (repeat L) D", repeat=L)
                m4 = torch.mean((x - meanv) ** 4, dim=1)
                m2 = torch.mean((x - meanv) ** 2, dim=1) + 0.0001
                kur = m4 / (m2 ** 2)
                kur = kur.unsqueeze(dim=1)
                kur = repeat(kur, "B L D -> B (repeat L) D", repeat=L)
                return kur

            if type == "last":
                rec_x = rec_list[-1]
                if kurtosis:
                    kurv = calculate_kur(raw_x)
                    rec_error_dim = (raw_x - rec_x) ** 2 * kurv
                    rec_score = torch.mean(rec_error_dim, dim=-1)
                else:
                    rec_error_dim = (raw_x - rec_x) ** 2
                    rec_score = torch.mean(rec_error_dim, dim=-1)
            elif type == "sum":
                N = len(rec_list)
                rec_x = 0
                for k in range(N):
                    rec_x = rec_x + rec_list[k]
                rec_error_dim = (raw_x - rec_x) ** 2
                rec_score = torch.mean(rec_error_dim, dim=-1)
            return rec_score, rec_x, rec_error_dim

        def compute_sim_score_window(Z_list, P_list):
            step = 6
            device = Z_list[0].device
            crit = nn.CosineSimilarity(dim=-1).to(device)
            # print("compute_sim_score_N>>>>>", len(Z_list))
            # print("compute_sim_score_N>>>>>", Z_list[0].shape)
            score = torch.zeros(Z_list[0].shape[0], Z_list[0].shape[1]).to(device)
            bench = torch.ones(Z_list[0].shape[0], Z_list[0].shape[1]).to(device)
            for i in range(len(Z_list)):
                z = Z_list[i]
                p = Z_list
                z = z.unsqueeze(0)
                z = repeat(z, "N B L D -> (repeat N) B L D", repeat=len(p))
                p = torch.stack(p, dim=0)
                p = torch.cat((p[:, :, -step:, :], p[:, :, :-step, :]), dim=2)  # skip-step
                s_ = torch.mean(bench - crit(z, p), dim=0)
                score = score + s_
            score = score / 3
            return score

        trained_model.eval()
        with torch.no_grad():
            gt = []
            rec_scores = []
            rec_data = []
            sim_scores = []
            for k, batch in enumerate(tqdm(testdl)):
                x, y = batch[0].to(args.device), batch[1].to(args.device)
                gt.append(torch.flatten(y))
                out = trained_model(x)
                # Rec Score
                x_raw, z, encoder_out, REC_list, Z_list, P_list = out[0], out[1], out[2], out[3], out[4], out[5]
                rec_score, rec_x, rec_score_dim = compute_rec_score_win(x_raw, REC_list, type="last", kurtosis=False)
                rec_data.append(rec_x.reshape(-1, rec_x.shape[-1]))

                rec_scores.append(torch.flatten(rec_score))
                # Sim Score
                # z1, z2, p1, p2 = Z_list[0], Z_list[-1], P_list[0].detach(), P_list[-1].detach()
                # sim_score = compute_sim_score(z1, z2)
                sim_score = compute_sim_score_window(Z_list, P_list)

                sim_scores.append(torch.flatten(sim_score))

            gt = torch.cat(gt, dim=0)
            rec_scores = torch.cat(rec_scores, dim=0)
            sim_scores = torch.cat(sim_scores, dim=0)

        return rec_scores, rec_data, sim_scores, gt

    def evaluate_simrec_point(self, trained_model, testdl, args):

        def compute_rec_score_point(raw_x, rec_list):
            rec_x = rec_list[-1]
            rec_error_dim = (raw_x - rec_x) ** 2
            rec_score = torch.mean(rec_error_dim, dim=-1)
            score = rec_score[:, -1]
            rec_x = rec_x[:, -1, :]
            rec_error_dim = rec_error_dim[:, -1, :]
            return score, rec_x, rec_error_dim

        def compute_sim_score_point(Z_list, P_list):
            step = 8
            device = Z_list[0].device
            crit = nn.CosineSimilarity(dim=-1).to(device)

            N = len(Z_list)
            assert len(Z_list) == len(P_list)
            z = torch.stack(Z_list, dim=2)[:, -1, :, :] # slect last time step
            p = torch.stack(P_list, dim=2)
            print(z.shape, p.shape)
            hh

            p = torch.flip(torch.flip(p, dims=[2])[:, step::step, :, :], dims=[2])
            score = torch.zeros(Z_list[0].shape[0]).to(device)
            for k in range(N):
                z_ = z[:, k, :].unsqueeze(dim=1)
                p_ = p[:, :, k, :]
                z_ = repeat(z_, "B L D -> B (repeat L) D", repeat=p_.shape[1])
                # print("++++++", z_.shape, p_.shape)
                score = score + (torch.mean(1-crit(z_, p_), dim=-1))
            score = score / N
            return score

        trained_model.eval()
        with torch.no_grad():
            gt = []
            rec_scores = []
            rec_data = []
            sim_scores = []
            for k, batch in enumerate(tqdm(testdl)):
                x, y = batch[0].to(args.device), batch[1].to(args.device)
                y_ = y[:, -1]
                gt.append(y_)
                # print("y_", y_.shape)
                out = trained_model(x)
                # Rec Score
                x_raw, z, encoder_out, REC_list, Z_list, P_list = out[0], out[1], out[2], out[3], out[4], out[5]
                rec_score, rec_x, rec_score_dim = compute_rec_score_point(x_raw, REC_list)
                rec_data.append(rec_x)
                rec_scores.append(rec_score)
                # print("rec_x/rec_score", rec_x.shape, rec_score.shape)
                # Sim Score
                sim_score = compute_sim_score_point(Z_list, P_list)
                sim_scores.append(sim_score)
                # print("sim_score", sim_score.shape)

            gt = torch.cat(gt, dim=0)
            rec_scores = torch.cat(rec_scores, dim=0)
            sim_scores = torch.cat(sim_scores, dim=0)

        return rec_scores, rec_data, sim_scores, gt

    def show_results(self, gt, pred):
        # print("Precision/Recall/F1")
        p = precision_score(gt, pred)
        r = recall_score(gt, pred)
        f1 = f1_score(gt, pred)
        print("\n>>>>>>>> Precision={}, Recall={}, F1={} <<<<<<<<\n".format(p, r, f1))



def compute_sim_score(short_rep, long_rep):
    sim = F.cosine_similarity(short_rep, long_rep, dim=-1)
    S = torch.ones_like(sim)
    score = S - sim
    return score

















def get_events(y_test, outlier=1, normal=0, breaks=[]):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
                # A break point was hit, end current event and start new one
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim

        else:
            # event_by_time_true[tim] = 0
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events


def get_f_score(prec, rec):
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec)
    return f_score


def get_point_adjust_scores(y_test, pred_labels, true_events):
    tp = 0
    fn = 0
    for true_event in true_events.keys():
        true_start, true_end = true_events[true_event]
        if pred_labels[true_start:true_end].sum() > 0:
            tp += (true_end - true_start)
        else:
            fn += (true_end - true_start)
    fp = np.sum(pred_labels) - np.sum(pred_labels * y_test)

    prec, rec, fscore = get_prec_rec_fscore(tp, fp, fn)
    return fp, fn, tp, prec, rec, fscore

def get_prec_rec_fscore(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
    fscore = get_f_score(precision, recall)
    return precision, recall, fscore

def get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t, return_prec_rec=False):
    pred_labels = score_t_test > thres
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

def adjust_prediction(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    return pred


"""================================================================================================================"""
"""Drop"""

def compute_simgrad_score(embed, encoder_out):
    B, L, D = embed.shape
    RP = len(encoder_out)
    embed = repeat(embed, "b l d -> (repeat b) l d", repeat=RP)
    encoder_out = torch.cat(encoder_out, dim=0)

    sim = F.cosine_similarity(embed, encoder_out, dim=-1)
    S = torch.ones_like(sim)
    score = S - sim

    score = score.reshape(-1,B,L)
    score = score.reshape(score.shape[0], -1)
    score = torch.mean(score, dim=0)
    score = score.reshape(B,L)
    return score

def evaluate_use_rec(self, trained_model, testdl, args):
    trained_model.eval()
    with torch.no_grad():
        gt = []
        anomalyscore = []
        rec = []
        for k, batch in enumerate(tqdm(testdl)):
            x, y = batch[0].to(args.device), batch[1].to(args.device)
            out = trained_model(x)
            # Extract Representations
            x_raw, z, encoder_out, REC_list, Z_list, P_list = out[0], out[1], out[2], out[3], out[4], out[5]

            scores, rec_x, rec_error = compute_reconstruct_score(x_raw, REC_list, type="last", kurtosis=False)

            rec.append(rec_x.reshape(-1, rec_x.shape[-1]))
            gt.append(torch.flatten(y))
            anomalyscore.append(torch.flatten(scores))

        gt = torch.cat(gt, dim=0)
        anomalyscore = torch.cat(anomalyscore, dim=0)
    return gt, anomalyscore, rec

def evaluate_use_sim(self, trained_model, testdl, args):
    trained_model.eval()
    with torch.no_grad():
        gt = []
        anomalyscore = []
        for k, batch in enumerate(tqdm(testdl)):
            x, y = batch[0].to(args.device), batch[1].to(args.device)
            out = trained_model(x)
            # Extract Representations
            x_raw, z, encoder_out, REC_list, Z_list, P_list = out[0], out[1], out[2], out[3], out[4], out[5]
            z1, z2, p1, p2 = Z_list[0], Z_list[-1], P_list[0].detach(), P_list[-1].detach()

            scores = compute_sim_score(z1, z2)

            y = torch.flatten(y)
            scores = torch.flatten(scores)
            gt.append(y)
            anomalyscore.append(scores)

        gt = torch.cat(gt, dim=0)
        anomalyscore = torch.cat(anomalyscore, dim=0)
    return gt, anomalyscore

# std = torch.std(x, dim=1)
# corr_one = torch.ones_like(std)
# corr_zero = torch.zeros_like(std)
# corr = torch.where(std > 0, corr_one, corr_zero)