import os
import sys
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import tqdm
import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from options import Options
from src import dataloader, runner, model, tools, optimizer, loss
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score

tools.set_seed(77)
args = Options().parse()
traindl, trainds = dataloader.get_loader(dataname=args.dataname, batch_size=args.batch_size, win_size=args.win_size, step=args.win_step, mode='train')
validdl, validds = dataloader.get_loader(dataname=args.dataname, batch_size=args.batch_size, win_size=args.win_size, step=args.win_step, mode='valid')
testdl, testds = dataloader.get_loader(dataname=args.dataname, batch_size=args.batch_size, win_size=args.win_size, step=args.win_step, mode='test')
print("   @traindl / validdl / testdl: ", len(traindl), len(validdl), len(testdl))
#print("data sample", trainds[0][0].shape, testds[0][0].shape)

if not args.evalonly:
    """Load Model"""
    expmodel = model.ContrastiveLearningTSAD(args)
    expmodel = expmodel.to(args.device)
    print("   @Number of parameteres:", tools.count_parameters(expmodel))

    """Bulid Loss"""
    lossf = loss.Loss(args=args)

    """Load optimizer"""
    optim = optimizer.build_optimizer(model=expmodel, args=args)

    """Load runner"""

    exp = runner.Runner(traindl=traindl, validdl=validdl, testdl=testdl,model=expmodel, optimizer=optim, lossf=lossf, args=args)
    trained_model = exp.train()
    checkpoint = {
        'epoch': args.epoch,
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optim.state_dict()}
    cpsavepath = "save/model/trained_model_{}.pth".format(args.dataname)
    torch.save(checkpoint, cpsavepath)
    print("   @model is saved at: {}".format(cpsavepath))


"""Evaluating"""
pretrain_model = model.ContrastiveLearningTSAD(args)
pretrain_model = pretrain_model.to(args.device)
pretrain_model = pretrain_model.to(args.device)
cppath = "save/model/trained_model_{}.pth".format(args.dataname)
cp = torch.load(cppath)
pretrain_model.load_state_dict(cp['model_state_dict'])
pretrain_exp = runner.Runner(traindl=traindl, validdl=validdl, testdl=testdl, model=pretrain_model, optimizer=None, lossf=None, args=args)
pretrain_exp.evaluate(pretrain_model, testdl, args)

# print(gt.shape, anomalyscore.shape)
# print(gt[0:10])
# print(anomalyscore[0:10])


