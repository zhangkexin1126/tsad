import os
import argparse
import torch

class Options(object):
    def __init__(self):
        # base args
        self.parser = argparse.ArgumentParser(description='interface of running experiments for CPCAD baselines')
        self.parser.add_argument('--datapath', type=str, default='data', help='[ ./data ] prefix path to data directory')
        self.parser.add_argument('-data', '--dataname', type=str, default='SMD', help='dataset name [SMD, MSL, PSM, SMAP, SWAT] ')
        self.parser.add_argument('--outputpath', type=str, default='save', help='[ ./save ] prefix path to save algorithm save')

        # dataloader args
        self.parser.add_argument('--batch_size', type=int, default=256, help='batch_size for training')

        # window args
        self.parser.add_argument('--win_size', type=int, default=100, help='size for whole window')
        self.parser.add_argument('--win_step', type=int, default=100, help='window step')
        self.parser.add_argument('--xp_size', type=int, default=80, help='size for history window')
        self.parser.add_argument('--xn_size', type=int, default=20, help='size for next window')
        self.parser.add_argument('--pred_step', type=int, default=8, help='which time step in next window is seleced to pred, must < xn_size-1')
        ## mask ratio [if use]
        self.parser.add_argument('--win_maskratio', type=float, default=0.8, help='mask_ratio')


        # network args
        self.parser.add_argument('--x_dim', type=int, default=1, help='Dim of input time-series')
        self.parser.add_argument('--ts_embed_dim', type=int, default=128, help='Dim of hidden representations in embedding space for each time-step')
        self.parser.add_argument('--encoder_select', type=str, default='tcn', help='trans or encoder')
        self.parser.add_argument('--encoder_embed_dim', type=int, default=128, help='dim of final encoder save')

        ## TCN encoder layers
        self.parser.add_argument('--tcnindim', type=int, default=128, help='dim of each tcn block save, except final block')
        self.parser.add_argument('--tcnblockdim', type=int, default=256, help='dim of each tcn block save, except final block')
        self.parser.add_argument('--tcnoutdim', type=int, default=128, help='dim of tcn encoder')
        self.parser.add_argument('--tcndepth', type=int, default=3, help='depth of TCN encoder')
        self.parser.add_argument('--tcnkernelsize', type=int, default=3, help='convolution kernel size')
        self.parser.add_argument('--tcndropout', type=float, default=0.1, help='dropout')
        self.parser.add_argument('--tcninputmode', type=str, default="blc", help='length first or dim first, blc / bcl')
        # Transformer Encoder layers
        self.parser.add_argument('--transindim', type=int, default=128, help='dim of each tcn block save, except final block')
        self.parser.add_argument('--trans_in_length', type=int, default=100, help='dim of each tcn block save, except final block')
        self.parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
        self.parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
        self.parser.add_argument('--transblockdim', type=int, default=256, help='dim of each trans block inside')
        # RNN regression
        self.parser.add_argument('--rnndepth', type=int, default=1, help='depth of RNN regression')
        self.parser.add_argument('--rnndim', type=int, default=256, help='depth of RNN regression')

        # loss args
        self.parser.add_argument('--temperature', type=float, default=0.07, help='Logits are divided by temperature before calculating the cross entropy')

        # train&eval
        self.parser.add_argument('--trainstride', type=int, default=50, help='for reduceing training samples')
        self.parser.add_argument("-epo", '--epoch', type=int, default=10, help='max iteration for training')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument('--use_gpu', type=bool, default=True, help='whether use gpu')
        self.parser.add_argument('-thre', '--threshold', type=float, default=0.5, help='threshold for determine anomaly points in threshold-based methods')
        self.parser.add_argument('--evalonly', action='store_true', help='whether to evaluate, if True, training process is not available')
        self.parser.add_argument('-evme', '--evalmethod', type=str, default='recsim', help='rec/sim/recsim')

    def parse(self):
        args = self.parser.parse_args()

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu:
            args.device = torch.device("cuda:0")
        else:
            args.device = torch.device("cpu")
        print(">>>>>>> GPU Info [use/cuda/device]", args.use_gpu, torch.cuda.is_available(), args.device)

        if not os.path.exists(args.outputpath):
            os.makedirs(args.outputpath)

        if args.dataname == "YAHOO":
            args.x_dim = 1
        elif args.dataname == "SMD":
            args.x_dim = 38
        elif args.dataname == "MSL":
            args.x_dim = 55
        elif args.dataname == "PSM":
            args.x_dim = 25
        elif args.dataname == "SMAP":
            args.x_dim = 25
        elif args.dataname == "SWAT":
            args.x_dim = 33333333
        else:
            args.x_dim = 0

        assert args.ts_embed_dim == args.tcnindim
        # assert args.ts_embed_dim == args.transindim
        # assert args.win_size == args.trans_in_length
        assert args.win_size == (args.xp_size + args.xn_size)

        return args

    def parsenb(self):
        args, unknown = self.parser.parse_known_args()

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        # print(args.use_gpu, torch.cuda.is_available())
        if args.use_gpu:
            args.device = torch.device("cuda:0")
        else:
            args.device = torch.device("cpu")
        # print(args.device)

        if not os.path.exists(args.outputpath):
            os.makedirs(args.outputpath)

        if args.dataname == "YAHOO":
            args.x_dim = 1
        elif args.dataname == "SMD":
            args.x_dim = 38
        elif args.dataname == "MSL":
            args.x_dim = 55
        elif args.dataname == "PSM":
            args.x_dim = 25
        elif args.dataname == "SMAP":
            args.x_dim = 25
        elif args.dataname == "SWAT":
            args.x_dim = 33333333
        else:
            args.x_dim = 0

        assert args.ts_embed_dim == args.tcnindim
        # assert args.ts_embed_dim == args.transindim
        assert args.win_size == args.trans_in_length

        return args