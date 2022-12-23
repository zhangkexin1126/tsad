import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from kmeans_pytorch import kmeans
from torch.nn.utils import weight_norm
from sklearn.cluster import KMeans
from src.model_tcn import TCN_Encoder
from src.model_transformer import Transformer_Encoder
from src.model_embed import Embedding_Encoder_MLP, Embedding_Encoder_Conv
from src.model_decoder import Reconstruct_Decoder_MLP, Context_Decoder, Predict_Decoder, Reconstruct_Decoder_RNN
from src.model_decoder import Poolinger, Projector, Predictor

model_options = {"mlp_embed": Embedding_Encoder_MLP,
              "conv_embed": Embedding_Encoder_Conv,
              "tcn_encoder": TCN_Encoder,
              "trans_encoder": Transformer_Encoder,
              "mlp_projector": Projector,
              "mlp_predictor": Predictor,
              "mlp_reconstruct": Reconstruct_Decoder_MLP,
              "pool": Poolinger}

class ContrastiveLearningTSAD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ts_embed = model_options["mlp_embed"](in_dim=args.x_dim, out_dim=args.ts_embed_dim)
        if args.encoder_select == "tcn":
            self.encoder_block = model_options["tcn_encoder"](in_channels=args.tcnindim,
                                                 tcnblockdim=args.tcnblockdim,
                                                 out_channels=args.tcnoutdim,
                                                 depth=args.tcndepth,
                                                 kernel_size=args.tcnkernelsize,
                                                 inputmode=args.tcninputmode)
        elif args.encoder_select == "trans":
            self.encoder_block = model_options["trans_encoder"](in_channels=args.transindim,
                                                     in_length=args.trans_in_length,
                                                     n_heads=args.n_heads,
                                                     n_layers=args.n_layers,
                                                     out_channels=args.encoder_embed_dim,
                                                     transblockdim=args.transblockdim)

        # reconstruct_decoders
        # self.add_decoder = model_options["mlp_reconstruct"](args.tcnblockdim, args.ts_embed_dim)
        self.decoders = nn.ModuleList([model_options["mlp_reconstruct"](args.tcnblockdim, args.x_dim) for _ in range(args.tcndepth)])
        self.decoders.append(model_options["mlp_reconstruct"](args.tcnoutdim, args.x_dim))
        # multi-resolution poolings
        NK = [1, 2, 4, 8]
        self.pools = nn.ModuleList([model_options["pool"](k, k, mode="max") for k in NK])
        # multi-layer predictor for contrastive
        self.predictor_out = model_options["mlp_predictor"](args.tcnoutdim)
        self.predictor_block = model_options["mlp_predictor"](args.tcnblockdim)

        # CPC module
        # self.contextual_block_1 = Context_Decoder(input_dim=args.tcnblockdim, output_dim=args.rnndim, rnn_layers=args.rnndepth)
        # self.contextual_block_2 = Context_Decoder(input_dim=args.tcnblockdim, output_dim=args.rnndim, rnn_layers=args.rnndepth)
        # self.pred_block_1 = Predict_Decoder(in_dim=args.rnndim, out_dim=args.tcnblockdim)
        # self.pred_block_2 = Predict_Decoder(in_dim=args.rnndim, out_dim=args.tcnblockdim)

    def forward(self, x):
        x_raw = x
        z = self.ts_embed(x_raw) # embedding
        encoder_out = self.encoder_block(z) # tcn_out [out1, out2, out3, ...], by num_layers

        """Global multi-granular Rescontruct view"""
        # h = encoder_out[-1]
        # rec = self.reconstruct_decoder(h)
        # print("LEN Decoders", len(self.decoders), len(encoder_out))
        REC_list = []
        for k in range(len(self.decoders)):
            h = encoder_out[k]
            rec = self.decoders[k](h)
            REC_list.append(rec)

        # addrec = self.add_decoder(encoder_out[k])
        # print("===", addrec.shape)
        #
        # hh

        """Local Contrastive view - different granular"""
        Z_list = []
        P_list = []
        Q_list = []
        for k in range(len(encoder_out) - 1): # drop last layer
            z = encoder_out[k] # out of each TCN layer
            # pred
            z_pred = self.predictor_block(z) #proj
            # pool
            # z_pool = self.pools[k](z)

            Z_list.append(z)
            P_list.append(z_pred)
            # Q_list.append(z_pool)




        """Local Cross Predictive view - skip-step learning, CPC"""
        # z1_hist = encoder_out[0][:, 0:self.args.xp_size,:]
        # z2_hist = encoder_out[-1][:, 0:self.args.xp_size,:]
        # # print(z1_hist.shape, z2_hist.shape)
        # c1, (h0, c0) = self.contextual_block_1(z1_hist)
        # c2, (h0, c0) = self.contextual_block_2(z2_hist)
        # c1 = c1[:, -1, :]
        # c2 = c2[:, -1, :]
        # # print("c1/c2", c1.shape, c2.shape)
        # pred2 = self.pred_block_1(c1)
        # pred1 = self.pred_block_2(c2)
        # # print("pred1/pred2", pred2.shape, pred1.shape)
        # target1 = encoder_out[0][:, -1,:]
        # target2 = encoder_out[-1][:, -1,:]
        # # print("target1/target2", target1.shape, target2.shape)


        """Variable-level, kur"""
        # B,L,C -> B,C,L
        # x_var = x_raw.permute(0, 2, 1)

        return x, z, encoder_out, REC_list, Z_list, P_list
        # pred1, target1.detach(), pred2, target2.detach()

        # return x, z, h, rec, xn_pred, xn_target

    def trash(self):
        pass
        ## if use RNN decoder
        # h_hidden = torch.mean(h, dim=1).unsqueeze(dim=0)
        # rec = self.reconstruct_decoder(h, h_hidden)


if __name__ == "__main__":
    pass
