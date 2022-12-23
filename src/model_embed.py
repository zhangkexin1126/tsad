
import torch
import torch.nn as nn

class Embedding_Encoder_Conv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.proj = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1,1),
                              padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.bn = nn.BatchNorm1d(num_features=out_dim)

    def forward(self, x):
        """
        First Embedding Layer
        :param x:
        :return:
        """
        x = x.permute(0, 2, 1)  # BLC -> BCL for time-series data
        x = self.bn(self.proj(x))
        x = x.permute(0, 2, 1)
        return x


class Embedding_Encoder_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self.bn = nn.BatchNorm1d(num_features=out_dim)
        # self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        First Embedding Layer
        :param x: input sequence with shape B*L*C
        :return: output sequence with shape B*C*L
        """
        x = self.proj(x)
        x = x.permute(0, 2, 1) # BLC -> BCL
        x = self.bn(x)
        x = x.permute(0, 2, 1) # BCL -> BLC
        return x