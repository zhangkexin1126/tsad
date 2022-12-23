import torch


"""
TCN
"""

class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of save channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            # return out_causal + res
            return out_causal + res
        else:
            # return self.relu(out_causal + res)
            return self.relu(out_causal + res)


class TCN_Encoder(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of save
           channels.
    @param depth Depth of the network.
    @param out_channels Number of save channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, tcnblockdim, depth, out_channels, kernel_size, inputmode):
        super(TCN_Encoder, self).__init__()

        self.inputmode = inputmode

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size, default = 1

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else tcnblockdim
            layers += [CausalConvolutionBlock(
                in_channels_block, tcnblockdim, kernel_size, dilation_size
            )]
            # dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            tcnblockdim, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)
        self.N_LAYER = len(layers)


    def forward(self, x):
        if self.inputmode == "blc":
            x = x.permute(0, 2, 1)
            out = self.forward_layer(x)
            # out = out.permute(0, 2, 1)  # return to B, L, out_dim
        elif self.inputmode == "bcl":
            out = self.network(x)
        return out

    def forward_layer(self, x):
        """Input X: BCL"""
        out = []
        for k in range(self.N_LAYER):
            x = self.network[k](x)
            h = x.permute(0, 2, 1)
            out.append(h)
        return out

"""
TCN version 2
"""



##  TCN Encoder ##
# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size
#
#     def forward(self, x):
#         return x[:, :, :-self.chomp_size]
#
#
# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = weight_norm(
#             nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
#
#         self.conv2 = weight_norm(
#             nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2,
#                                  self.relu2, self.dropout2)
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()
#
#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)
#
#
# class TCN_Encoder(nn.Module):
#     def __init__(self, in_channels, tcnblockdim, out_channels, depth, kernel_size, dropout, inputmode):
#         super(TCN_Encoder, self).__init__()
#         layers = []
#         dilation_size = 1  # Init dilation size
#         tcnstride = 1
#         self.inputmode = inputmode
#         for i in range(depth):
#             in_channels_block = in_channels if i == 0 else tcnblockdim
#             layers += [TemporalBlock(in_channels_block, tcnblockdim, kernel_size, tcnstride, dilation_size,
#                                      (kernel_size - 1) * dilation_size, dropout)]
#             dilation_size = 2 ** i
#
#         # add last layer
#         layers += [TemporalBlock(tcnblockdim, out_channels, kernel_size, tcnstride, dilation_size,
#                                  (kernel_size - 1) * dilation_size, dropout)]
#         self.network = nn.Sequential(*layers)
#
#     def forward(self, x):
#         'input shape: B L C'
#
#         if self.inputmode == "blc":
#
#             x = x.permute(0, 2, 1)
#             print("--->", x.shape)
#             out = self.network(x)
#             out = out.permute(0, 2, 1)  # return to B, L, out_dim
#         elif self.inputmode == "bcl":
#             out = self.network(x)
#         return out
