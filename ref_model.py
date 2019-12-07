"""Reference encoder module for prosody transfer experiment

"""

from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
# from layers import ConvNorm, LinearNorm
# from utils import to_gpu, get_mask_from_lengths

def conv_output_dims(conv_net, input_dims):
    """A trick to learn the output dimensions of a convnet with known input
    dimensions. Replace this if you find a better way to do it.

    This assumes the convnet should take (N, 1, W1, H1) -> (N, C2, W2, H2). We
    want to know W2, H2.

    """
    input_tensor = torch.zeros(1, 1, *input_dims)
    return conv_net(input_tensor).shape[-2:]


class ReferenceEncoder(nn.Module):
    r"""Recall the architecture of the reference encoder:

               Final linear
               layer & act.
                    ^
                    |
                   GRU       (an RNN block)
                    ^
                    |
                CNN block
                    ^
                    |
                reference
               spec frames
    """
    def __init__(self, hparams):
        r"""Accepted hparams
        ===========================
        input_dims       : (L_R, d_R)
        filter_size      : receptive field of the CNN neurons
        filter_stride    : stride of the CNN layers
        filter_channels  : array of channel depths in the CNN layers
        ref_embedding_dim: dimension of the prosody space
        activation       : an activation function for the final layer
        """
        super(ReferenceEncoder, self).__init__()

        self.ref_embedding_dim = hparams.ref_embedding_dim
        self.activation = hparams.activation

        self.conv_block = ConvBlock(hparams.input_dims, hparams.filter_size,
                                    hparams.filter_stride, hparams.filter_channels)

        # rnn input dimension should be (cnn output channels) * dR_reduced
        rnn_input_dim = hparams.filter_channels[-1] *\
            conv_output_dims(self.conv_block, hparams.input_dims)[0]
        self.rnn_block = RnnBlock(rnn_input_dim, self.ref_embedding_dim, hparams.batch_size)

        # TODO: should there be bias in the linear layer?
        self.linear_layer = nn.Linear(self.ref_embedding_dim, self.ref_embedding_dim, bias=True)

    def forward(self, x):
        # assert x.shape[-2:] == torch.Size(self.input_dims)  # this makes me a little uneasy
        x = self.conv_block(x.unsqueeze(1))
        x = self.rnn_block(x)
        x = self.activation(self.linear_layer(x))
        return x


class ConvBlock(nn.Module):
    r"""The convolutional block of the reference encoder. I'm made a little
    uncertain by the requirement of constant L_R

    """
    def __init__(self, input_dims, filter_size, filter_stride, filter_channels):
        r"""input_dims: the dimensions (L_R, d_R) of the reference signal to be encoded
        filter_size: the (identical) receptive field of each filter
        filter_stride: the (identical) stride of each filter filter_channels: an
        array containing the /number of filters/ in each layer

        """
        super(ConvBlock, self).__init__()

        # self.filter_size = filter_size
        # self.filter_stride = filter_stride
        # self.filter_channels = filter_channels

        # # TODO: double-check this is correct
        # # I don't really think it is.
        # self.output_dim = self.filter_channels[-1]

        ### Internal layers ###
        # In each layer, filters with SAME padding, ReLU activation, and
        # Batchnorm.
        self.layers = nn.Sequential()
        for i, filters in enumerate(filter_channels):
            # TODO: how many in_channels and out_channels?
            in_channels = 1 if i == 0 else filter_channels[i - 1]
            # TODO: should use SAME padding. This doesn't appear to exist in
            # PyTorch. Newer versions of PyTorch do have a `padding_mode`
            # keyword argument, but it has no `SAME` option.
            same_padding = int(filter_size / 2)
            self.layers.add_module('conv{}'.format(i),
                nn.Conv2d(in_channels, filters, filter_size,
                          stride=filter_stride,
                          padding=same_padding)
            )

            self.layers.add_module('bn{}'.format(i),
                nn.BatchNorm2d(num_features=filters,
                               affine=True)
            )

            self.layers.add_module('relu{}'.format(i),
                                   nn.ReLU())


    def forward(self, x):
        r"""Forward pass through the convolution stage of the reference encoder. The if
        x has shape (1, 1, n, m), the output will have shape (1,
        self.filter_channels[-1], ~n/(stride ** depth), ~m/(stride ** depth)).

        """
        return self.layers(x)


class RnnBlock(nn.Module):
    r"""The GRU-RNN block that compresses the output down to a fixed dimension.
    A thin wrapper around the built-in torch.nn.GRU module.

    TODO: how should we initialize h_0? Currently just zeros.
    TODO: need to calculate input_dim

    N.b. the PyTorch GRU is a little odd: its batch index is the /second/
    index, rather than the first. see https://pytorch.org/docs/stable/nn.html.

    I think where the prosody paper says "the dR/64 feature dimensions and 128
    channels of the final convolution layer are unrolled as the inner dimension
    or the resulting LR/64 x (128 dR/64) matrix", they are talking about the
    reshaping operation below. (Please excuse the crummy ascii art).

              LR, dR in wrong order. I'm not going to re-draw it.
                                V

              (N,   128,  LR/64,    dR/64)
                \     \___/           /
                 \_______/\          /
                        /\ \____x___/
                       /  \     |
                     (L,   N,  H_in)
                                 ^ input_dim

    """
    def __init__(self, input_dim, embedding_dim, batch_size):
        # Sorry, batch size shouldn't be an explicit argument here. This is really ugly.
        super(RnnBlock, self).__init__()
        # TODO: should we use bias here? dropout?
        self.gru = nn.GRU(input_size=input_dim, hidden_size=embedding_dim, num_layers=1)

        # is this the right/best way to set the initialization? Seems wasteful.
        self.register_buffer('h0', torch.zeros(1, batch_size, embedding_dim))

    def forward(self, x):
        r"""Native input dimension is (N, 128, dR/64, LR/64). Need to first permute the
        batch index into the order supported by the GRU, then flatten the
        ConvNet channel and reference dimensions.

        """
        x = x.permute(3,0,1,2).flatten(start_dim=-2)

        # native input dimension for h0 is (S, N, hidden_size).
        # h0 = torch.zeros(1, x.shape[1], self.embedding_dim)

        # discard the GRU output: don't care about it.
        _, hn = self.gru(x, self.h0[:,:x.shape[1],:]) # why are some batch sizes different?

        # Native output dimension is (S, N, H_out). Want (N, H_out).
        return hn.squeeze()
