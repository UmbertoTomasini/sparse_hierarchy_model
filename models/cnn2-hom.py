import torch
from torch import nn
import torch.nn.functional as F

class NonOverlappingConv1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, out_dim, s0, s,bias=False
    ):
        self.s0 = s0
        self.s = s
        super(NonOverlappingConv1d, self).__init__()
        self.weight = nn.Parameter( # input [bs, cin, space / 2, 2], weight [cout, cin, 1, 2]
            torch.ones(
                out_channels,
                input_channels,
                1,
                s*(s0+1),
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, out_dim))
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels

    def forward(self, x):
        bs, cin, d = x.shape
        #print(x.shape)
        #print(d,self.s,self.s0, self.s*(self.s0+1))
        x = x.view(bs, 1, cin, d // (self.s*(self.s0+1)), self.s*(self.s0+1)) # [bs, 1, cin, space // 2, 2]
        x = x * self.weight # [bs, cout, cin, space // 2, 2]
        x = x.sum(dim=[-1, -3]) # [bs, cout, space // 2]
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class CNN2(nn.Module):
    """
        CNN crafted to have an effective size equal to the corresponding HLCN.
    """
    def __init__(self, input_channels, h, out_dim, num_layers, s0, s,bias=False):
        super(CNN2, self).__init__()

        d = (s*(s0+1)) ** num_layers

        self.hier = nn.Sequential(
            NonOverlappingConv1d(
                input_channels, h, d // (s*(s0+1)), s0,s,bias
            ),
            nn.ReLU(),
            *[nn.Sequential(
                    NonOverlappingConv1d(
                        h, h, d // ((s*(s0+1)) ** (l + 1)),s0,s, bias
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )

        # force last layer representation
        # beta_onehot = F.one_hot(torch.arange(out_dim)[None].repeat(h // out_dim, 1).t().flatten()).float() * 10
        # self.beta = nn.Parameter(beta_onehot)
        # self.register_buffer('beta', beta_onehot)
        self.beta = nn.Parameter(torch.randn(h, out_dim)) 
    def forward(self, x):
        y = self.hier(x)
        y = y.mean(dim=[-1])
        y = y @ self.beta / self.beta.size(0)
        return y
