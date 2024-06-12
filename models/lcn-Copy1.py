import torch
from torch import nn

class NonOverlappingLocallyConnected1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, out_dim, s, s0,bias=False
    ):
        self.s0 = s0
        self.s = s
        super(NonOverlappingLocallyConnected1d, self).__init__()
        self.weight = nn.Parameter( # input [bs, cin, space], weight [cout, cin, space]
            torch.randn(
                out_channels,
                input_channels,
                out_dim *s* (s0+1),
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, out_dim))
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels

    def forward(self, x):
        x = x[:, None] * self.weight # [bs, cout, cin, space]

        bs, cout, cin, d = x.shape
        
        x = x.view(bs, cout, cin, d // (self.s*(self.s0+1)), self.s*(self.s0+1)) # [bs, cout, cin, space // 2, 2]
        #*x.shape[:-1]
        
        x = x.sum(dim=[-1, -3]) # [bs, cout, space // 2]
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class LocallyHierarchicalNet(nn.Module):
    def __init__(self, input_channels, h, out_dim, num_layers, s, s0, bias=False):
        super(LocallyHierarchicalNet, self).__init__()

        d = (s*(s0+1)) ** num_layers

        self.hier = nn.Sequential(
            NonOverlappingLocallyConnected1d(
                input_channels, h, d // (s*(s0+1)), s, s0, bias
            ),
            nn.ReLU(),
            *[nn.Sequential(
                    NonOverlappingLocallyConnected1d(
                        h, h, d // (s*(s0+1)) ** (l + 1), s, s0, bias
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):
        y = self.hier(x)
        y = y.mean(dim=[-1])
        y = y @ self.beta / self.beta.size(0)
        return y


# import torch
# from torch import nn
# from .source vm_umt/bin/activatefcn import DenseBlock
#
#
# class LocallyConnected1d(nn.Module):
#     def __init__(
#         self, input_channels, out_channels, out_dim, kernel_size, stride, bias=False
#     ):
#         super(LocallyConnected1d, self).__init__()
#         self.weight = nn.Parameter(
#             torch.randn(
#                 1,
#                 out_channels,
#                 input_channels,
#                 out_dim,
#                 kernel_size,
#             )
#         )
#         if bias:
#             self.bias = nn.Parameter(torch.randn(1, out_channels, out_dim))
#         else:
#             self.register_parameter("bias", None)
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.input_channels = input_channels
#
#     def forward(self, x):
#         k = self.kernel_size
#         s = self.stride
#         x = x.unfold(2, k, s)
#         x = x.contiguous()
#         # Sum in in_channel and kernel_size dims
#         out = (x.unsqueeze(1) * self.weight).sum([2, -1]).div(self.input_channels ** .5)
#         if self.bias is not None:
#             out += self.bias * 0.1
#         return out
#
#
# class LocallyHierarchicalNet(nn.Module):
#     def __init__(self, input_channels, h, out_dim, filter_size, num_layers, bias=False):
#         super(LocallyHierarchicalNet, self).__init__()
#
#         d = filter_size ** num_layers
#
#         self.net = nn.Sequential(
#             LocallyConnected1d(
#                 input_channels, h, d // filter_size, filter_size, filter_size, bias
#             ),
#             nn.ReLU(),
#             *[nn.Sequential(
#                     LocallyConnected1d(
#                         h,
#                         h,
#                         d // filter_size ** (l + 2),
#                         filter_size,
#                         filter_size,
#                         bias,
#                     ),
#                     nn.ReLU(),
#                 )
#                 for l in range(num_layers - 1)
#             ],
#             DenseBlock(h, out_dim, last=True, bias=False)
#         )
#
#     def forward(self, x):
#         return self.net(x)
