import torch
from torch import nn

class NonOverlappingLocallyConnected1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, out_dim, s, s0,sharings,idx_layer, bias=False
    ):
        sharing = sharings[idx_layer]
        ###
        print(sharing)
        self.s0 = s0
        self.s = s
        super(NonOverlappingLocallyConnected1d, self).__init__()
        if idx_layer==0:
            if sharing=='True':
                out_channels = int(out_channels*(s0+1))
        else:
            if sharings[idx_layer-1]=='True':
                input_channels = input_channels*(s0+1)
            if sharings[idx_layer]=='True':
                out_channels = out_channels*(s0+1)
        ###
        print(input_channels, out_channels)
        if sharing=='True':
            self.weight = nn.Parameter( # input [bs, cin, space], weight [cout, cin, space]
                torch.randn(
                    out_channels,
                    input_channels,
                    out_dim *s,
                )
            )            
        else:
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
        self.sharing = sharing
    def forward(self, x):
        if self.sharing=='True':

            bs, cin, d = x.shape
            #print("input layer")
            #print(x)
            #adding partial sum every s0+1 elements, each w paramter will be multiplied with this partial sum
            x = x.view(bs, cin, d//(self.s0+1),self.s0+1)        
            x = x.sum(dim = [-1])
            #print("after s0+1 pooling")
            #print(x)
            x = x[:,None]*self.weight
            
            bs, cout, cin, d = x.shape
            x = x.view(bs, cout, cin, d // (self.s), self.s)    

            x = x.sum(dim=[-1, -3])
            #print("after w")
            #print(x)
        else:
            x = x[:, None] * self.weight # [bs, cout, cin, space]
            bs, cout, cin, d = x.shape
            
            x = x.view(bs, cout, cin, d // (self.s*(self.s0+1)), self.s*(self.s0+1)) # [bs, cout, cin, space // 2, 2]
            x = x.sum(dim=[-1, -3]) # [bs, cout, space // 2]
        
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class LocallyHierarchicalNetShared(nn.Module):
    def __init__(self, input_channels, h, out_dim, num_layers, s, s0, sharings, bias=False):
        super(LocallyHierarchicalNetShared, self).__init__()

        d = (s*(s0+1)) ** num_layers
        
        #sharings is a list of boolean values, each one accounting for whether sharing or not the parameters
        #whether sharing or not the parameters of a given layer
        
        self.hier = nn.Sequential(
            NonOverlappingLocallyConnected1d(
                input_channels, h, d // (s*(s0+1)), s, s0, sharings, 0 ,bias=bias
            ),
            nn.ReLU(),
            *[nn.Sequential(
                    NonOverlappingLocallyConnected1d(
                        h, h, d // (s*(s0+1)) ** (l + 1), s, s0, sharings, l, bias=bias
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )
        if sharings[-1]=='True':
            self.beta = nn.Parameter(torch.randn(h*(s0+1), out_dim))
        else:
            self.beta = nn.Parameter(torch.randn(h, out_dim))
    def forward(self, x):
        y = self.hier(x)
        y = y.mean(dim=[-1])
        y = y @ self.beta / self.beta.size(0)
        return y

