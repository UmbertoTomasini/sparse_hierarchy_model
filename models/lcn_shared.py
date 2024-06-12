import torch
from torch import nn


class NonOverlappingLocallyConnected1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, out_dim, s, s0,idx_layer, sharing = 0, bias=False
    ):
        ###
        self.s0 = s0
        self.s = s
        super(NonOverlappingLocallyConnected1d, self).__init__()
        
        out_channels = int(out_channels)#*(s0+1)**sharing)
        ###
        print(input_channels, out_channels)

        self.weight = nn.Parameter( # input [bs, cin, space], weight [cout, cin, space]
            torch.randn(
                out_channels,
                input_channels,
                int(out_dim *s* (s0+1)/((s0+1)**sharing)),
            )
        )
         
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, out_dim))
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels
        self.sharing = sharing
    def forward(self, x):
        #print('sharing self')
        #print(self.sharing)
        
        #print('before anything')
        #print(x)
        if self.sharing==2:
            bs, cin, d = x.shape
            #print("input layer")
            #adding partial sum every s0+1 elements, each w paramter will be multiplied with this partial sum
            x = x.view(bs, cin, d//(self.s0+1),self.s0+1)        
            x = x.sum(dim = [-1])
            #print("after s0+1 pooling")
            #for L=2
            x = x.view(bs, cin, self.s,(self.s0+1)*self.s)

            for i in range(self.s):
                tmp = x[:,:,i,:].view(bs,cin,self.s0+1,self.s)
            
                if i==0:
                    mat = tmp
                else:
                    mat = torch.concat((mat,tmp),dim=-1)
            #print(mat.shape)
            #print(self.weight.shape)
            
            for j in range(self.s0+1):
                #print(j)
                #print(mat[:,:,j,:,None].shape)
                #print(self.weight.shape)
                tmp_mat = mat[:,:,j,:]
                #print('one instance of s0')
                #print(tmp_mat)
                #print(self.weight)
                tmp_row = tmp_mat[:,None]*self.weight
                if j==0:
                    x = tmp_row
                    #print('x:' +str(x.shape))
                elif j==1:
                    x = torch.stack((x, tmp_row),dim =3)
                    #print('x:' +str(x.shape))
                else:
                    x = torch.concat((x, tmp_row.unsqueeze(dim=3)),dim =3)
            #x = mat[:,None]*self.weight
            
            bs, cout, cin, s0s, d = x.shape

            x = x.view(bs, cout, cin, s0s, d // (self.s), self.s)    
            
            x = x.sum(dim=[-1, -4])

            x = x.view(bs,cout,s0s*(d // (self.s)))
            #print('final')
            #print(x.shape)
        
        elif self.sharing==1:

            bs, cin, d = x.shape
            #print("input layer")
            #print(x)
            #adding partial sum every s0+1 elements, each w paramter will be multiplied with this partial sum
            x = x.view(bs, cin, d//(self.s0+1),self.s0+1)        
            x = x.sum(dim = [-1])
            #print("after s0+1 pooling")
            #print(x)

            #for L=2
            
            x = x[:,None]*self.weight
            
            bs, cout, cin, d = x.shape
            x = x.view(bs, cout, cin, d // (self.s), self.s)    

            x = x.sum(dim=[-1, -3])
            #print("after w")
            #print(x)
        elif self.sharing== 0:
            
            x = x[:, None] * self.weight # [bs, cout, cin, space]
            bs, cout, cin, d = x.shape
            
            x = x.view(bs, cout, cin, d // (self.s*(self.s0+1)), self.s*(self.s0+1)) # [bs, cout, cin, space // 2, 2]
            x = x.sum(dim=[-1, -3]) # [bs, cout, space // 2]
        
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class LocallyHierarchicalNetShared(nn.Module):
    def __init__(self, input_channels, h, out_dim, num_layers, s, s0, sharing, bias=False):
        super(LocallyHierarchicalNetShared, self).__init__()

        d = (s*(s0+1)) ** num_layers
        
        #sharings is a list of boolean values, each one accounting for whether sharing or not the parameters
        #whether sharing or not the parameters of a given layer
        if sharing ==2: 
            sharing_upper = 1
        else:
            sharing_upper = 0
        self.hier = nn.Sequential(
            NonOverlappingLocallyConnected1d(
                input_channels, h, d // (s*(s0+1)), s, s0,  0 ,sharing,bias=bias
            ),
            nn.ReLU(),
            *[nn.Sequential(
                    NonOverlappingLocallyConnected1d(
                        h, h, d // (s*(s0+1)) ** (l + 1), s, s0, l, sharing=sharing_upper,bias=bias
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

