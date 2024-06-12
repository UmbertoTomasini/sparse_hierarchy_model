import torch
#from hierarchical_s0_bs_S_diffeoB import HierarchicalDataset
from hierarchical_s0_bs_S import HierarchicalDataset
import math
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--n", type=int, default="4")
parser.add_argument("--m", type=int, default="4")
parser.add_argument("--nc", type=int, default="4")
parser.add_argument("--s0", type=int, default="0")
parser.add_argument("--s", type=int, default="2")
parser.add_argument("--L", type=int, default="2")
parser.add_argument("--pickle", type=str, required=False, default="None")
parser.add_argument("--output", type=str, required=False, default="None")
parser.add_argument("--correct", type=int, default="0")
parser.add_argument("--max_seed", type=int, default="2")
args = parser.parse_args()


n = args.n
m = args.m
nc = args.nc
s0 = args.s0
s = args.s 
L = args.L
device = args.device
max_seed = args.max_seed
bs = 4
#pmax = int(nc*m**((s**L-1)/(s-1)))
#print(pmax)
p_pred = nc*m**(L)
#print(p_pred)
max_num = int((300*p_pred + 50000))

num_batches = math.ceil(max_num / bs)
print(num_batches)
if args.correct == 1:
    if n==6 and L==4 and s0==1:
        st = 200000
    elif n==8 and L==3 and s0==6:
        st = 133908
    elif n==10 and L==3 and s0==2:
        st = 527498
    elif n==12 and L==3 and s0==1: 
        st = 658641   
    elif n==4 and L==4 and s0==4:
        st = 67000
else:
    st = 0
for seed in np.arange(1,max_seed+1,dtype=int):
    for idx_bs in range(st,num_batches):
        print(idx_bs)
        start_time = time.time()
        trainset = HierarchicalDataset(
                        num_features=n,
                        m=m,  # features multiplicity
                        num_layers=L,
                        num_classes=nc,
                        input_format='all0',
                        whitening=1,
                        seed=seed,
                        idx_bs = idx_bs,
                        bs = bs,
                        train=True,
                        transform=None,
                        testsize=0,
                        memory_constraint= bs,#int(pmax/4), #2 * (args.ptr + args.pte),
                        s0 = s0,
                        s = s
                )
        #print(len(trainset))
        #torch.manual_seed(2)
        #perm = torch.randperm(pmax)
        #trainset = torch.utils.data.Subset(trainset, perm)
        print('time for creation: '+str(time.time()-start_time))
        torch.save(trainset, 'storing_all0/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d.pt'%(n,m,nc,L,s0,s,idx_bs))
         #print('time for creation: '+str(time.time()-start_time))
