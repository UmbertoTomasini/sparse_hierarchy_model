import torch
from hierarchical_s0_bs_S_MOD import HierarchicalDataset
import math
import time
import argparse


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
args = parser.parse_args()


n = args.n
m = args.m
nc = args.nc
s0 = args.s0
s = args.s 
L = args.L
device = args.device

bs = 100
idx_bs = 0

#num_batches = 100
#st = 0

seed_reset_layer_pos = 42
seed_reset_layer_gauss = 42
x = []
for seed_reset_layer_sem in range(L, -1, -1):
    trainset = HierarchicalDataset(
                    num_features=n,
                    m=m,  # features multiplicity
                    num_layers=L,
                    num_classes=nc,
                    input_format='all0',
                    whitening=1,
                    seed=1,
                    idx_bs = idx_bs,
                    bs = bs,
                    train=True,
                    transform=None,
                    testsize=0,
                    memory_constraint= bs,
                    s0 = s0,
                    s = s,
                    seed_reset_layer_sem =seed_reset_layer_sem,
                    seed_reset_layer_pos =seed_reset_layer_pos,
                    seed_reset_layer_gauss =seed_reset_layer_gauss,
                    unique_datapoints=0
            )
    x.append(trainset.x)
x = torch.cat(x)
torch.save(x, 'storing_sem/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d.pt'%(n,m,nc,L,s0,s,idx_bs))

seed_reset_layer_sem = 42
seed_reset_layer_gauss = 42
x = []
for seed_reset_layer_pos in range(L, -1, -1):
    trainset = HierarchicalDataset(
                    num_features=n,
                    m=m,  # features multiplicity
                    num_layers=L,
                    num_classes=nc,
                    input_format='all0',
                    whitening=1,
                    seed=1,
                    idx_bs = idx_bs,
                    bs = bs,
                    train=True,
                    transform=None,
                    testsize=0,
                    memory_constraint= bs,
                    s0 = s0,
                    s = s,
                    seed_reset_layer_sem =seed_reset_layer_sem,
                    seed_reset_layer_pos =seed_reset_layer_pos,
                    seed_reset_layer_gauss =seed_reset_layer_gauss,
                    unique_datapoints=0
            )
    x.append(trainset.x)
x = torch.cat(x)
torch.save(x, 'storing_pos/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d.pt'%(n,m,nc,L,s0,s,idx_bs))

seed_reset_layer_sem = 42
seed_reset_layer_pos = 42
x = []
for seed_reset_layer_gauss in range(L, 0, -1):
    trainset = HierarchicalDataset(
                    num_features=n,
                    m=m,  # features multiplicity
                    num_layers=L,
                    num_classes=nc,
                    input_format='all0',
                    whitening=1,
                    seed=1,
                    idx_bs = idx_bs,
                    bs = bs,
                    train=True,
                    transform=None,
                    testsize=0,
                    memory_constraint= bs,
                    s0 = s0,
                    s = s,
                    seed_reset_layer_sem =seed_reset_layer_sem,
                    seed_reset_layer_pos =seed_reset_layer_pos,
                    seed_reset_layer_gauss =seed_reset_layer_gauss,
                    unique_datapoints=0
            )
    x.append(trainset.x)
x = torch.cat(x)
torch.save(x, 'storing_gauss/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d.pt'%(n,m,nc,L,s0,s,idx_bs))
