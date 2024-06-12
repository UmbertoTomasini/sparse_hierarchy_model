import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import itertools 
import random
import numpy as np
#import matplotlib.lines as mlines
import pickle
import io
from stabs import*
from torchvision.models.feature_extraction import get_graph_node_names


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
def load_file(f):
    with open(f, "rb") as rb:
        pickle.load(rb)
        #return pickle.load(rb)
        return CPU_Unpickler(rb).load()
def remove_module_state(state):
    # original saved file with DataParallel
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        #print(k)
        name = k[7:] # remove `module.`
        new_state[name] = v
        #print(name)
    return new_state

def choice_lr(net,s,s0,L):
    if net=='fcn2':
        if L==2:
            if s0<4:
                lr = 0.01
            else:
                lr = 0.003
        else:
            if s0<4:
                lr = 0.003
            else:
                lr = 0.001
    elif net=="cnn2":
        if s>2:
            if s0<4:
                lr = 0.01
            elif s0==4:
                lr = 0.003
            elif s0==6:
                lr = 0.0003
            if s==3 and n==4:
                if s0<4:
                    lr = 0.003
        else:
            if s0>0:
                if L>=3:
                    if s0<4:
                        lr = 0.01
                    elif s0==4:
                        lr = 0.003
                    else:
                        lr = 0.001
                elif L==2:
                    if s0<4:
                        lr = 0.1
                    elif s0==4:
                        lr = 0.03
                    else:
                        lr = 0.01
            else:
                lr = 0.01
            if L==1:
                if s0<4:
                    lr = 0.1
                else:
                    lr = .03
    elif net=='lcn':
        if s>2:
            lr = .003
        else:
            if s0<4:
                lr = 0.01
            else:
                lr = 0.003
    return lr
    
def training_point(net,s,s0,L,n,m,type_diffeo):
    if  net=='fcn2':
        if s==2:
            p_pred = n**(L)
            #pmax = ((2*s0+2)**(2**L -1))*n**(2**L)


            if s0 <4:
                xx= np.logspace(np.log10(int(p_pred)),np.log10(200*p_pred),15)
                xx =xx[:-1]
            else:
                xx= np.logspace(np.log10(p_pred),np.log10(500*p_pred),15)
                xx =xx[:-1]
                #xx1 = np.array([107999,135917])
                #xx = np.concatenate((xx,xx1))
        #elif s==2 and L==3:
        #    p_pred = n**(L)
            #pmax = ((2*s0+2)**(2**L -1))*n**(2**L)


        #    if s0 <4:
        #        xx= np.logspace(np.log10(int(p_pred)),np.log10(400*p_pred),15)
        #        xx =xx[:-1]
        #    else:
        #        xx= np.logspace(np.log10(p_pred),np.log10(600*p_pred),15)
        #        xx =xx[:-1]
                #xx1 = np.array([107999,135917])
                #xx = np.concatenate((xx,xx1))
        elif s>2:
            p_pred = m**(L) #((s0+1)**L)*m**(L)

            xx= np.logspace(np.log10(int(0.1*p_pred)),np.log10(10*p_pred),10)
            xx1 = np.logspace(np.log10(10*p_pred),np.log10(50*p_pred),7)
            xx = np.concatenate((xx,xx1))
    elif 'VGG' in net:
        p_pred = n**(L)
        xx= np.logspace(np.log10(0.1*int(p_pred)),np.log10(200*p_pred),15)
        xx0 = np.array([i for i in range(440,1500,100)])
        xx = np.concatenate((xx,xx0))
        xx = np.sort(xx)
    elif 'ResNet' in net:
        p_pred = n**(L)
        xx= np.logspace(np.log10(0.1*int(p_pred)),np.log10(200*p_pred),15)
        xx0 = np.array([i for i in range(440,1500,100)])
        xx = np.concatenate((xx,xx0))
        xx = np.sort(xx)
    elif 'Efficient' in net:
        p_pred = n**(L)
        xx= np.logspace(np.log10(0.1*int(p_pred)),np.log10(200*p_pred),15)
        xx0 = np.array([i for i in range(440,1500,100)])
        xx = np.concatenate((xx,xx0))
        xx = np.sort(xx)
    elif net=='lcn':
        if s==2:
            p_pred = n**(L)
            #pmax = ((2*s0+2)**(2**L -1))*n**(2**L)


            if s0 <4:
                xx= np.logspace(np.log10(int(p_pred)),np.log10(200*p_pred),15)
                xx =xx[:-1]
            else:
                p_pred = ((s0+1)**L)*m**(L)
                xx= np.logspace(np.log10(int(0.1*p_pred)),np.log10(10*p_pred),10)
                xx1 = np.logspace(np.log10(10*p_pred),np.log10(100*p_pred),10)
                xx = np.concatenate((xx,xx1))
                
                #xx= np.logspace(np.log10(p_pred),np.log10(500*p_pred),15)
                xx =xx[:-1]
                #xx1 = np.array([107999,135917])
                #xx = np.concatenate((xx,xx1))
        if s>2:
            p_pred = ((s0+1)**L)*m**(L)
            
            xx= np.logspace(np.log10(int(0.1*p_pred)),np.log10(10*p_pred),10)
            xx1 = np.logspace(np.log10(10*p_pred),np.log10(100*p_pred),10)
            xx = np.concatenate((xx,xx1))
    else:
        if s>2: 
            p_pred = n**(L)
            #pmax = ((2*s0+2)**(2**L -1))*n**(2**L)


            if s0 <=4:
                xx= np.logspace(np.log10(int(p_pred)),np.log10(100*p_pred),15)
                xx =xx[:-1]
            else:
                xx= np.logspace(np.log10(p_pred),np.log10(500*p_pred),15)
                xx =xx[:-1]
                #xx1 = np.array([107999,135917])
                #xx = np.concatenate((xx,xx1))

            if s==3 or s==4:
                if L==2:
                    xx1= np.logspace(np.log10(100*p_pred),np.log10(500*p_pred),15)
                    xx = np.concatenate((xx,xx1))
                    xx1= np.logspace(np.log10(500*p_pred),np.log10(1000*p_pred),15)
                    xx = np.concatenate((xx,xx1))
                if s==4:
                    xx1= np.logspace(np.log10(1000*p_pred),np.log10(2000*p_pred),10)                            
                    xx = np.concatenate((xx,xx1))
                    xx1= np.logspace(np.log10(2000*p_pred),np.log10(10000*p_pred),10)                            
                    xx = np.concatenate((xx,xx1))
                    xx1= np.logspace(np.log10(10000*p_pred),np.log10(50000*p_pred),10)                            
                    xx = np.concatenate((xx,xx1))
                if s==3 and s0==4:
                    xx1= np.logspace(np.log10(int(1000*p_pred)),np.log10(10000*p_pred),10)
                    xx = np.concatenate((xx,xx1))
        else:
            p_pred = n**(L+1)
            #pmax = ((2*s0+2)**(2**L -1))*n**(2**L)
            if L>1:
                if s0>0:
                    if L>=3:
                        pmax = 110000

                        if s0 <4:
                            xx= np.logspace(np.log10(p_pred),np.log10(100*p_pred),15)

                        else:
                            xx= np.logspace(np.log10(p_pred),np.log10(300*p_pred),15)
                        if s0==1:
                            if n==8 or n==10:
                                xx0 = np.logspace(np.log10(int(0.1*p_pred)),np.log10(p_pred),5)
                                xx0 = xx0[:-1]
                                xx = np.concatenate((xx0,xx))
                    elif L==2:
                        pmax = ((2*s0+2)**(2**L -1))*n**(2**L)
                        p_pred = n**(L+1)
                        p_max_used = min(pmax,110000)
                        #p_pred = n**(L+1)
                        if s0>=4:
                            xx= np.logspace(np.log10(p_pred),np.log10(p_max_used),15)
                        else:
                            xx= np.logspace(np.log10(p_pred),np.log10(pmax),15)

                else:
                    p_pred_red = n**L
                    xx= np.logspace(np.log10(int(0.2*p_pred_red)),np.log10(100*p_pred_red),15)
            else:
                p_pred_1 = n**L
                if s0 <4:
                    xx= np.logspace(np.log10(p_pred_1),np.log10(200*p_pred_1),15)

                else:
                    xx= np.logspace(np.log10(p_pred_1),np.log10(500*p_pred_1),15)
            if s==3:
                p_pred = (s0+1)*m**(L)
                #p_max_used = min(pmax,110000)
                #p_pred = n**(L+1)
                if s0 <= 4:
                    #xx1= np.logspace(np.log10(int(1000*p_pred)),np.log10(2000*p_pred),10)
                    #xx = np.concatenate((xx,xx1))
                    xx= np.logspace(np.log10(int(0.1*p_pred)),np.log10(10*p_pred),15)
            xx =xx[:-1]
    return xx

def load_net(net,ptr,n,m,L,s0,s,seed,width,lr,type_diffeo):
    #type_diffeo = 'A'
    if net =='fcn2':
        print("fcn2_data/hier1_w_0_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")
        tmp = load_file("fcn2_data/hier1_w_0_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")
    
    
    elif 'VGG' in net:
        
        tmp= load_file("new_nets/hier1_"+net+"_"+type_diffeo+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_lr_"+str(lr)+".npy")
        
    elif 'ResNet' in net:
        tmp= load_file("new_nets/hier1_"+net+"_"+type_diffeo+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_lr_"+str(lr)+".npy")
    elif 'Efficient' in net:
        print("new_nets/hier1_"+net+"_"+type_diffeo+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_lr_"+str(lr)+".npy")
        tmp= load_file("new_nets/hier1_"+net+"_"+type_diffeo+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_lr_"+str(lr)+".npy")
        
    elif net=='lcn':
        if s>2:
            #net+"_data_L"+str(2)+"_A_0s
            tmp = load_file("lcn_s_L2/hier1_w_0_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")
            #if L>3:
            #    tmp = load_file(net+"_data_L"+str(3)+"_A_0s/hier1_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_L_"+str(L)+"_s0_"+str(s0)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")
        else:
            if L==1:
                #print(ptr)
                tmp = load_file("lcn_L1/hier1_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(2)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")

            else:
                
                if s0<4:
                    tmp = load_file(net+"_data_L"+str(2)+"_A_0s/hier1_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_L_"+str(L)+"_s0_"+str(s0)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")
                else:
                    tmp = load_file(net+"_data_L"+str(2)+"_A_0s/hier1_w_0_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")
    elif net=='cnn2':                       
        if s>2:
            if ptr>= 176055 and s==4:
                lr_tmp = 0.005
                tmp = load_file("cnn_s_max/hier1_w_0_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr_tmp)+".npy")
            else:
                tmp = load_file("cnn_s_max/hier1_w_0_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")
        else:
            if L==1:
            #print(ptr)
                tmp = load_file("cnn_L1/hier1_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_L_"+str(L)+"_s0_"+str(s0)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")

            else:
                if s0==0:
                    tmp = load_file("check_s0/hier1_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_L_"+str(L)+"_s0_"+str(s0)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")

                else:
                    if L<=3:
                        tmp = load_file(net+"_data_L"+str(L)+"_A_0s/hier1_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_L_"+str(L)+"_s0_"+str(s0)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")
                    if L>3:
                        tmp = load_file(net+"_data_L"+str(3)+"_A_0s/hier1_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_L_"+str(L)+"_s0_"+str(s0)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)+".npy")
    return tmp


def training_point_new(xx,net,n,m,L,s0,s,width,lr,type_diffeo):
    xx_new = []
                
    for ptrx in xx:

        #print(ptrx,pmax)
        ptr = int(ptrx) 


        tmp_seed = 0

        num_seeds = 0
        for (idx_seed,seed) in enumerate(np.array([1])):

            try:
            
                tmp = load_net(net,ptr,n,m,L,s0,s,seed,width,lr,type_diffeo)

                num_seeds +=1
                xx_new.append(ptr)
            except:
                print("new_nets/hier1_"+net+"_"+type_diffeo+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_lr_"+str(lr)+".npy")
                pass
    xx_new = torch.tensor(xx_new)
    return xx_new


from models import*
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--net", type=str, default="cnn2")
parser.add_argument("--n", type=int, default="4")
parser.add_argument("--m", type=int, default="4")
parser.add_argument("--s0", type=int, default="0")
parser.add_argument("--s", type=int, default="2")
parser.add_argument("--L", type=int, default="2")
parser.add_argument("--bs", type=int, default="20")
parser.add_argument("--output", type=str, required=False, default="None")
args = parser.parse_args()




net = args.net
n = args.n
L = args.L
s =args.s
type_diffeo= 'A'

input_dim = (s*(args.s0))**L

m =args.m
seed = 1
width = 512
s0 = args.s0
bs = args.bs


print('BS: '+str(bs))

if 'VGG' in net:
    lr = 0.0001
elif 'ResNet' in net:
    lr = 0.0001
elif 'EfficientNet' in net:
    lr = 0.0001
else:
    lr = choice_lr(net,s,s0,L)

xx = training_point(net,s,s0,L,n,m, type_diffeo)
if L==3:
    p_pred = n**(L)
    xx0= np.logspace(np.log10(int(200*p_pred)),np.log10(600*p_pred),6)
    if s0==1 or s0==2:
        xx = np.concatenate((xx,xx0))
        if n==8 and s0==2:
            xx = xx[xx<=50000] 
if L==2:
    width = 512
else:
    width = 256
xx_new = training_point_new(xx,net,n,m,L,s0,s,width,lr, type_diffeo)

print('xx_new')
print(xx_new) 

xx_new = torch.tensor(xx_new)
len_xx = len(xx_new) 
print(xx_new)



for (idx_p,ptrc) in enumerate(xx_new):

    ptr = int(ptrc)   
    print(ptr)

    tmp = load_net(net,ptr,n,m,L,s0,s,seed,width,lr, type_diffeo)
            
    state = tmp['best']['net']
    state = remove_module_state(state)
    
    args = tmp['args']
    args.device = 'cpu'
    args.whitening = 1
    args.input_format = 'all0'
    if idx_p==0:
        net1 = model_initialization(args, input_dim=input_dim, ch=n, s=s)
        net1.load_state_dict(state)
        net1.to(args.device) 
        nodes, _ = get_graph_node_names(net1)
        if 'VGG11' in args.net:
            nodes = ['features.%d'%(i) for i in [2,6,10,13,17,20,24,27]]
        elif 'VGG16' in args.net:
            nodes = ['features.%d'%(i) for i in [2,5,9,12,16,19,22,26,29,32,36,39,42]]   
        elif 'ResNet' in args.net:
            nodes = [item for item in nodes if 'relu' in item]
        elif 'Efficient' in args.net:
            nodes = [item for item in nodes if 'sigmoid' in item]    
        
        num_layers = len(nodes)
               
    
        mat_all_sem  = torch.zeros((len_xx,num_layers ,L))
        mat_all_pos = torch.zeros((len_xx,num_layers ,L))
        mat_terr = torch.zeros(len_xx)    
    
    if net=='cnn2' or net=='lcn' or net =='fcn2':

        mat, norms = sem_state2permutation_stability(state, args,s,bs)    
        mat_all_sem[idx_p,0,:] = mat['hier.1']
        for l in range(1,L):
            mat_all_sem[idx_p,l,:] = mat['hier.%d.1' %(l+1)] 
        mat_all_sem [idx_p,L,:] = mat['truediv']
        '''
        mat, norms = gauss_state2permutation_stability(state, args,s,bs)    
        mat_all_gauss[idx_p,0,:] = mat['hier.1']
        for l in range(1,L):
            mat_all_gauss[idx_p,l,:] = mat['hier.%d.1' %(l+1)] 
        mat_all_gauss[idx_p,L,:] = mat['truediv']
        '''
        mat, norms  = pos_state2permutation_stability(state, args,s,bs)  
        mat_all_pos[idx_p,0,:] = mat['hier.1']
        #mat_norms[idx_p,0,:] = norms['hier.1']
        for l in range(1,L):
            mat_all_pos[idx_p,l,:] = mat['hier.%d.1' %(l+1)] 
            #mat_norms[idx_p,l,:] = norms['hier.%d.1' %(l+1)] 
        mat_all_pos[idx_p,L,:] = mat['truediv']
        #mat_norms[idx_p,L,:] = norms['truediv'] 
    
        mat_terr[idx_p] = (100-tmp['best']["acc"])/100
    else:
        for (idxnode,node) in enumerate(nodes):
            mat, norms = sem_state2permutation_stability(state, args,s,bs)   
            print(idxnode,node)
            mat_all_sem[idx_p,idxnode,:] = mat[node]

            mat, norms  = pos_state2permutation_stability(state, args,s,bs)  
            mat_all_pos[idx_p,idxnode,:] = mat[node]
        mat_terr[idx_p] = (100-tmp['best']["acc"])/100
        
torch.save(mat_terr,'mats/mat_terr_L_%d_n_%d_m_%d_s0_%d_s_%d_bs_%d_%s.pt'%(L,n,m,s0,s,bs, net))
torch.save(mat_all_sem,'mats/mat_sem_L_%d_n_%d_m_%d_s0_%d_s_%d_bs_%d_%s.pt'%(L,n,m,s0,s,bs, net))
torch.save(mat_all_pos,'mats/mat_pos_L_%d_n_%d_m_%d_s0_%d_s_%d_bs_%d_%s.pt'%(L,n,m,s0,s,bs, net))
torch.save(xx_new,'mats/xx_L_%d_n_%d_m_%d_s0_%d_s_%d_bs_%d_%s.pt'%(L,n,m,s0,s,bs, net))

#fixed P
#each row is for a different representation layer. going from lower to upper layer
#each column is for permuting different levels of data hierarchy. from left to right you permute from lower to upper ii
