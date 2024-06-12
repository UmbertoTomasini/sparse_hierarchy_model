import torch

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

#from .hierarchical_s0_gauss import HierarchicalDataset
from models import model_initialization

def x_reduce(x,bs):
    factor = 100
    #print(x.shape)
    x = x.reshape(int(x.shape[0]/factor),100,x.shape[1],x.shape[2])
    x = x[:,:bs,:,:]
    #print(x.shape)
    x = x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3])
    #print(x.shape)
    return x


def sem_state2permutation_stability(state, args,s,bs):
    
    x = torch.load("dataset_creating/storing_sem/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_0.pt"%(args.num_features, args.m, args.num_classes,args.num_layers, args.s0,s))
    #build_sem_permuted_datasets(args)
    x = x_reduce(x,bs)
    _, ch, input_dim = x.shape
    l = state2feature_extractor(state, args, input_dim, ch,s)
    
    with torch.no_grad():
        o = l(x.to(args.device))
        #print(o)
    return measure_stability(o, args)

def pos_state2permutation_stability(state, args,s,bs):

    x = torch.load("dataset_creating/storing_pos/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_0.pt"%(args.num_features, args.m, args.num_classes,args.num_layers, args.s0,s))
    x = x_reduce(x,bs)

    #x = x[:,:bs,:]
    _, ch, input_dim = x.shape
    l = state2feature_extractor(state, args, input_dim, ch,s)

    with torch.no_grad():
        o = l(x.to(args.device))

    return measure_stability(o, args)

def gauss_state2permutation_stability(state, args, s,bs):

    x_tot = torch.load("dataset_creating/storing_pos/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_0.pt"%(args.num_features, args.m, args.num_classes,args.num_layers, args.s0,s))
    x_tot = x_reduce(x_tot,bs)    
    x = x_tot[:bs,:,:]
    x_diff = x_tot[bs:2*bs,:,:]
    
    magnitude = (x-x_diff).pow(2).mean(dim=(1,2)).pow(.5).mean(0)
    magnitude = magnitude.tolist()#[0]
    
    #magnitude= 2*(args.s**args.num_layers)
    #size_noise = x.shape[2] #((args.s0+1)*args.s)**args.num_layers
    mean =0 
    #std_dev = (magnitude/size)**(.5)
    noise = torch.normal(mean=mean, std=magnitude, size=(x.shape[0],x.shape[1],x.shape[2]))
    x_gauss = x + noise
    x = torch.cat((x,x_gauss))
    
    _, ch, input_dim = x.shape
    l = state2feature_extractor(state, args, input_dim, ch,s)

    with torch.no_grad():
        o = l(x.to(args.device))

    return measure_stability_gauss(o, args)




def state2feature_extractor(state, args, input_dim, ch,s):
    
    net = model_initialization(args, input_dim=input_dim, ch=ch, s=s)
    net.load_state_dict(state)
    net.to(args.device) 
    nodes, _ = get_graph_node_names(net)
    #print(nodes)
    #CNN2
    if 'VGG11' in args.net:
        nodes = ['features.%d'%(i) for i in [2,6,10,13,17,20,24,27]] + ['classifier']  
    elif 'VGG16' in args.net:
        nodes = ['features.%d'%(i) for i in [2,5,9,12,16,19,22,26,29,32,36,39,42]] + ['classifier']   
    elif 'ResNet' in args.net:
        nodes = [item for item in nodes if 'relu' in item] +['linear']
    elif 'Efficient' in args.net:
        nodes = [item for item in nodes if 'sigmoid' in item] + ['linear'] 
    else:
        nodes = ['hier.1', 'truediv'] + [f'hier.{i}.1' for i in range(2, args.num_layers + 1)]
        nodes.sort()
    #print(nodes)
    return create_feature_extractor(net, return_nodes=nodes)


def measure_stability_gauss(o, args):
    stability = {}
    norms = {}
    
    for node in o.keys():
        print('Node:')
        print(node)
        print('----------------')

        
        on = o[node].detach()
        on = on.flatten(1)
        #print(on.shape())
        
        on = on.reshape(2, -1, on.shape[1])
        #print('on')
        
        normalization = (on[0][None] - on[0][:, None]).pow(2).sum(dim=-1)
        #/ normalization.mean()
        stability[node] = ((on[0] - on[1:]).pow(2).sum(dim=-1).mean(1)/ normalization.mean() ).cpu()
        #norms[node] = normalization.mean()
    return stability #, norms

def measure_stability(o, args):
    stability = {}
    norms = {}
    
    for node in o.keys():
        print('Node:')
        print(node)
        print('----------------')
        on = o[node].detach()
        on = on.flatten(1)
        #print(on.shape())
        
        on = on.reshape(args.num_layers+1 , -1, on.shape[1])
        #print('on')
        #print(on.shape)
        normalization = (on[0][None] - on[0][:, None]).pow(2).sum(dim=-1)
        #/ normalization.mean()
        stability[node] = ((on[0] - on[1:]).pow(2).sum(dim=-1).mean(1)/ normalization.mean() ).cpu()
        norms[node] = normalization.mean()
    return stability, norms



#--------------------------------------------
'''
def build_pos_permuted_datasets(args):
    x = []
    #PERMUTING SEMANTICALLY
    seed_reset_layer_sem = 42
    
    for seed_reset_layer_pos in range(args.num_layers, -1, -1):
        dataset = HierarchicalDataset(
            num_features=args.num_features,
            m=args.m,  # features multiplicity
            num_layers=args.num_layers,
            num_classes=args.num_features,
            input_format=args.input_format,
            whitening=args.whitening,
            seed=args.seed_init,
            train=True,
            transform=None,
            testsize=0,
            s0 =args.s0,
            seed_reset_layer_sem =seed_reset_layer_sem,
            seed_reset_layer_pos =seed_reset_layer_pos,
            unique_datapoints=0,
            memory_constraint=30 # small here b/c do not need very precise measure
        )
        x.append(dataset.x)
        #print(seed_reset_layer_pos)
        #print(dataset.x)
    return torch.cat(x)

def build_sem_permuted_datasets(args):
    x = []
    #PERMUTING SEMANTICALLY
    seed_reset_layer_pos = 42
    
    for seed_reset_layer_sem in range(args.num_layers, -1, -1):
        dataset = HierarchicalDataset(
            num_features=args.num_features,
            m=args.m,  # features multiplicity
            num_layers=args.num_layers,
            num_classes=args.num_features,
            input_format=args.input_format,
            whitening=args.whitening,
            seed=args.seed_init,
            train=True,
            transform=None,
            testsize=0,
            s0 =args.s0,
            seed_reset_layer_sem =seed_reset_layer_sem,
            seed_reset_layer_pos =seed_reset_layer_pos,
            unique_datapoints=0,
            memory_constraint= 30 # small here b/c do not need very precise measure
        )
        x.append(dataset.x)
        #print(seed_reset_layer_pos)
        #print(dataset.x)
    return torch.cat(x)
'''