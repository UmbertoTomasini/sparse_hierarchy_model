"""
    Train networks on 1d hierarchical models of data.
"""

import os
import argparse
import time
import math
import pickle
from models import *
import copy
from functools import partial


from init_bs import init_fun
from optim_loss import loss_func, regularize, opt_algo, measure_accuracy
from utils import cpu_state_dict
from observables import locality_measure
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load the dataset part from the file at the given index
        dataset_part = torch.load(self.file_paths[index])

        # Perform any preprocessing on the dataset part if needed

        # Return the dataset part
        return dataset_part.x, dataset_part.targets

def run(args):

    best_acc = 0  # best test accuracy
    criterion = partial(loss_func, args)

    net0 = init_fun(args)

    # scale batch size when smaller than train-set size
    if (args.batch_size <= args.ptr) and args.scale_batch_size:
        args.batch_size = args.ptr // 2

    if args.save_dynamics:
        dynamics = [{"acc": 0.0, "epoch": 0., "net": cpu_state_dict(net0)}]
    else:
        dynamics = None

    loss = []
    terr = []
    locality = []
    epochs_list = []

    best = dict()
    trloss_flag = 0

    for net, epoch, losstr, avg_epoch_time in train(args, net0, criterion):

        assert str(losstr) != "nan", "Loss is nan value!!"
        loss.append(losstr)
        epochs_list.append(epoch)



        # avoid computing accuracy each and every epoch if dataset is small and epochs are rescaled
        # if epoch > 250:
        #     if epoch % (args.epochs // 250) != 0:
        #         continue

        if epoch % 10 != 0 and not args.save_dynamics: continue

        acc = test(args, net, criterion, print_flag=epoch % 5 == 0)
        terr.append(100 - acc)

        if args.save_dynamics:
        #     and (
        #     epoch
        #     in (10 ** torch.linspace(-1, math.log10(args.epochs), 30)).int().unique()
        # ):
            # save dynamics at 30 log-spaced points in time
            dynamics.append(
                {"acc": acc, "epoch": epoch, "net": cpu_state_dict(net)}
            )
        if acc > best_acc:
            best["acc"] = acc
            best["epoch"] = epoch
            if args.save_best_net:
                best["net"] = cpu_state_dict(net)
            # if args.save_dynamics:
            #     dynamics.append(best)
            best_acc = acc
            print(f"BEST ACCURACY ({acc:.02f}) at epoch {epoch:.02f} !!", flush=True)

        out = {
            "args": args,
            "epoch": epochs_list,
            "train loss": loss,
            "terr": terr,
            "locality": locality,
            "dynamics": dynamics,
            "best": best,
        }

        yield out

        if (losstr == 0 and args.loss == 'hinge') or (losstr < args.zero_loss_threshold and args.loss == 'cross_entropy'):
            trloss_flag += 1
            if trloss_flag >= args.zero_loss_epochs:
                break

    try:
        wo = weights_evolution(net0, net)
    except:
        print("Weights evolution failed!")
        wo = None

    out = {
        "args": args,
        "epoch": epochs_list,
        "train loss": loss,
        "terr": terr,
        "locality": locality,
        "dynamics": dynamics,
        "init": cpu_state_dict(net0) if args.save_init_net else None,
        "best": best,
        "last": cpu_state_dict(net) if args.save_last_net else None,
        "weight_evo": wo,
        'avg_epoch_time': avg_epoch_time,
    }
    yield out


def train(args, net0, criterion):
    
    net = copy.deepcopy(net0)

    optimizer, scheduler = opt_algo(args, net)
    print(f"Training for {args.epochs} epochs...")

    start_time = time.time()

    num_batches = math.ceil(args.ptr / args.batch_size)
    checkpoint_batches = torch.linspace(0, num_batches, 10, dtype=int)
    if args.type =='A':
        if args.s==2:
            st = 'dataset_creating/storing_all0'
            if args.num_layers ==4 and args.num_features != 4:
                st = 'dataset_creating/storing_all0_L4'
            elif args.num_layers==4 and args.num_features == 4:
                st = 'dataset_creating/storing_all0_L4_n4'
            if args.s0==0:
                st = 'dataset_creating/storing_all0_s0_0'
            
            if args.num_features==6 and args.m==6 and args.s0>0:
                file_paths = [st+'/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d.pt'%(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,args.s,idx_bs) for idx_bs in range(num_batches)]
            else:
                file_paths = [st+'/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_idx_%d.pt'%(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,args.s,idx_bs) for idx_bs in range(num_batches)]
        else:
            st = 'dataset_creating/storing_all0_ss_max'
            file_paths = [st+'/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d.pt' %(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,args.s,idx_bs) for idx_bs in range(num_batches)]
    
    else:
        st = 'dataset_creating/storing_diffeoB'
        file_paths = [st+'/dataHier_B_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d_seed_%d.pt'%(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,args.s,idx_bs,args.seed_init) for idx_bs in range(num_batches)]
    
    dataset = MyDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):          #, (inputs, targets) in enumerate(trainloader):
            #start_time = time.time()
            #trainset_b = torch.load('dataset_creating/storing_all0/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_idx_%d.pt'%(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,batch_idx)) 
            #inputs = trainset_b.x.to(args.device)
            #targets = trainset_b.targets.to(args.device)
            inputs = inputs.squeeze()
            targets = targets.squeeze()
            #print('b_idx: '+str(batch_idx))
            #print('targets_size: '+str(targets.size(0)))
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            train_loss += loss.detach().item()
            regularize(loss, net, args.weight_decay, reg_type=args.reg_type)
            loss.backward()
            optimizer.step()
            #print('time for batch: '+str(batch_idx)+" is: "+str(time.time()-start_time))
            correct, total = measure_accuracy(args, outputs, targets, correct, total)
              
            # during first epoch, save some sgd steps instead of after whole epoch
            if epoch < 10 and batch_idx in checkpoint_batches and batch_idx != (num_batches - 1):
                yield net, epoch + (batch_idx + 1) / num_batches, train_loss / (batch_idx + 1), None

        avg_epoch_time = (time.time() - start_time) / (epoch + 1)

        if epoch % 5 == 0:
            print(
                f"[Train epoch {epoch+1} / {args.epochs}, {print_time(avg_epoch_time)}/epoch, ETA: {print_time(avg_epoch_time * (args.epochs - epoch - 1))}]"
                f"[tr.Loss: {train_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]",
                flush=True
            )

        scheduler.step()

        yield net, epoch + 1, train_loss / (batch_idx + 1), avg_epoch_time


def test(args, net, criterion, print_flag=True):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        num_batches_te = math.ceil(args.pte / args.batch_size)
        p_pred = args.num_classes*args.m**(args.num_layers)
        #num_batches_tr = math.ceil(args.ptr / args.batch_size)
        #max_num = int(300*p_pred + 50000)
        
        if net =='lcn':
            max_num = int(100*(args.s0+1)*p_pred/args.num_classes)
            num_batches = math.ceil(max_num / args.batch_size)
        elif net=='cnn':
            max_num = int((300*p_pred + 50000))  
            num_batches = math.ceil(max_num / args.batch_size)
        else:
            max_num = int((300*p_pred + 50000))  
            num_batches = math.ceil(max_num / args.batch_size)
            
        #to correct bug with lcn
        if net=='lcn':
            if args.num_classes==4 and args.num_layers==2 and args.s==2 and args.s0==4:
                max_num = int((300*p_pred + 50000))
                num_batches = math.ceil(max_num / args.batch_size)
            if args.num_classes==4 and args.num_layers==2 and args.s==2 and args.s0==4:
                max_num = int((300*p_pred + 50000))
                num_batches = math.ceil(max_num / args.batch_size)

        #not everything is created
        if args.num_classes==6 and args.num_layers==2 and args.s==4:
            if args.s0==0:
                num_batches = 1504342
            elif args.s0==1:
                num_batches = 876340
            elif args.s0==2:
                num_batches = 527856
            elif args.s0==4:
                num_batches = 236318
        if args.num_classes==6 and args.num_layers==3 and args.s==3:
            if args.s0==0:
                num_batches = 279450
            elif args.s0==1:
                num_batches = 56998
            elif args.s0==2 or args.s0==4:
                num_batches = 20000    
        if args.num_classes==4 and args.num_layers==3 and args.s==3:
            if args.s0==0:
                num_batches = 1241297
            elif args.s0==1:
                num_batches = 444265
            elif args.s0==2:
                num_batches = 224862
            elif args.s0==4:
                num_batches = 56980
        #elif args.s0==2:
            #    num_batches = 527856
            #elif args.s0==4:
            #    num_batches = 236318
        if args.type =='B':
            if args.num_layers==3:
                if args.s0==2:
                    num_batches = 533850
                elif args.s0==4:
                    num_batches = 161709

        a0 = num_batches - num_batches_te
        #if args.L==4 and args.num_features==4 and args.s0==4:
        #    a0 = 67000 - num_batches_te
        a1 = num_batches
        
        if args.type =='A':
            if args.s==2:
                st = 'dataset_creating/storing_all0'
                if args.num_layers ==4 and args.num_features != 4:
                    st = 'dataset_creating/storing_all0_L4'
                elif args.num_layers==4 and args.num_features == 4:
                    st = 'dataset_creating/storing_all0_L4_n4'
                if args.s0==0:
                    st = 'dataset_creating/storing_all0_s0_0'
                if args.num_features==6 and args.m==6 and args.s0>0:

                    file_paths = [st+'/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d.pt'%(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,args.s,idx_bs) for idx_bs in range(a0,a1)]
                else:
                    file_paths = [st+'/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_idx_%d.pt'%(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,args.s,idx_bs) for idx_bs in range(a0,a1)]
            else:
                st = 'dataset_creating/storing_all0_ss_max'
                file_paths = [st+'/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d.pt' %(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,args.s,idx_bs) for idx_bs in range(a0,a1)]
        
        else:
            st = 'dataset_creating/storing_diffeoB'

            file_paths = [st+'/dataHier_B_n_%d_m_%d_nc_%d_L_%d_s0_%d_s_%d_idx_%d_seed_%d.pt'%(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,args.s,idx_bs,args.seed_init) for idx_bs in range(a0,a1)]
        dataset = MyDataset(file_paths)
        testloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
        
        for batch_idx, (inputs, targets) in enumerate(testloader):  
        
            inputs = inputs.squeeze()
            targets = targets.squeeze()
            #trainset_b = torch.load('dataset_creating/storing_all0/dataHier_n_%d_m_%d_nc_%d_L_%d_s0_%d_idx_%d.pt'%(args.num_features,args.m,args.num_classes,args.num_layers,args.s0,batch_idx))
            #inputs = trainset_b.x.to(args.device)
            #targets = trainset_b.targets.to(args.device)

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()

            correct, total = measure_accuracy(args, outputs, targets, correct, total)
            #print("total "+str(total)+" at batch_idx "+str(batch_idx))
        if print_flag:
            print(
                f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]",
                flush=True
            )

    return 100.0 * correct / total


# timing function
def print_time(elapsed_time):

    # if less than a second, print milliseconds
    if elapsed_time < 1:
        return f"{elapsed_time * 1000:.00f}ms"

    elapsed_seconds = round(elapsed_time)

    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)

    elapsed_time = []
    if h > 0:
        elapsed_time.append(f"{h}h")
    if not (h == 0 and m == 0):
        elapsed_time.append(f"{m:02}m")
    elapsed_time.append(f"{s:02}s")

    return "".join(elapsed_time)


def weights_evolution(f0, f):
    s0 = f0.state_dict()
    s = f.state_dict()
    nd = 0
    for k in s:
        nd += (s0[k] - s[k]).norm() / s0[k].norm()
    nd /= len(s)
    return nd


def main():

    parser = argparse.ArgumentParser()

    ### Tensors type ###
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float64")

    ### Seeds ###
    parser.add_argument("--seed_init", type=int, default=0)
    parser.add_argument("--seed_net", type=int, default=-1)
    parser.add_argument("--seed_trainset", type=int, default=-1)

    ### DATASET ARGS ###
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=float, default=0.8,
        help="Number of training point. If in [0, 1], fraction of training points w.r.t. total.",
    )
    parser.add_argument("--pte", type=float, default=.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scale_batch_size", type=int, default=0)

    parser.add_argument("--background_noise", type=float, default=0)

    # Hierarchical dataset #
    parser.add_argument("--num_features", type=int, default=8)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--input_format", type=str, default="onehot")
    parser.add_argument("--whitening", type=int, default=0)
    parser.add_argument("--auto_regression", type=int, default=0)
    parser.add_argument("--s0", type=int, default=0)
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--type", type=str, default='A')
    
    ### ARCHITECTURES ARGS ###
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--random_features", type=int, default=0)

    ## Nets params ##
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--net_layers", type=int, default=3)
    parser.add_argument("--filter_size", type=int, default=2)
    parser.add_argument("--pooling_size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument("--bias", type=int, default=1, help="for some archs, controls bias presence")
    parser.add_argument("--pbc", type=int, default=0, help="periodic boundaries cnn")
    parser.add_argument("--sharing", type=int, default = 0,help="which layers do implement (s_0+1) sharing")
    parser.add_argument("--pretrained", type=int, default=0)
    
    ## Auto-regression with Transformers ##
    parser.add_argument("--pmask", type=float, default=.2)


    ### ALGORITHM ARGS ###
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--scheduler", type=str, default="cosineannealing")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--reg_type", default='l2', type=str)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--zero_loss_epochs", type=int, default=0)
    parser.add_argument("--zero_loss_threshold", type=float, default=0.01)
    parser.add_argument("--rescale_epochs", type=int, default=0)

    parser.add_argument(
        "--alpha", default=1.0, type=float, help="alpha-trick parameter"
    )

    ### SAVING ARGS ###
    parser.add_argument("--save_init_net", type=int, default=1)
    parser.add_argument("--save_best_net", type=int, default=1)
    parser.add_argument("--save_last_net", type=int, default=1)
    parser.add_argument("--save_dynamics", type=int, default=0)

    ## saving path ##
    parser.add_argument("--pickle", type=str, required=False, default="None")
    parser.add_argument("--output", type=str, required=False, default="None")
    args = parser.parse_args()
    print(args.sharing) ##sharings check
    print(args.type)
    if args.type== 'A':
        from hierarchical_s0_bs_S import HierarchicalDataset
    elif args.type== 'B':
        from hierarchical_s0_bs_S_diffeoB import HierarchicalDataset
        
    if args.pickle == "None":
        assert (
            args.output != "None"
        ), "either `pickle` or `output` must be given to the parser!!"
        args.pickle = args.output

    # special value -1 to set some equal arguments
    if args.seed_trainset == -1:
        args.seed_trainset = args.seed_init
    if args.seed_net == -1:
        args.seed_net = args.seed_init
    if args.num_classes == -1:
        args.num_classes = args.num_features
    if args.net_layers == -1:
        args.net_layers = args.num_layers
    if args.m == -1:
        args.m = args.num_features

    # define train and test sets sizes
    #if args.num_features< 10:

    Pmax = (((args.s0+1)**args.s)*args.m) ** (2 ** args.num_layers - 1) * args.num_classes

    if 0 < args.pte <= 1:
        args.pte = int(args.pte * Pmax)
    elif args.pte == -1:
        if args.num_features<10:
            args.pte = min(Pmax // 5, 20000)
        else:
            args.pte = 20000
    else:
        args.pte = int(args.pte)
    #print('pts test: '+str(args.pte))
    if args.ptr >= 0:
        if args.ptr <= 1:
            args.ptr = int(args.ptr * Pmax)
        else:
            args.ptr = int(args.ptr)
        assert args.ptr > 0, "relative dataset size (P/Pmax) too small for such dataset!"
    else:
        args.ptr = int(- args.ptr * args.m ** (args.num_layers) * args.num_features)

    args.pte = min(Pmax - args.ptr, args.pte)
    print('pts test: '+str(args.pte))
    with open(args.output, "wb") as handle:
        pickle.dump(args, handle)
    try:
        for data in run(args):
            with open(args.output, "wb") as handle:
                pickle.dump(args, handle)
                pickle.dump(data, handle)
    except:
        os.remove(args.output)
        raise


if __name__ == "__main__":
    main()
