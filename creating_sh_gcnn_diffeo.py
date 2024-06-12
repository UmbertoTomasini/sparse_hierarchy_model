import sys
import os
import re
import subprocess
#import h5py
import GooseSLURM as gs
import numpy as np
# ----

#files = sorted(list(filter(None, subprocess.check_output(
 #   "find . -iname 'id*.hdf5'", shell=True).decode('utf-8').split('\n'))))

#d=np.array([2,3,4,5,6,7,11])
#u=np.iarray([0.125,0.25,0.375,0.5,0.75,1,1.25,1.5,2])
#ptrs =inp.arange([0.])
ns=np.array([3,4,6,8,10])
net ="gcnn"
# ----

slurm = '''
# for safety set the number of cores
export OMP_NUM_THREADS=1
'''
# load conda environment
###source ~/miniconda3/etc/profile.d/conda.sh

#if [[ "${{SYS_TYPE}}" == *E5v4* ]]; then
 #   conda activate code_flow_E5v4
#elif [[ "${{SYS_TYPE}}" == *s6g1* ]]; then
 #   conda activate code_flow_s6g1
#str("")
device = "cuda"
s0 =str('#SBATCH --qos gpu \n')
s00 = str('#SBATCH --gres gpu:1 \n')
#s1=str( '#SBATCH --job-name '+ basename + " \n")
#s2=str(  '#SBATCH --out '+ basename + '.out'+ " \n")
s3=str(   '#SBATCH --nodes 1 \n')
s4=str(    '#SBATCH --ntasks-per-node=1 \n')
s5= str('') #str('#SBATCH --gpus-per-task 1 \n') #str(    '#SBATCH --gpus-per-task 1 \n')
s6=str(    '#SBATCH --time 24:00:00 \n')
s7 = str('#SBATCH --mem 32G \n')
#s8=str(   '#SBATCH --partition serial \n')

#{0:s}
#for file in files:

Ls = np.array([3])
for L in Ls:
    for n in ns:
        psc = n**(2**(L-1)+1)
        pmax = n**(2**L)
        p_pred = n**(L+1)
        p1 = np.logspace(np.log10(1*p_pred),np.log10(10*p_pred),10)
        p2 = np.logspace(np.log10(10*p_pred),np.log10(20*p_pred),4)
        for ptrx in np.concatenate((p1,p2)):
            #ptei = 36 
            #ptr = ptrx
            #lr = 0.05
            ptr = int(min(ptrx,pmax))
            pte = min(int(pmax - ptr),int(0.2*pmax))
            #for seed in np.array([1]):
            seed =1
            bs = 8
            for seed in np.array([0]):

                for lr in np.array([0.01]):
                    for width in np.array([512]):

                        basename = "hier1_"+net+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_L_"+str(L)+"_seed_"+str(seed)+"_width_"+str(width)+"_lr_"+str(lr)
                        s1=str( '#SBATCH --job-name '+ basename + " \n")
                        s2=str(  '#SBATCH --out '+ basename + '.out'+ " \n")
                        aa = str("#!/bin/bash \n")+s0+s00+s1+s2+s3+s4+s5+s6+s7+"python3 main.py --device "+str(device)+" --seed_init "+str(seed)+" --pte "+str(-1)+" --ptr "+str(ptr)+" --num_features "+str(n)+" --m "+str(n)+" --num_layers "+str(L)+" --net "+ net+" --filter_size "+str(4)+" --stride 4 --dataset hier1 --width "+str(width)+" --net_layers -1 --lr "+str(lr)+" --weight_decay 0 --input_format onehot --zero_loss_threshold 1e-3 --epochs 3000 --whitening 1 --batch_size "+str(bs)+" --scheduler none --pbc 0 --zero_loss_epochs 0 --num_classes -1 --loss cross_entropy  --output "+str(basename)+".npy"

#+" >> "+str(basename)+".out &"    
        
                        open(basename + '.sh', 'w').write(aa)
        

