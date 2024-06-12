import sys
import os
import re
import subprocess
#import h5py
import GooseSLURM as gs
import numpy as np
from fun_choices import*
# ----

#files = sorted(list(filter(None, subprocess.check_output(
 #   "find . -iname 'id*.hdf5'", shell=True).decode('utf-8').split('\n'))))

#d=np.array([2,3,4,5,6,7,11])
#u=np.iarray([0.125,0.25,0.375,0.5,0.75,1,1.25,1.5,2])
#ptrs =inp.arange([0.])
#ns=np.array([8])
#net ="cnn2"
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
str0 =str('#SBATCH --qos gpu \n')
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
ns=np.array([10])
nets =["EfficientNetB0"]
input_format = 'all0'
Ls = [2]
sss=[2]
s0s = [2]

type_diffeo = 'A'

for net in nets:
    for L in Ls:
        for s in sss:
            for s0 in s0s:
                print(s0)
                for n in ns:
                    m = n**(s-1)
                    xx = training_point(net,s,s0,L,n,m)
            
                    xx = xx[:-1]  
                    #xx = [100]
                    #xx = np.array([i for i in range(440,1500,100)])
                    xx = np.array([590])
                    for ptrx in xx:
                
                        ptr = int(ptrx)
                        pte = -1 #int(ptrx) 
                        
                        for seed in np.arange(1,2,dtype=int):
                            lr = 0.1 #choice_lr(net,s,s0,L)
                            
                            batch_size = 4 
                            m = n**(s-1)   
                            for lr in np.array([0.0001]):
                                
                                #width = 512
                                basename = "hier1_"+net+"_"+type_diffeo+"_diffeo_ptr_"+str(ptr)+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)+"_seed_"+str(seed)+"_lr_"+str(lr)
                                s1=str( '#SBATCH --job-name '+ basename + " \n")
                                s2=str(  '#SBATCH --out '+ basename + '.out'+ " \n")
                                epochs = 250
                                tr = 0.001
                                aa = str("#!/bin/bash \n")+str0+s00+s1+s2+s3+s4+s5+s6+s7+"python3 main_bs_S.py --device "+str(device)+" --seed_init "+str(seed)+" --pte "+str(pte)+" --ptr "+str(ptr)+" --num_features "+str(n)+" --m "+str(m)+" --num_layers "+str(L)+" --s0 "+str(s0)+"  --s "+str(s)+" --type "+str(type_diffeo)+" --net "+ net+" --dataset hier1 --lr "+str(lr)+" --input_format "+ input_format +" --zero_loss_threshold "+ str(tr) +" --epochs " +str(epochs)+" --whitening 1 --batch_size "+str(batch_size)+" --scheduler none --pbc 0 --zero_loss_epochs 0 --num_classes -1 --loss cross_entropy  --output "+str(basename)+".npy"

#+" >> "+str(basename)+".out &"    
         
                                open(basename + '.sh', 'w').write(aa)
        

