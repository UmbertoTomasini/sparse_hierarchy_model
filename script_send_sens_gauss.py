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
#ns=np.array([3,4,8,10,12])

#net ="fcn2"
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

nets = ['lcn','cnn2']#['VGG11','VGG16','ResNet18','ResNet34','EfficientNetB0']
ns = np.array([10])
Ls=[2] 
bs = 20
s0s = [2]
sss = [2]
for net in nets:
    for L in Ls:
        for s0 in s0s:
            for s in sss:
                print(s)
        
                for n in ns:
                    for m in [int(n**(s-1))]:
                    #m = int(n**(s-1))
                    #int(0.5*n**(s-1))
                        basename = "send_gauss_"+net+"_n_"+str(n)+"_m_"+str(m)+"_L_"+str(L)+"_s0_"+str(s0)+"_s_"+str(s)
                        s1=str( '#SBATCH --job-name '+ basename + " \n")
                        s2=str(  '#SBATCH --out '+ basename + '.out'+ " \n")
                            
                        aa = str("#!/bin/bash \n")+str0+s00+s1+s2+s3+s4+s5+s6+s7+"python3 sens_s-gauss.py --device "+str(device)+ " --net "+str(net)+" --s "+str(s)+" --n "+str(n)+" --m "+str(m)+" --s0 "+str(s0)+" --L "+str(L)+" --bs "+str(bs)
    
    #+" >> "+str(basename)+".out &"    
            
                        open(basename + '.sh', 'w').write(aa)
        

