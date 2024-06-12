#!/bin/bash
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --job-name hier1_w_0_cnn2_diffeo_ptr_100172_n_6_m_36_L_2_s0_4_s_3_seed_1_width_512_lr_0.003
#SBATCH --out hier1_w_0_cnn2_diffeo_ptr_100172_n_6_m_36_L_2_s0_4_s_3_seed_1_width_512_lr_0.003.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --time 48:00:00
#SBATCH --mem 32G

python sens_s.py
