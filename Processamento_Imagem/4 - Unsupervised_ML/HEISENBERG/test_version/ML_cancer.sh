#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -p batch-AMD
#SBATCH --time=30:00:00

source ~/.bashrc
conda activate ilumpy

ipaddress=$(ip addr | grep 172 | awk 'NR==1{print $2}' | sed 's!/23!!g' | sed 's!/0!!g')
echo $ipaddress

python3 unsupervised_ML_cancer.py

#jupyter-notebook --ip=$ipaddress

