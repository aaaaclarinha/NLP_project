#!/bin/bash
#SBATCH -n 15
#SBATCH --ntasks-per-node=15
#SBATCH -p batch-AMD
#SBATCH --time=35:00:00

source ~/.bashrc
conda activate ilumpy

ipaddress=$(ip addr | grep 172 | awk 'NR==1{print $2}' | sed 's!/23!!g' | sed 's!/0!!g')
echo $ipaddress

python3 deep_learning_cancer_100.py

#jupyter-notebook --ip=$ipaddress

