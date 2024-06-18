#!/bin/bash
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH -p batch-AMD
#SBATCH --time=40:00:00

source ~/.bashrc
conda activate ilumpy

ipaddress=$(ip addr | grep 172 | awk 'NR==1{print $2}' | sed 's!/23!!g' | sed 's!/0!!g')
echo $ipaddress

python3 deep_learning_cancer_500.py

#jupyter-notebook --ip=$ipaddress

