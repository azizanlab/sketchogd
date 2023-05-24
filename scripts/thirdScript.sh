#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2022a

# SEED 1:

# SEED 2:


# rotated mnist
#python3 main.py --method sogd --memory_size 1200 --dataset rotated --sketch_per_task 480 --sketch_method basic --seed 1
#python3 main.py --method sogd --memory_size 600 --dataset rotated --sketch_per_task 480 --sketch_method lowrankapprox --seed 1
#python3 main.py --method sogd --memory_size 300 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 1
#python3 main.py --method pca --memory_size 120 --dataset rotated --sketch_per_task 480 --pca_sample 480 --sketch_method lowranksym --seed 1
#python3 main.py --method ogd --memory_size 120 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 1
#python3 main.py --method sgd --memory_size 120 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 1
#python3 main.py --method ogd --memory_size 480 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 1
#
## permuted mnist
#python3 main.py --method sogd --memory_size 1200 --dataset permuted --sketch_per_task 480 --sketch_method basic --seed 1
#python3 main.py --method sogd --memory_size 600 --dataset permuted --sketch_per_task 480 --sketch_method lowrankapprox --seed 1
#python3 main.py --method sogd --memory_size 300 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 1
#python3 main.py --method pca --memory_size 120 --dataset permuted --sketch_per_task 480 --pca_sample 480 --sketch_method lowranksym --seed 1
#python3 main.py --method ogd --memory_size 120 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 1
#python3 main.py --method sgd --memory_size 120 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 1
#python3 main.py --method ogd --memory_size 480 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 1
#
## split mnist
#python3 main.py --method sogd --memory_size 1200 --dataset split_mnist --sketch_per_task 960 --sketch_method basic --seed 1
#python3 main.py --method sogd --memory_size 600 --dataset split_mnist --sketch_per_task 960 --sketch_method lowrankapprox --seed 1
#python3 main.py --method sogd --memory_size 300 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 1
#python3 main.py --method pca --memory_size 240 --dataset split_mnist --sketch_per_task 960 --pca_sample 960 --sketch_method lowranksym --seed 1
#python3 main.py --method ogd --memory_size 240 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 1
#python3 main.py --method sgd --memory_size 240 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 1
#python3 main.py --method ogd --memory_size 960 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 1
#
## rotated
#python3 main.py --method pca --memory_size 100 --dataset rotated --sketch_per_task 200 --pca_sample 200 --sketch_method lowranksym --seed 1
#python3 main.py --method pca --memory_size 100 --dataset rotated --sketch_per_task 200 --pca_sample 200 --sketch_method lowranksym --seed 2
#python3 main.py --method pca --memory_size 100 --dataset rotated --sketch_per_task 200 --pca_sample 200 --sketch_method lowranksym --seed 3
#
#
## permuted
#python3 main.py --method pca --memory_size 100 --dataset permuted --sketch_per_task 200 --pca_sample 200 --sketch_method lowranksym --seed 1
#python3 main.py --method pca --memory_size 100 --dataset permuted --sketch_per_task 200 --pca_sample 200 --sketch_method lowranksym --seed 2
#python3 main.py --method pca --memory_size 100 --dataset permuted --sketch_per_task 200 --pca_sample 200 --sketch_method lowranksym --seed 3
#
#
##split
#python3 main.py --method pca --memory_size 180 --dataset split_mnist --sketch_per_task 300 --pca_sample 300 --sketch_method lowranksym --seed 1
#python3 main.py --method pca --memory_size 180 --dataset split_mnist --sketch_per_task 300 --pca_sample 300 --sketch_method lowranksym --seed 2
#python3 main.py --method pca --memory_size 180 --dataset split_mnist --sketch_per_task 300 --pca_sample 300 --sketch_method lowranksym --seed 3

# calc singular values
#python3 main.py --method ogd --memory_size 480 --dataset rotated --compute_singular_values True --seed 1
#python3 main.py --method pca --memory_size 120 --pca_sample 10000 --sketch_per_task 10000 --sketch_method lowranksym --dataset rotated --compute_singular_values True --nepoch 30 --seed 1
#python3 main.py --method pca --memory_size 120 --pca_sample 10000 --sketch_per_task 10000 --sketch_method basic --dataset rotated --compute_singular_values True --nepoch 30 --seed 1 --n_tasks 10 --rotate_step 5

#python3 main.py --method ogd --memory_size 960 --dataset split_mnist --compute_singular_values True --seed 1

#python3 main.py --method ogd --memory_size 480 --dataset rotated --compute_singular_values True --seed 2
#python3 main.py --method ogd --memory_size 960 --dataset split_mnist --compute_singular_values True --seed 2

#python3 main.py --method ogd --memory_size 480 --dataset rotated --compute_singular_values True --seed 3
#python3 main.py --method ogd --memory_size 960 --dataset split_mnist --compute_singular_values True --seed 3

# Test AGEM Rotated

python3 main.py --method agem --memory_size 120 --agem_mem_batch_size 120 --dataset rotated --hidden_dim 100 --nepoch 30 --seed 1 --n_tasks 10
python3 main.py --method agem --memory_size 120 --agem_mem_batch_size 120 --dataset rotated --hidden_dim 100 --nepoch 30 --seed 2 --n_tasks 10
python3 main.py --method agem --memory_size 120 --agem_mem_batch_size 120 --dataset rotated --hidden_dim 100 --nepoch 30 --seed 3 --n_tasks 10




# 5 Tasks Rotated:
# 200 total gradients for OGD, 40 per task
# 200 data points, 40 per task, average 40 gradients per task
# Compress 50 down to 30 for pca
# Compress 200 down to 40 for pca
# 200 for sogd1 basic, compress from 10,000
# 50 for sogd3 lowranksym, compress from 10,000


#python3 main.py --method agem --memory_size 40 --agem_mem_batch_size 200 --n_tasks 5 --rotate_step 10 --dataset rotated --seed 1 --subset_size 10000 --nepoch 5
#python3 main.py --method agem --memory_size 40 --agem_mem_batch_size 40 --n_tasks 5 --rotate_step 10 --dataset rotated --seed 1 --subset_size 10000 --nepoch 5

# Split CIFAR
#python3 main.py --method sogd --memory_size 600 --dataset split_cifar --hidden_dim 500 --sketch_per_task 10000 --sketch_method lowrankapprox --nepoch 30 --n_tasks 20 --seed 3

# SPLIT CIFAR AGEM:
#python3 main.py --method agem --memory_size 60 --agem_mem_batch_size 1200 --n_tasks 20 --dataset split_cifar --seed 2 --nepoch 30
#python3 main.py --method agem --memory_size 60 --agem_mem_batch_size 1200 --n_tasks 20 --dataset split_cifar --seed 5 --nepoch 30




