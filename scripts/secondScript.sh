#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2022a

# SEED 1:
#python3 main.py --method sogd --memory_size 600 --dataset permuted --sketch_per_task 480 --sketch_method lowrankapprox --seed 1

# SEED 2:


# rotated mnist
#python3 main.py --method sogd --memory_size 1200 --dataset rotated --sketch_per_task 480 --sketch_method basic --seed 2
#python3 main.py --method sogd --memory_size 600 --dataset rotated --sketch_per_task 480 --sketch_method lowrankapprox --seed 2
#python3 main.py --method sogd --memory_size 300 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 2
#python3 main.py --method pca --memory_size 120 --dataset rotated --sketch_per_task 480 --pca_sample 480 --sketch_method lowranksym --seed 2
#python3 main.py --method ogd --memory_size 120 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 2
#python3 main.py --method sgd --memory_size 120 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 2
#python3 main.py --method ogd --memory_size 480 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 2
#
## permuted mnist
#python3 main.py --method sogd --memory_size 1200 --dataset permuted --sketch_per_task 480 --sketch_method basic --seed 2
#python3 main.py --method sogd --memory_size 600 --dataset permuted --sketch_per_task 480 --sketch_method lowrankapprox --seed 2
#python3 main.py --method sogd --memory_size 300 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 2
#python3 main.py --method pca --memory_size 120 --dataset permuted --sketch_per_task 480 --pca_sample 480 --sketch_method lowranksym --seed 2
#python3 main.py --method ogd --memory_size 120 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 2
#python3 main.py --method sgd --memory_size 120 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 2
#python3 main.py --method ogd --memory_size 480 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 2
#
## split mnist
#python3 main.py --method sogd --memory_size 1200 --dataset split_mnist --sketch_per_task 960 --sketch_method basic --seed 2
#python3 main.py --method sogd --memory_size 600 --dataset split_mnist --sketch_per_task 960 --sketch_method lowrankapprox --seed 2
#python3 main.py --method sogd --memory_size 300 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 2
#python3 main.py --method pca --memory_size 240 --dataset split_mnist --sketch_per_task 960 --pca_sample 960 --sketch_method lowranksym --seed 2
#python3 main.py --method ogd --memory_size 240 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 2
#python3 main.py --method sgd --memory_size 240 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 2
#python3 main.py --method ogd --memory_size 960 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 2

# unconstrained sketching

#python3 main.py --method sogd --memory_size 1200 --dataset split_cifar --hidden_dim 500 --sketch_per_task 2500 --sketch_method basic --seed 1
#python3 main.py --method sogd --memory_size 300 --dataset split_cifar --hidden_dim 500 --sketch_per_task 2500 --sketch_method lowranksym --nepoch 30 --seed 1
#python3 main.py --method sogd --memory_size 600 --dataset split_cifar --hidden_dim 500 --sketch_per_task 2500 --sketch_method lowrankapprox --seed 1

# Split CIFAR

#python3 main.py --method sogd --memory_size 300 --dataset split_cifar --hidden_dim 500 --sketch_per_task 10000 --sketch_method lowranksym --nepoch 30 --n_tasks 20 --seed 5
#python3 main.py --method sogd --memory_size 1200 --dataset split_cifar --hidden_dim 500 --sketch_per_task 10000 --sketch_method basic --nepoch 30 --n_tasks 20 --seed 2

#python3 main.py --method sogd --memory_size 300 --dataset split_cifar --hidden_dim 500 --sketch_per_task 10000 --sketch_method lowranksym --nepoch 30 --n_tasks 20 --seed 3


# 5 Task REPLICATING OGD EXPERIMENT

#python3 main.py --method sogd --memory_size 200 --n_tasks 5 --rotate_step 10 --dataset rotated --sketch_per_task 60000 --sketch_method basic --seed 1 --subset_size 10000 --nepoch 5
#python3 main.py --method sogd --memory_size 50 --n_tasks 5 --rotate_step 10 --dataset rotated --sketch_per_task 60000 --sketch_method lowranksym --seed 1 --subset_size 10000 --nepoch 5
#python3 main.py --method sogd --memory_size 100 --n_tasks 5 --rotate_step 10 --dataset rotated --sketch_per_task 10000 --sketch_method lowrankapprox --seed 1

# AGEM SPLIT MNIST

python3 main.py --method agem --memory_size 240 --agem_mem_batch_size 240 --dataset split_mnist --hidden_dim 100 --nepoch 30 --seed 1 --n_tasks 5
python3 main.py --method agem --memory_size 240 --agem_mem_batch_size 240 --dataset split_mnist --hidden_dim 100 --nepoch 30 --seed 2 --n_tasks 5
python3 main.py --method agem --memory_size 240 --agem_mem_batch_size 240 --dataset split_mnist --hidden_dim 100 --nepoch 30 --seed 3 --n_tasks 5
