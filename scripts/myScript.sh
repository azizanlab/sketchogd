#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2022a

# SEED 1:

# permuted
#python3 main.py --method sogd --memory_size 600 --dataset permuted --sketch_per_task 480 --sketch_method lowrankapprox


# SEED 2:


# SEED 3:

## rotated mnist
#python3 main.py --method sogd --memory_size 1200 --dataset rotated --sketch_per_task 480 --sketch_method basic --seed 3
#python3 main.py --method sogd --memory_size 600 --dataset rotated --sketch_per_task 480 --sketch_method lowrankapprox --seed 3
#python3 main.py --method sogd --memory_size 300 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 3
#python3 main.py --method pca --memory_size 120 --dataset rotated --sketch_per_task 480 --pca_sample 480 --sketch_method lowranksym --seed 3
#python3 main.py --method ogd --memory_size 120 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 3
#python3 main.py --method sgd --memory_size 120 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 3
#python3 main.py --method ogd --memory_size 480 --dataset rotated --sketch_per_task 480 --sketch_method lowranksym --seed 3
#
## permuted mnist
#python3 main.py --method sogd --memory_size 1200 --dataset permuted --sketch_per_task 480 --sketch_method basic --seed 3
#python3 main.py --method sogd --memory_size 600 --dataset permuted --sketch_per_task 480 --sketch_method lowrankapprox --seed 3
#python3 main.py --method sogd --memory_size 300 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 3
#python3 main.py --method pca --memory_size 120 --dataset permuted --sketch_per_task 480 --pca_sample 480 --sketch_method lowranksym --seed 3
#python3 main.py --method ogd --memory_size 120 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 3
#python3 main.py --method sgd --memory_size 120 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 3
#python3 main.py --method ogd --memory_size 480 --dataset permuted --sketch_per_task 480 --sketch_method lowranksym --seed 3
#
## split mnist
#python3 main.py --method sogd --memory_size 1200 --dataset split_mnist --sketch_per_task 960 --sketch_method basic --seed 3
#python3 main.py --method sogd --memory_size 600 --dataset split_mnist --sketch_per_task 960 --sketch_method lowrankapprox --seed 3
#python3 main.py --method sogd --memory_size 300 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 3
#python3 main.py --method pca --memory_size 240 --dataset split_mnist --sketch_per_task 960 --pca_sample 960 --sketch_method lowranksym --seed 3
#python3 main.py --method ogd --memory_size 240 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 3
#python3 main.py --method sgd --memory_size 240 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 3
#python3 main.py --method ogd --memory_size 960 --dataset split_mnist --sketch_per_task 960 --sketch_method lowranksym --seed 3


# calc singular values
#python3 main.py --method ogd --memory_size 480 --dataset rotated --compute_singular_values True --seed 1
#python3 main.py --method ogd --memory_size 480 --dataset permuted --compute_singular_values True --seed 1

#python3 main.py --method ogd --memory_size 480 --dataset rotated --compute_singular_values True --seed 2
#python3 main.py --method ogd --memory_size 480 --dataset permuted --compute_singular_values True --seed 2

#python3 main.py --method ogd --memory_size 480 --dataset rotated --compute_singular_values True --seed 3
#python3 main.py --method ogd --memory_size 480 --dataset permuted --compute_singular_values True --seed 3


# unconstrained sketching
#
## basic
## rotated
#python3 main.py --method sogd --memory_size 1200 --dataset rotated --sketch_per_task 10000 --sketch_method basic --seed 1
#python3 main.py --method sogd --memory_size 1200 --dataset rotated --sketch_per_task 10000 --sketch_method basic --seed 2
#python3 main.py --method sogd --memory_size 1200 --dataset rotated --sketch_per_task 10000 --sketch_method basic --seed 3
#
#
## permuted
#python3 main.py --method sogd --memory_size 1200 --dataset permuted --sketch_per_task 10000 --sketch_method basic --seed 1
#python3 main.py --method sogd --memory_size 1200 --dataset permuted --sketch_per_task 10000 --sketch_method basic --seed 2
#python3 main.py --method sogd --memory_size 1200 --dataset permuted --sketch_per_task 10000 --sketch_method basic --seed 3
#
## split
#python3 main.py --method sogd --memory_size 1200 --dataset split_mnist --sketch_per_task 10000 --sketch_method basic --seed 1
#python3 main.py --method sogd --memory_size 1200 --dataset split_mnist --sketch_per_task 10000 --sketch_method basic --seed 2
#python3 main.py --method sogd --memory_size 1200 --dataset split_mnist --sketch_per_task 10000 --sketch_method basic --seed 3
#
#
#
## lowranksym
#
## rotated
#python3 main.py --method sogd --memory_size 300 --dataset rotated --sketch_per_task 10000 --sketch_method lowrankapprox --seed 1 --nepoch 30 --n_tasks 10
#python3 main.py --method sogd --memory_size 300 --dataset rotated --sketch_per_task 10000 --sketch_method lowrankapprox --seed 2 --nepoch 30 --n_tasks 10
#python3 main.py --method sogd --memory_size 300 --dataset rotated --sketch_per_task 10000 --sketch_method lowrankapprox --seed 3 --nepoch 30 --n_tasks 10
#
#
#
## permuted
#python3 main.py --method sogd --memory_size 300 --dataset permuted --sketch_per_task 10000 --sketch_method lowrankapprox --seed 1 --nepoch 30 --n_tasks 10
#python3 main.py --method sogd --memory_size 300 --dataset permuted --sketch_per_task 10000 --sketch_method lowrankapprox --seed 2 --nepoch 30 --n_tasks 10
#python3 main.py --method sogd --memory_size 300 --dataset permuted --sketch_per_task 10000 --sketch_method lowrankapprox --seed 3 --nepoch 30 --n_tasks 10
#
#
#
##split
#python3 main.py --method sogd --memory_size 300 --dataset split_mnist --sketch_per_task 10000 --sketch_method lowrankapprox --seed 1 --nepoch 30 --n_tasks 5
#python3 main.py --method sogd --memory_size 300 --dataset split_mnist --sketch_per_task 10000 --sketch_method lowrankapprox --seed 2 --nepoch 30 --n_tasks 5
#python3 main.py --method sogd --memory_size 300 --dataset split_mnist --sketch_per_task 10000 --sketch_method lowrankapprox --seed 3 --nepoch 30 --n_tasks 5

# split cifar
# USE 20 TASKS
#python3 main.py --method ogd --memory_size 60 --dataset split_cifar --hidden_dim 500 --sketch_per_task 480 --nepoch 30 --n_tasks 20 --seed 5
#python3 main.py --method pca --memory_size 50 --dataset split_cifar --hidden_dim 500 --pca_sample 200 --nepoch 30 --n_tasks 20 --seed 5

#python3 main.py --method sgd --memory_size 60 --dataset split_cifar --hidden_dim 500 --sketch_per_task 480 --nepoch 30 --n_tasks 20 --seed 1
#python3 main.py --method sgd --memory_size 60 --dataset split_cifar --hidden_dim 500 --sketch_per_task 480 --nepoch 30 --n_tasks 20 --seed 2
#python3 main.py --method sgd --memory_size 60 --dataset split_cifar --hidden_dim 500 --sketch_per_task 480 --nepoch 30 --n_tasks 20 --seed 3
#python3 main.py --method sgd --memory_size 60 --dataset split_cifar --hidden_dim 500 --sketch_per_task 480 --nepoch 30 --n_tasks 20 --seed 4
#python3 main.py --method sgd --memory_size 60 --dataset split_cifar --hidden_dim 500 --sketch_per_task 480 --nepoch 30 --n_tasks 20 --seed 5
#

# Split CIFAR Restricted
# maxogd
python3 main.py --method ogd --memory_size 240 --dataset split_cifar --hidden_dim 500 --sketch_per_task 240 --nepoch 30 --n_tasks 20 --seed 3
python3 main.py --method pca --memory_size 60 --dataset split_cifar --hidden_dim 500 --pca_sample 240 --nepoch 30 --n_tasks 20 --seed 3
python3 main.py --method sogd --memory_size 1200 --dataset split_cifar --hidden_dim 500 --sketch_per_task 240 --sketch_method basic --nepoch 30 --n_tasks 20 --seed 3
python3 main.py --method sogd --memory_size 600 --dataset split_cifar --hidden_dim 500 --sketch_per_task 240 --sketch_method lowrankapprox --nepoch 30 --n_tasks 20 --seed 3
python3 main.py --method sogd --memory_size 300 --dataset split_cifar --hidden_dim 500 --sketch_per_task 240 --sketch_method lowranksym --nepoch 30 --n_tasks 20 --seed 3


# 5 TASK REPLICATING OGD EXPERIMENT
#python3 main.py --method ogd --memory_size 200 --n_tasks 5 --rotate_step 10 --dataset rotated --seed 1
#python3 main.py --method pca --memory_size 180 --n_tasks 5 --rotate_step 10 --dataset rotated --pca_sample 400 --seed 1

#python3 main.py --method ogd --memory_size 40 --n_tasks 5 --rotate_step 10 --dataset rotated --seed 1 --subset_size 10000

#python3 main.py --method pca --memory_size 30 --n_tasks 5 --rotate_step 10 --dataset rotated --pca_sample 50 --seed 1
#python3 main.py --method pca --memory_size 30 --n_tasks 5 --rotate_step 10 --dataset rotated --pca_sample 110 --seed 1
#python3 main.py --method pca --memory_size 40 --n_tasks 5 --rotate_step 10 --dataset rotated --pca_sample 200 --seed 1

