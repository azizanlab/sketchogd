import matplotlib as mpl
mpl.use('Agg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import trainer
import pickle
import os
from utils.utils import get_benchmark_data_loader, test_error, train_error
from algos import ogd,ewc,gem,sogd
from tqdm.auto import tqdm
import time

if __name__ == '__main__':
#if True:
    parser = argparse.ArgumentParser()

    ### Algo parameters
    parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--val_size", default=256, type=int)
    parser.add_argument("--nepoch", default=30, type=int)             # Number of epoches
    parser.add_argument("--batch_size", default=10, type=int)      # Batch size
    parser.add_argument("--memory_size", default=480, type=int)     # size of the memory 302sogd vs 122ogd
    parser.add_argument("--hidden_dim", default=100, type=int)      # size of the hidden layer
    parser.add_argument('--lr',default=1e-3, type=float)
    parser.add_argument('--n_tasks',default=20, type=int)  # Sets number of tasks to run, standard is 10
    parser.add_argument('--workers',default=0, type=int)  # original was 2, error when more than 0
    parser.add_argument('--eval_freq',default=1000, type=int)
    parser.add_argument('--compute_singular_values', default=False, type=bool)  # rotated, permuted, split_mnist

## Methods parameters
    parser.add_argument("--all_features",default=0, type=int) # Leave it to 0, this is for the case when using Lenet, projecting orthogonally only against the linear layers seems to work better

    ## Dataset
    parser.add_argument('--dataset_root_path',default="~/PycharmProjects/ContinuumLearning/datasets", type=str,help="path to your dataset  ex: /home/usr/datasets/")
    parser.add_argument('--subset_size',default=1000, type=int, help="number of samples per class, ex: for MNIST, \
                subset_size=1000 wil results in a dataset of total size 10,000") # default 1000, change to 10,000 for split_mnist
    parser.add_argument('--dataset',default="rotated", type=str) #rotated, permuted, split_mnist, split_cifar
    parser.add_argument('--rotate_step',default=5, type=int)
    parser.add_argument("--is_split", action="store_true")
    parser.add_argument('--first_split_size',default=2, type=int)
    parser.add_argument('--other_split_size',default=2, type=int)
    parser.add_argument("--rand_split",default=False, action="store_true")
    parser.add_argument('--force_out_dim', type=int, default=0,
                                  help="Set 0 to let the task decide the required output dimension", required=False)

    ## Method
    parser.add_argument('--method',default="ogd", type=str,help="sgd,ogd,pca,agem,gem-nt,sogd")

    ## SOGD
    parser.add_argument("--sketch_per_task", default=480, # 480
                    type=int)  # Number of sketched/compressed vectors per task, only for sogd
    parser.add_argument("--sketch_method", default="basic",
                    type=str)  # Changes the type of sketching. SketchOGD-1: basic, SketchOGD-2: lowrankapprox, SketchOGD-3: lowranksym

    ## PCA-OGD
    parser.add_argument('--pca_sample',default=480, type=int)
    ## agem
    parser.add_argument("--agem_mem_batch_size", default=256, type=int)     # size of the memory
    parser.add_argument('--margin_gem',default=0.5, type=float)
    ## EWC
    parser.add_argument('--ewc_reg',default=10, type=float)
    parser.add_argument('--fisher_sample',default=1024, type=int)

    ## Folder / Logging results
    parser.add_argument('--save_name',default="result", type=str,  help="name of the file")

    config = parser.parse_args()
    config.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



    np.set_printoptions(suppress=True)

    config_dict=vars(config)

    ### setting seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.enabled=True

    config.folder="method_{}_dataset_{}_memory_size_{}_bs_{}_lr_{}_epochs_per_task_{}".format(config.method,
                                                    config.dataset,config.memory_size,config.batch_size, config.lr,config.nepoch)

    if config.method=='sogd':
        config.folder = "method_{}_dataset_{}_memory_size_{}_bs_{}_lr_{}_epochs_per_task_{}_sketched_{}_sketchmethod_{}".format(config.method,
                                                                                                    config.dataset,
                                                                                                    config.memory_size,
                                                                                                    config.batch_size,
                                                                                                    config.lr,
                                                                                                    config.nepoch,
                                                                                                    config.sketch_per_task,
                                                                                                    config.sketch_method,)
    if config.method=='pca':
        config.folder = "method_{}_dataset_{}_memory_size_{}_bs_{}_lr_{}_epochs_per_task_{}_compressed_{}".format(config.method,
                                                                                                    config.dataset,
                                                                                                    config.memory_size,
                                                                                                    config.batch_size,
                                                                                                    config.lr,
                                                                                                    config.nepoch,
                                                                                                    config.pca_sample)
    if config.method=='agem':
        config.folder = "method_{}_dataset_{}_memory_size_{}_bs_{}_lr_{}_epochs_per_task_{}_averaged_{}".format(config.method,
                                                                                                    config.dataset,
                                                                                                    config.memory_size,
                                                                                                    config.batch_size,
                                                                                                    config.lr,
                                                                                                    config.nepoch,
                                                                                                    config.agem_mem_batch_size)
    config.folder = config.folder + "_tasks_{}".format(config.n_tasks)
    if config.dataset == "rotated":
        config.folder = config.folder + "_angle_{}".format(config.rotate_step)

    if config.subset_size!= 1000:
        config.folder = config.folder + "_subset_size"

    ## create folder to log results
    if not os.path.exists("results/"+config.folder):
        os.makedirs("results/"+config.folder, exist_ok=True)

    print("Starting test for: ", config.folder)

    ### name of the file
    config.save_name=config.save_name+'_seed_'+str(config.seed)
    config.save_name2='singular_values'+'_seed_'+str(config.seed)


    ### dataset path
    # config.dataset_root_path="..."


    ########################################################################################
    ### dataset ############################################################################
    print('loading dataset')
    train_dataset_splits,val_loaders,task_output_space=get_benchmark_data_loader(config)
    config.out_dim = {'All': config.force_out_dim} if config.force_out_dim > 0 else task_output_space


    ### loading trainer module
    trainer=trainer.Trainer(config,val_loaders)




    start_time = time.time()

    t=0
    ########################################################################################
    ### start training #####################################################################
    first_task_inputs = []
    first_task_targets = []
    for task_in in range(config.n_tasks):
        rr=0

        train_loader = torch.utils.data.DataLoader(train_dataset_splits[str(task_in+1)],
                                                           batch_size=config.batch_size,
                                                           shuffle=True,
                                                           num_workers=config.workers)


        ### train for EPOCH times
        print("================== TASK {} / {} =================".format(task_in+1, config.n_tasks))
        #for epoch in tqdm(range(config.nepoch), desc="Train task"):
        for epoch in range(config.nepoch):
            #trainer.ogd_basis.to(trainer.config.device)
            #print("starting epoch: ", epoch)
            for i, (input, target, task) in enumerate(train_loader):


                trainer.task_id = int(task[0])
                t+=1
                rr+=1
                inputs = input.to(trainer.config.device)
                target = target.long().to(trainer.config.device)

                out = trainer.forward(inputs,task).to(trainer.config.device)
                loss = trainer.criterion(out, target)
                if config.method=="ewc" and (task_in)>0:
                    loss+=config.ewc_reg*ewc.penalty(trainer)

                loss.backward()
                trainer.optimizer_step(first_task=(task_in==0))
                ### validation accuracy

                if (rr-1)%trainer.config.eval_freq==0:
                    for element in range(task_in+1):
                        trainer.acc[element]['test_acc'].append(test_error(trainer,element))
                        trainer.acc[element]['training_steps'].append(t)
            #print("    train score: ", train_error(trainer,train_loader, task_in))
            #print("    test score: ", test_error(trainer,task_in))
            print("    t (epochs_so_far*step_per_epoch: ", t)

        for element in range(task_in+1):
            trainer.acc[element]['test_acc'].append(test_error(trainer,element))
            trainer.acc[element]['training_steps'].append(t)
            print("  task {} / accuracy: {}  ".format(element+1, trainer.acc[element]['test_acc'][-1]))


        ## update memory at the end of each tasks depending on the method
        if config.method in ['ogd','pca']:
            trainer.ogd_basis.to(trainer.config.device)
            ogd.update_mem(trainer,train_loader,task_in+1)

        if config.method=='sogd' or (config.method in ['ogd','pca'] and config.compute_singular_values):  # update sketch memory
            sogd.update_mem(trainer,train_loader, method=config.sketch_method)

        if config.method=="agem":
            gem.update_agem_memory(trainer,train_loader,task_in+1)

        if config.method=="gem-nt":  ## GEM-NT
            gem.update_gem_no_transfer_memory(trainer,train_loader,task_in+1)

        if config.method=="ewc":
            ewc.update_means(trainer)
            ewc.consolidate(trainer,ewc._diag_fisher(trainer,train_loader))

        #U, D, V = torch.linalg.svd(trainer.unchanged_gradients)
        #print("SINGULAR VALUES: ", D)
        if task_in == 0 and config.compute_singular_values:
            G = trainer.unchanged_gradients.to(trainer.config.device)
            print("G shape: ", G.shape)

            # random_ogd_indices = torch.randperm(G.shape[0])[:int(G.shape[1]/4)]
            # G_rogd = torch.linalg.qr(G[:,:120])[0]
            # dif_G_rogd = G - ogd.project_vec(G,
            #             proj_basis=G_rogd)
            #
            # rogd_norm = torch.linalg.matrix_norm(dif_G_rogd)
            # print("rogd_norm shape: ",rogd_norm.shape)
            # print("rogd try .item(): ", rogd_norm.item())
            # #print("||G - G_rogd||^2: ", rogd_norm)

            G.to(trainer.config.device)

            # _, _, v1 = torch.pca_lowrank(G[:,:480].T.cpu(), q=120, center=True, niter=2)
            #
            # G_pca = ogd.project_vec(G,
            #                         proj_basis=v1.to(trainer.config.device))
            # print("480->120 pca ||G - G_pca||^2: ", torch.linalg.matrix_norm(G - G_pca))
            #
            # _, _, v2 = torch.pca_lowrank(G[:, :200].T.cpu(), q=100, center=True, niter=2)
            #
            # G_pca2 = ogd.project_vec(G,
            #                         proj_basis=v2.to(trainer.config.device))
            # print("200->100 pca ||G - G_pca||^2: ", torch.linalg.matrix_norm(G - G_pca2))

            _, _, v3 = torch.pca_lowrank(G[:, :1100].T.cpu(), q=100, center=True, niter=2)

            G_pca3 = ogd.project_vec(G,
                                     proj_basis=v3.to(trainer.config.device))
            print("1100->100 pca ||G - G_pca||^2: ", torch.linalg.matrix_norm(G - G_pca3))

            G_sogd = sogd.project_vec(G,
                             proj_basis=trainer.sketch_basis.to(trainer.config.device))
            print("||G - G_sogd||^2: ", torch.linalg.matrix_norm(G - G_sogd))



    end_time = time.time()
    time_spent = end_time - start_time

    ### Plotting accuracies
    print('plotting accuracies')
    plt.close('all')
    for tasks_id in range(len(trainer.acc.items())):
        plt.plot(trainer.acc[tasks_id]['training_steps'],trainer.acc[tasks_id]['test_acc'])
    plt.xlabel("Trained per Class")
    plt.ylabel("Accuracy")
    title_string = str(config.method) + ", "+str(config.dataset)+", "+str(config.n_tasks)+" Tasks"
    plt.title(title_string)
    plt.grid()
    plt.savefig('results/'+config.folder+'/'+config.save_name+".png",dpi=72)




    print('Saving results')
    print(str(config.folder))
    output = open('results/'+config.folder+'/'+config.save_name+'.p', 'wb')
    pickle.dump(trainer.acc, output)
    output.close()

    # # save singular values
    # U, D, V = torch.linalg.svd(trainer.unchanged_gradients)
    # print("SINGULAR VALUES: ", D)
    # svd_output = open('results/'+config.folder+'/'+config.save_name2+'.p', 'wb')
    # pickle.dump(D, svd_output)
    # svd_output.close()

    time_output = open('results/'+config.folder+'/'+'time_spent'+'.p', 'wb')
    pickle.dump(time_spent, time_output)
    time_output.close()
    print('time spent: ', time_spent)

    for element in range(config.n_tasks):
        print("  task {} / accuracy: {}  ".format(element + 1, trainer.acc[element]['test_acc'][-1]))




