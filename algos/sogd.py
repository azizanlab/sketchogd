import torch
from algos.common import Memory
from utils.utils import parameters_to_grad_vector, count_parameter
import time


def _get_new_ogd_basis(trainer, train_loader, device, optimizer, model, forward):
    for _, element in enumerate(train_loader):
        inputs = element[0].to(device)

        targets = element[1].to(device)

        task = element[2]

        out = forward(inputs, task)

        assert out.shape[0] == 1

        pred = out[0, int(targets.item())].cpu()

        optimizer.zero_grad()
        pred.backward()

        ### retrieve  \nabla f(x) wrt theta
        yield parameters_to_grad_vector(trainer.get_params_dict(last=False)).cpu()
        #new_basis.append(parameters_to_grad_vector(trainer.get_params_dict(last=False)).cpu())

    #del out, inputs, targets
    #torch.cuda.empty_cache()


def project_vec(vec, proj_basis):
    # if proj_basis.shape[1] > 0:  # param x basis_size
    #     dots = torch.matmul(vec, proj_basis)  # basis_size  dots= [  <vec, i >   for i in proj_basis ]
    #     out = torch.matmul(proj_basis, dots)  # out = [  <vec, i > i for i in proj_basis ]
    #     return out
    # else:
    #     return torch.zeros_like(vec)
    if proj_basis.ndim > 2 or vec.ndim > 2:
        raise Exception("Too high dimensional input to project_vec, may not work")

    if proj_basis.shape[1] > 0 :  # param x basis_size
        dots = torch.matmul(proj_basis.T, vec).T  # basis_size  dots= [  <vec, i >   for i in proj_basis ]
        try:
            out = torch.zeros_like(vec)
            index = 0
            while index < out.shape[-1]:
                out[index:index + 100_000] = torch.matmul(dots, proj_basis[index:index + 100_000].T).T
                index += 100_000
            #out = torch.matmul(proj_basis, dots)  # out = [  <vec, i > i for i in proj_basis ]
        except:
            out = 0
            print("dots: ", dots)
            print("proj_basis: ", proj_basis)
            print("shape vec: ", vec.shape)
            raise RuntimeError("Cuda error? mismatch inputs?")
        return out
    else:
        return torch.zeros_like(vec)


def update_mem(trainer, train_loader, method="fixed_rank_sym_approx"):
    """
    For every datapoint in the task, get its gradient vector and add it to the sketch
    train_loader is only for the current task
    """
    ####################################### Sketch MEM ###########################

    # Randomly select data points to sketch, to save time
    # We choose to sketch "density" vectors per sketch rank per task
    num_sketched = trainer.config.sketch_per_task
    num_sample_per_task = min(len(train_loader.dataset), num_sketched)
    randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]

    selected_data = Memory()
    for ind in randind:  # save it to the memory
        selected_data.append(train_loader.dataset[ind])

    ogd_train_loader = torch.utils.data.DataLoader(selected_data,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=1)

    new_basis_generator = _get_new_ogd_basis(trainer,
                                          ogd_train_loader,
                                          trainer.config.device,
                                          trainer.optimizer,
                                          trainer.model,
                                          trainer.forward)

    #  for each new basis vector, add it to the sketch
    for t, basis_vector in enumerate(new_basis_generator):
        vector_to_sketch = torch.unsqueeze(basis_vector, dim=1)
        if method=="basic":  # can make this faster by combining into one matrix maybe?
            trainer.sketch.update_concatenate(vector_to_sketch.cpu())
        else:
            trainer.sketch.update_add_vector(vector_to_sketch.cpu())


    if method=="basic": # low rank approx applied directly to G
        trainer.sketch_basis = trainer.sketch.low_rank_approx()[0].to(trainer.config.device)
    if method=="lowrankapprox": # low rank approx applied to GG^T, SketchOGD 2
        print("using lowrankapprox Method")
        trainer.sketch_basis = trainer.sketch.low_rank_approx()[0].to(trainer.config.device)
    if method=="rowspace": # just using the X part
        print("using rowspace Method")
        X = trainer.sketch.low_rank_approx()[1].to(trainer.config.device)
        trainer.sketch_basis = torch.linalg.qr(torch.transpose(X, 0, 1))[0]

    if method=="lowranksym":  # SketchOGD 3
        trainer.sketch_basis = trainer.sketch.low_rank_sym_approx()[0].to(trainer.config.device)
    if method == "fixedranksym":
        # Perform SVD to diagonalize S, but don't truncate
        trainer.sketch_basis = trainer.sketch.fixed_rank_sym_approx(truncate=False)[0].to(trainer.config.device)
    if method=="truncatedsym":
        # truncate rank back down to r
        trainer.sketch_basis = trainer.sketch.fixed_rank_sym_approx(truncate=True)[0].to(trainer.config.device)

