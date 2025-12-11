import torch
import torch.distributed as dist


def check_model_params_equal(model):
    """
    Check if all model parameters are the same across all distributed ranks.
    Raises a RuntimeError if parameters differ.
    Assumes that torch.distributed has already been initialized.

    Parameters:
        model (torch.nn.Module): The model whose parameters will be checked.
    """
    # Flatten all parameters into a single 1D tensor
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)

    # Get the world size and current rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Prepare a list of tensors to gather all parameters from each rank
    gathered_params = [torch.zeros_like(flat_params) for _ in range(world_size)]

    # Gather parameters from all ranks
    dist.all_gather(gathered_params, flat_params)

    # On each rank, check if all gathered parameters match the local rank's parameters
    all_equal_local = True
    for gp in gathered_params:
        if not torch.allclose(flat_params, gp):
            all_equal_local = False
            break

    # Aggregate results
    all_equal_tensor = torch.tensor(1.0 if all_equal_local else 0.0, device=flat_params.device)
    dist.all_reduce(all_equal_tensor, op=dist.ReduceOp.SUM)

    # If not all are equal, raise an error on rank 0
    if rank == 0:
        if int(all_equal_tensor.item()) != world_size:
            raise RuntimeError("Model parameters differ across ranks.")
