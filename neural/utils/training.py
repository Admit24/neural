import torch
from torch import distributed as dist


def setup_distributed(enable=True, local_rank=0):
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)

    if enable:
        dist.init_process_group('nccl', init_method='env://')
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
        local_rank = local_rank
    else:
        world_size = 1
        world_rank = 0
        local_rank = 0

    return world_size, world_rank, local_rank


def get_datasets_root(name):
    import os
    from logging import fatal

    dataset_dir = os.environ['DATASET_DIR']
    if not os.path.isdir(dataset_dir):
        fatal("DATASET_DIR is not specified or is invalid. Please specify it using the environment variable as `DATASET_DIR=/the/dataset/location/`.")

    dataset_location = os.path.join(dataset_dir, name)

    if not os.path.isdir(dataset_location):
        fatal(
            "{} dataset was not found. Please extract the dataset to {}."
            .format(name, dataset_location))

    return dataset_location


def create_sampler(dataset, world_size, local_rank, training=True, enable=True):
    from torch.utils.data import DistributedSampler

    if enable:
        return DistributedSampler(
            dataset,
            num_replicas=world_size, rank=local_rank,
            shuffle=training)

    else:
        return None
