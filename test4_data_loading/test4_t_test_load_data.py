from test4_1_finding_data_files import find_all_hdf5
from test4_2_computing_normalization_statistics import get_norm_stats
from test4_3_episodic_dataset import EpisodicDataset
from test4_4_batch_sampler import BatchSampler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy import random
def flatten_list(l):
    return [item for sublist in l for item in sublist]
def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99):
    # Find all dataset files
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]

    # Get episode lengths and IDs
    num_episodes = len(dataset_path_list)
    shuffled_episode_ids = np.random.permutation(num_episodes)
    train_episode_ids = shuffled_episode_ids[:int(train_ratio * num_episodes)]
    val_episode_ids = shuffled_episode_ids[int(train_ratio * num_episodes):]

    # Compute normalization statistics
    norm_stats, all_episode_len = get_norm_stats(dataset_path_list)

    # Create datasets
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, all_episode_len, chunk_size, policy_class)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, all_episode_len, chunk_size, policy_class)

    # Create batch samplers
    batch_sampler_train = BatchSampler(batch_size_train, [all_episode_len[i] for i in train_episode_ids], sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, [all_episode_len[i] for i in val_episode_ids], None)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=4)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim