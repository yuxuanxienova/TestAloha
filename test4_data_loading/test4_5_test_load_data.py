import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test4_data_loading.test4_1_finding_data_files import find_all_hdf5
from test4_data_loading.test4_2_computing_normalization_statistics import get_norm_stats
from test4_data_loading.test4_3_episodic_dataset import EpisodicDataset
from test4_data_loading.test4_4_batch_sampler import BatchSampler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy import random
from itertools import repeat

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1
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

if __name__ == "__main__":
    from constants import SIM_TASK_CONFIGS
    
    task_name = 'sim_transfer_cube_scripted'
    task_config = SIM_TASK_CONFIGS[task_name]
    
    dataset_dir_l = []
    dataset_dir = os.path.join(os.path.dirname(__file__) , '../saved_data/')
    dataset_dir_l.append(dataset_dir)
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)
    batch_size_train = 8
    batch_size_val = 8
    chunk_size = 100
    
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, True, True, '', stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)
    #Replaces the original train_dataloader with an infinite generator provided by the repeater function.Now, train_dataloader is an iterator that can supply data batches endlessly.
    train_dataloader = repeater(train_dataloader)
    #Fetching Data from the Repeater: The next function is used to fetch the next batch of data from the train_dataloader iterator.
    data = next(train_dataloader)
    image_data, qpos_data, action_data, is_pad = data
    #data[0]=image_data: Dim(batch_size, num_cameras=3, 3, 480, 640)
    #data[1]=qpos_data: Dim(batch_size, 14)
    #data[2]=action_data: Dim(batch_size, chunk_size, 16)
    #data[3]=is_pad: Dim(batch_size, chunk_size)
    
    print(data)