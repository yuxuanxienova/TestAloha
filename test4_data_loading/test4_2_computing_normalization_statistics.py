import h5py
import numpy as np
import torch
import os
from test4_1_finding_data_files import find_all_hdf5
def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # Compute statistics
    action_mean = all_action_data.mean(dim=0).float()
    action_std = all_action_data.std(dim=0).float()
    action_std = torch.clamp(action_std, min=1e-2)  # Avoid division by zero

    qpos_mean = all_qpos_data.mean(dim=0).float()
    qpos_std = all_qpos_data.std(dim=0).float()
    qpos_std = torch.clamp(qpos_std, min=1e-2)

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "action_min": action_min.numpy() - eps,
        "action_max": action_max.numpy() + eps,
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": qpos
    }

    return stats, all_episode_len

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__) , './saved_data/')
    skip_mirrored_data = True
    hdf5_files = find_all_hdf5(dataset_dir, skip_mirrored_data)
    print(hdf5_files)
    stats, all_episode_len = get_norm_stats(hdf5_files)
    print(stats)