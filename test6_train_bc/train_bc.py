import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test4_data_loading.test4_1_finding_data_files import find_all_hdf5
from test4_data_loading.test4_2_computing_normalization_statistics import get_norm_stats
from test4_data_loading.test4_3_episodic_dataset import EpisodicDataset
from test4_data_loading.test4_4_batch_sampler import BatchSampler
from test4_data_loading.test4_5_test_load_data import repeater, load_data
from test5_ACT_policy.test5_4_ACTpolicy import ACT_policy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy import random
from itertools import repeat
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# def check_model_device(model):
#     for name, param in model.named_parameters():
#         if param.device != device:
#             print(f"Parameter '{name}' is on {param.device}, expected {device}.")
# def kl_divergence(mu, logvar):
#     # mu: Dim(batch_size, z_latent_dim)
#     # logvar: Dim(batch_size, z_latent_dim)
#     batch_size = mu.size(0)
#     assert batch_size != 0
#     if mu.data.ndimension() == 4:
#         mu = mu.view(mu.size(0), mu.size(1))
#     if logvar.data.ndimension() == 4:
#         logvar = logvar.view(logvar.size(0), logvar.size(1))

#     klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
#     total_kld = klds.sum(1).mean(0, True)
#     dimension_wise_kld = klds.mean(0)
#     mean_kld = klds.mean(1).mean(0, True)

#     return total_kld, dimension_wise_kld, mean_kld

if __name__ == "__main__":
    from constants import SIM_TASK_CONFIGS

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    task_name = 'sim_transfer_cube_scripted'
    task_config = SIM_TASK_CONFIGS[task_name]

    dataset_dir_l = []
    dataset_dir = os.path.join(os.path.dirname(__file__), '../saved_data/')
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
    action_chunk_size = 100
    num_steps = 5000
    z_latent_dim = 32
    qpos_dim = 14
    action_dim = 16
    embed_dim = 512
    
    # Create a directory to save checkpoints
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Define the path for saving the model
    checkpoint_path = os.path.join(checkpoint_dir, 'policy_final.pth')

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, action_chunk_size, True, True, '', stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)
    # Replaces the original train_dataloader with an infinite generator provided by the repeater function. Now, train_dataloader is an iterator that can supply data batches endlessly.
    train_dataloader = repeater(train_dataloader)

    # Define the model parameters
    args = {
        'camera_names': camera_names,
        'qpos_dim': qpos_dim,
        'z_latent_dim':  z_latent_dim,
        'embed_dim': embed_dim,
        'action_dim': action_dim,
        'action_chunk_size': action_chunk_size,
        'num_input_tokens': len(camera_names) * 300 + 2,  # 300 tokens per camera, plus z and qpos tokens
        'num_output_tokens': action_chunk_size,  # Same as action_chunk_size
        'out_dim': action_dim,
        'depth': 6,
        'n_heads': 4
    }

    # Initialize the policy and move it to the GPU
    policy = ACT_policy(args).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    for step in tqdm(range(num_steps + 1)):
        # Training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        image_data, qpos_data, action_data, is_pad = data

        # Move data to the GPU
        image_data = image_data.to(device)
        qpos_data = qpos_data.to(device)
        action_data = action_data.to(device)
        is_pad = is_pad.to(device)

        # Run the policy in training
        # TODO: check padding
        action_hat = policy(image_data, qpos_data, action_data, is_pad)  # (batch_size, num_output_tokens, out_dim)

        # Compute the loss
        # 1. Reconstruction loss
        all_l1 = F.l1_loss(action_data, action_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss = l1
        loss.backward()
        optimizer.step()
        print('Step:', step, 'Loss:', loss.item())
        
        if step % 100 == 0 and step != 0:
            checkpoint = {
                'step': step,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'args': args  # Optionally save args for reference
            }
            checkpoint_file = os.path.join(checkpoint_dir, f'policy_step_{step}.pth')
            torch.save(checkpoint, checkpoint_file)
            print(f'Checkpoint saved at step {step} to {checkpoint_file}')

    
    # Save the final model after training
    final_checkpoint = {
        'step': num_steps,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'args': args
    }
    torch.save(final_checkpoint, checkpoint_path)
    print(f'Final model saved to {checkpoint_path}')