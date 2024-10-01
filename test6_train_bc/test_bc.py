import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test4_data_loading.test4_1_finding_data_files import find_all_hdf5
from test4_data_loading.test4_2_computing_normalization_statistics import get_norm_stats
from test4_data_loading.test4_3_episodic_dataset import EpisodicDataset
from test4_data_loading.test4_4_batch_sampler import BatchSampler
from test4_data_loading.test4_5_test_load_data import repeater, load_data
from test5_ACT_policy.test5_4_ACTpolicy import ACT_policy
from test1_simpleAlohaEnv.sim_env import make_sim_env,BOX_POSE
from test1_simpleAlohaEnv.utils import sample_box_pose
from test4_data_loading.test4_1_finding_data_files import find_all_hdf5
from test4_data_loading.test4_2_computing_normalization_statistics import get_norm_stats
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
import matplotlib.pyplot as plt
import time
from constants import FPS
import numpy as np
from einops import rearrange
# Function to load a checkpoint
def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint.get('step', None)
    loss = checkpoint.get('loss', None)
    args = checkpoint.get('args', None)
    return model, optimizer, step, loss, args
def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    
    return curr_image
if __name__ == "__main__":
    from constants import SIM_TASK_CONFIGS

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Set the task
    task_name = 'sim_transfer_cube'
    onscreen_cam = 'angle'
    camera_names =  ['top', 'left_wrist', 'right_wrist']
    batch_size_train = 8
    batch_size_val = 8
    action_chunk_size = 100
    num_steps = 5000
    z_latent_dim = 32
    qpos_dim = 14
    action_dim = 16
    embed_dim = 512
    max_timesteps = 100
    
    #Load in the model
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
    
    #Calculate datsset statistics
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)) , './saved_data/')
    skip_mirrored_data = True
    hdf5_files = find_all_hdf5(dataset_dir, skip_mirrored_data)
    stats, all_episode_len = get_norm_stats(hdf5_files)
    #define the pre_process and post process function
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    # Initialize your model and optimizer
    policy = ACT_policy(args).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Load the checkpoint
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'policy_step_3300.pth')
    policy, optimizer, step, loss, args = load_checkpoint(checkpoint_path, policy, optimizer, device)
    policy.cuda()
    policy.eval()
    #make the environment
    env = make_sim_env(task_name)
    BOX_POSE[0] = sample_box_pose() # used in sim reset
    ts = env.reset()
    
    ### onscreen render
    ax = plt.subplot()
    plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
    plt.ion()
    
    
    # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
    qpos_history_raw = np.zeros((max_timesteps, qpos_dim))
    image_list = [] # for visualization
    qpos_list = []
    target_qpos_list = []
    rewards = []
    # if use_actuator_net:
    #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
    with torch.inference_mode():
        time0 = time.time()
        DT = 1 / FPS
        culmulated_delay = 0 
        for t in range(max_timesteps):
            time1 = time.time()
            ### update onscreen render and wait for DT
            image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            plt_img.set_data(image)
            plt.pause(DT)
            
            ### process previous timestep to get qpos and image_list
            time2 = time.time()
            obs = ts.observation
            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})
            qpos_numpy = np.array(obs['qpos'])#(14,)
            qpos_history_raw[t] = qpos_numpy#
            qpos = pre_process(qpos_numpy)#(14,)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)#(1,14)
            curr_image = get_image(ts, camera_names, rand_crop_resize=False)
            
            # warm up
            if t == 0:
                for _ in range(10):
                    policy(image_data=curr_image, qpos_data=qpos, action_data=None, is_pad=None)
                print('network warm up done')
                time1 = time.time()
                
            ### query policy
            time3 = time.time()
            all_actions = policy(image_data=curr_image, qpos_data=qpos, action_data=None, is_pad=None)#(1,100,16)
            raw_action = all_actions[:, 0]
            
            ### post-process actions
            time4 = time.time()
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            target_qpos = action[:-2]
            
            ### step the environment
            time5 = time.time()
            ts = env.step(target_qpos)
            
            ### for visualization
            qpos_list.append(qpos_numpy)
            target_qpos_list.append(target_qpos)
            rewards.append(ts.reward)
            duration = time.time() - time1
            sleep_time = max(0, DT - duration)
            # print(sleep_time)
            time.sleep(sleep_time)