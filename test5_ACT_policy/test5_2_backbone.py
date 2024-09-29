import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test4_data_loading.test4_1_finding_data_files import find_all_hdf5
from test4_data_loading.test4_2_computing_normalization_statistics import get_norm_stats
from test4_data_loading.test4_3_episodic_dataset import EpisodicDataset
from test4_data_loading.test4_4_batch_sampler import BatchSampler
from test4_data_loading.test4_5_test_load_data import repeater,load_data
import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import transforms

class ImagePreprocessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone(name='resnet18', train_backbone=False, return_interm_layers=False, dilation=False)
    def forward(self, x):
        #x: Dim(batch_size, 3, 480, 640)
        x = self.backbone(x)#Dim:(batch_size, 512, 15, 20)
        x = x['0']
        # Flatten only the last two dimensions
        x = torch.flatten(x, start_dim=2, end_dim=-1)  # Dim: (batch_size, 512, 300)
        x = x.permute(0, 2, 1)  # Dim: (batch_size, 300, 512)

        return x

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x_input):
        x_output = self.body(x_input)
        return x_output

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
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

    #image_data: Dim(batch_size, num_cameras=3, 3, 480, 640)
    cam_id = 0
    image_batch = image_data[:, cam_id] # Dim(batch_size, 3, 480, 640)

    # Normalize the image
    # Define the normalization transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
    input_batch = normalize(image_batch)
    # Move the input to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_batch = input_batch.to(device)
    
    model = ImagePreprocessModel()
    model.to(device)
    
    with torch.no_grad():
        output = model(input_batch)#Dim:(batch_size, 300=15*20 , 512)
    print(output.shape)