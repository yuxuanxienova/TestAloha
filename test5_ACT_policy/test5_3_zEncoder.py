import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from test2_transformer.Transformer import TransformerEncoder
from torch.distributions import Normal



class z_encoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # 
        self.z_latent_dim = args['z_latent_dim']
        self.action_dim = args['action_dim']
        self.embed_dim = args['embed_dim'] 
        self.qpos_dim = args['qpos_dim'] 
        self.k = args['k']#action chunk size 
        
        #layers
        self.encoder_action_layer = nn.Linear(self.action_dim, self.embed_dim)
        self.encoder_joint_layer = nn.Linear(self.qpos_dim, self.embed_dim)
        args_transformer = {
            "depth": 4,
            "n_heads": 4,
            "entity_dim": self.embed_dim,
            "num_input_tokens": self.k + 1, # action chunk size + joint q embed token
            "out_dim": self.embed_dim,
        }
        self.transformer_encoder = TransformerEncoder(args_transformer)
        self.post_layer = nn.Linear(self.embed_dim, self.z_latent_dim * 2)
    def encode(self, action_tokens, qpos_token):
        """
        Args:
            actions: (batch_size, k, action_dim)
            qpos: (batch_size, qpos_dim)
        Returns:
            z: (batch_size, z_latent_dim)
        """
        action_tokens = self.encoder_action_layer(action_tokens)#(batch_size, k, embed_dim)
        qpos_token = self.encoder_joint_layer(qpos_token)#(batch_size, embed_dim)
        qpos_token = torch.unsqueeze(qpos_token, axis=1)#(batch_size, 1, embed_dim)
        input_tokens = torch.cat((action_tokens, qpos_token), dim=1)#(batch_size, k+1, embed_dim)
        output_tokens = self.transformer_encoder(input_tokens)#(batch_size,num_input_tokens + 1  ,embed_dim)
        latent_info = output_tokens[:, 0, :]#(batch_size, embed_dim)
        z_mu_logstd = self.post_layer(latent_info)#(batch_size, z_latent_dim * 2)
        
        z_mu = z_mu_logstd[:, :self.z_latent_dim]#(batch_size, z_latent_dim)
        z_logstd = z_mu_logstd[:, self.z_latent_dim:]#(batch_size, z_latent_dim)
        
        dis = Normal(z_mu, torch.exp(z_logstd))
        z = dis.rsample() # reparameterization trick
        return z
    

    
#--------------------------- Test code ---------------------------
if __name__ == "__main__":
    # Sample arguments
    args = {
        'z_latent_dim': 32,
        'action_dim': 14,
        'embed_dim': 512,
        'state_dim': 14,
        'k': 10  # action chunk size
    }

    # Instantiate the z_encoder
    encoder = z_encoder(args)

    # Define batch size
    batch_size = 100

    # Create dummy input tensors
    action_tokens = torch.randn(batch_size, args['k'], args['action_dim'])  # (batch_size, k, action_dim)
    qpos_token = torch.randn(batch_size, args['state_dim'])  # (batch_size, state_dim)

    # Run the encode function
    z = encoder.encode(action_tokens, qpos_token)

    # Print the shape of the output
    print('Output z shape:', z.shape)
    # Expected output: Output z shape: torch.Size([32, 128])
    print('Output z:', z)