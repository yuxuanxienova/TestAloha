import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test5_ACT_policy.test5_3_zEncoder import z_encoder
from test5_ACT_policy.test5_2_backbone import ImagePreprocessModel
from test2_transformer.Transformer import TransformerEncoder, TransformerDecoder
import torch.nn as nn
import torch
class ACT_policy(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        #initialize the arguments
        self.camera_names = args['camera_names']
        self.qpos_dim = args['qpos_dim']
        self.z_latent_dim = args['z_latent_dim']
        self.embed_dim = args['embed_dim']
        self.action_dim = args['action_dim']
        self.action_chunk_size = args['action_chunk_size']
        self.num_input_tokens = args['num_input_tokens']
        self.num_output_tokens = args['num_output_tokens']
        self.out_dim = args['out_dim']
        
        #initialize the z_encoder, image_preprocess_model, transformer_encoder, transformer_decoder
        
        #z_encoder
        z_encoder_args = {
        'z_latent_dim': self.z_latent_dim,
        'action_dim': self.action_dim,
        'embed_dim': self.embed_dim,
        'qpos_dim': self.qpos_dim,
        'k': self.action_chunk_size  # action chunk size
        }
        self.z_encoder = z_encoder(z_encoder_args)
        self.image_preprocess_model = ImagePreprocessModel()
        
        #Transformer Encoder
        transformer_encoder_args = {
            "depth": args["depth"],
            "n_heads": args["n_heads"],
            "entity_dim": self.embed_dim,
            "num_input_tokens": self.num_input_tokens,
            "out_dim": self.embed_dim,  # Encoder outputs embeddings of size embed_dim
        }
        self.transformer_encoder = TransformerEncoder(transformer_encoder_args)
        
        #Transformer Decoder
        transformer_decoder_args = {
            "depth": args["depth"],
            "n_heads": args["n_heads"],
            "embed_dim": self.embed_dim,
            "num_output_tokens": self.num_output_tokens,
            "out_dim": self.out_dim,
        }
        self.transformer_decoder = TransformerDecoder(transformer_decoder_args)
        
        #Project the z and qpos to the embed_dim
        self.project_z = nn.Linear(self.z_encoder.z_latent_dim, self.transformer_encoder.embed_dim)
        self.project_qpos = nn.Linear(self.qpos_dim, self.transformer_encoder.embed_dim)
    def forward(self,data):
        image_data, qpos_data, action_data, is_pad = data
        #data[0]=image_data: Dim(batch_size, num_cameras=3, 3, 480, 640)
        #data[1]=qpos_data: Dim(batch_size, 14)
        #data[2]=action_data: Dim(batch_size, chunk_size, 16)
        #data[3]=is_pad: Dim(batch_size, chunk_size)
        assert image_data.shape[1] == len(self.camera_names)
        assert self.qpos_dim == qpos_data.shape[1]
        self.batch_size = image_data.shape[0]
        
        #1. Infer the latent state z
        z = self.z_encoder.encode(action_data, qpos_data)#Dim:(batch_size, z_latent_dim)
        z_token = self.project_z(z).unsqueeze(1)#Dim:(batch_size, 1, embed_dim)
        
        #2. Preprocess the image data amd qpos data
        list_image_feature_tokens = []
        for cam_id, cam_name in enumerate(self.camera_names):
            image_batch = image_data[:, cam_id]# Dim(batch_size, 3, 480, 640)
            image_feature_tokens = self.image_preprocess_model(image_batch)#Dim:(batch_size, 300=15*20 , embed_dim)
            list_image_feature_tokens.append(image_feature_tokens)
        combined_image_feature_tokens = torch.cat(list_image_feature_tokens, dim=1)#Dim:(batch_size, num_cameras*300=3*15*20 , embed_dim)
       
        
        qpos_tokens = self.project_qpos(qpos_data).unsqueeze(1)#Dim:(batch_size, 1, embed_dim)\
        
        #3. Cancatinating all input tokens
        encoder_input_tokens = torch.cat((z_token, combined_image_feature_tokens, qpos_tokens), dim=1)#Dim:(batch_size, num_cameras*300+2, embed_dim)
        #decoder input is only the positional encoding, so input tokens are zeros
        decoder_input_tokens = torch.zeros((self.batch_size, self.action_chunk_size, self.embed_dim))#Dim:(batch_size, action_chunk_size, embed_dim)
        # encoder_input_tokens: (batch_size, num_input_tokens, embed_dim)
        # decoder_input_tokens: (batch_size, num_output_tokens, embed_dim)
        
        # Embedding source input
        encoder_output = self.transformer_encoder(encoder_input_tokens)  # (batch_size, num_input_tokens + 1, embed_dim)
        
        # Remove the CLS token from encoder output if not needed
        encoder_output = encoder_output[:, 1:, :]  # (batch_size, num_input_tokens, embed_dim)
        
        # Run the decoder
        decoder_output = self.transformer_decoder(decoder_input_tokens, encoder_output, tgt_mask=None)#(batch_size, num_output_tokens, out_dim)
        
        return decoder_output
            
        
if __name__ == "__main__":
    # Sample arguments
    from constants import SIM_TASK_CONFIGS
    task_name='sim_transfer_cube_scripted'
    task_config = SIM_TASK_CONFIGS[task_name]
    camera_names = task_config['camera_names']
    # Define the model parameters
    args = {
        'camera_names': camera_names,
        'qpos_dim': 14,
        'z_latent_dim': 32,
        'embed_dim': 512,
        'action_dim': 16,
        'action_chunk_size': 5,
        'num_input_tokens': len(camera_names) * 300 + 2,  # 300 tokens per camera, plus z and qpos tokens
        'num_output_tokens': 5,  # Same as action_chunk_size
        'out_dim': 64,
        'depth': 2,
        'n_heads': 4
    }
    
    # Initialize the policy
    policy = ACT_policy(args)
    
    # Create sample input data
    batch_size = 8
    num_cameras = len(camera_names)
    image_height = 480
    image_width = 640
    qpos_dim = args['qpos_dim']
    action_chunk_size = args['action_chunk_size']
    action_dim = args['action_dim']
    
    # Image data: (batch_size, num_cameras, 3, 480, 640)
    image_data = torch.randn(batch_size, num_cameras, 3, image_height, image_width)
    
    # qpos data: (batch_size, qpos_dim)
    qpos_data = torch.randn(batch_size, qpos_dim)
    
    # action data: (batch_size, action_chunk_size, action_dim)
    action_data = torch.randn(batch_size, action_chunk_size, action_dim)
    
    # is_pad: (batch_size, action_chunk_size)
    is_pad = torch.zeros(batch_size, action_chunk_size, dtype=torch.bool)
    
    # Prepare the input data tuple
    data = (image_data, qpos_data, action_data, is_pad)
    
    # Run the policy
    output = policy(data)
    
    # Print the output shape
    print("Decoder Output shape:", output.shape)  # Should be (batch_size, action_chunk_size, out_dim)
        