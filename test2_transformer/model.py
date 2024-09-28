
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# ----------------------------------------------  Main model part  -----------------------------------------------------
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        # Model parameters
        self.depth = args["depth"]
        self.n_heads = args["n_heads"]
        self.embed_dim = args["entity_dim"]
        self.num_input_tokens = args["num_input_tokens"]
        self.out_dim = args["out_dim"]
        # Standard Positional Encoding        
        self.pos_encoding = PositionalEncoding(self.embed_dim, dropout=0.1, max_len=5000)
        
        # naive positional embedding
        # self.pos_embed = nn.Parameter(
        #         torch.zeros(1, self.num_input_tokens, self.embed_dim)
        # )#Dim(1,num_input_tokens,embed_dim)

        # Transformer Encoder
        self.transformer_encoder_blocks = nn.ModuleList()
        for _ in range(self.depth):
            self.transformer_encoder_blocks.append(TransformerEncoderBlock(dim = self.embed_dim , n_heads=self.n_heads))

        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, x):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        #x: Dim (n_samples, num_tokens, embed_dim)
        batch_size = x.shape[0]#batch_size = n_samples
        cls_token = nn.Parameter(torch.zeros(batch_size, 1, self.embed_dim))#Dim(n_samples,1,embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_encoding(x)
        for encoder_block in self.transformer_encoder_blocks:
            x = encoder_block(x)
        # Dim (n_samples, num_tokens, embed_dim)     
        x = self.norm(x)#Dim(n_samples, num_tokens,embed_dim)
        cls_token_final = x[:, 0]  # just the CLS token ;Dim(n_samples,embed_dim)
        pred = self.head(cls_token_final)#Dim(n_samples,out_dim)
        return pred
    
class TransformerEncoderBlock(nn.Module):
    """Transformer block.
    Parameters
    ----------
    dim : int
        Embeddinig dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attention
        Attention module.
    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0,):
        super().__init__()
        mlp_ratio=4.0
        hidden_features = int(dim*mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Self_Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )
    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
    
class Self_Attention(nn.Module):
    """Multi Head Self Attention  + Dense Layer.
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.
    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.
    W_q : nn.Linear
        Linear projection for the query
    W_k : nn.Linear
        Linear projection for the keywords
    W_v : nn.Linear
        Linear projection for the value

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_oneHead = dim//n_heads
        self.head_scale = self.dim_oneHead**-0.5
        #Layers
        self.W_q = nn.Linear(in_features=dim, out_features=dim, bias=qkv_bias)
        self.W_k = nn.Linear(in_features=dim, out_features=dim, bias=qkv_bias)
        self.W_v = nn.Linear(in_features=dim, out_features=dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj_drop = nn.Dropout(proj_p)
        self.proj = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim),
            nn.ReLU()
        )

    def forward(self,x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_tokens , dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_tokens , dim)`.
        """
        n_samples, n_tokens, dim = x.shape #n_tokens = n_patches + 1 

        if dim!= self.dim:
            raise ValueError

        #1. -----Linear Projection-----
        #x: (n_samples, n_tokens, dim)
        Q = self.W_q(x)#shape:(n_samples, n_tokens, dim)
        K = self.W_k(x)#shape:(n_samples, n_tokens, dim)
        V = self.W_v(x)#shape:(n_samples, n_tokens, dim)

        #Reshape to split the heads
        Q = Q.reshape(n_samples, n_tokens, self.n_heads, self.dim_oneHead) # shape: (n_samples, n_tokens, n_heads, dim_oneHead)
        K = K.reshape(n_samples, n_tokens, self.n_heads, self.dim_oneHead) # shape: (n_samples, n_tokens, n_heads, dim_oneHead)
        V = V.reshape(n_samples, n_tokens, self.n_heads, self.dim_oneHead) # shape: (n_samples, n_tokens, n_heads, dim_oneHead)

        #Permute to align the formular in the paper
        Q = Q.permute(0, 2, 1, 3)  # shape:(n_samples, n_heads, n_tokens, dim_oneHead)
        K = K.permute(0, 2, 1, 3)  # shape:(n_samples, n_heads, n_tokens, dim_oneHead)
        V = V.permute(0, 2, 1, 3)  # shape:(n_samples, n_heads, n_tokens, dim_oneHead)
        
        #Further permutation to aline with the formular in the paper
        Q = Q.permute(0, 1, 3, 2)  # shape:(n_samples, n_heads, dim_oneHead, n_tokens)
        K = K.permute(0, 1, 3, 2)  # shape:(n_samples, n_heads, dim_oneHead, n_tokens)
        V = V.permute(0, 1, 3, 2)  # shape:(n_samples, n_heads, dim_oneHead, n_tokens)

        K_T = K.transpose(-2,-1) # shape:(n_samples, n_heads, n_tokens, dim_oneHead)

        #2. -----Calculate Attention Weight-----
        alpha_atten = ((K_T @ Q) * self.head_scale).softmax(dim = -2)# shape:(n_samples, n_heads, n_tokens, n_tokens)
        alpha_atten = self.attn_drop(alpha_atten)

        #3. -----Calculate the context feature-----
        C = V @ alpha_atten # shape:(n_samples, n_heads, dim_oneHead, n_tokens)

        #4. -------A dense Layer-------
        #  permutation back
        C = C.permute(0, 1, 3, 2) # shape:(n_samples, n_heads, n_tokens, dim_oneHead)
        C = C.transpose(1,2) # shape:(n_samples, n_tokens1, n_heads, dim_oneHead)
        #merge the heads
        C = C.flatten(2)# shape:(n_samples, n_tokens, dim) ; n_heads * dim_oneHead = dim
        x = self.proj(C)# shape:(n_samples, n_tokens, dim)
        x = self.proj_drop(x)# shape:(n_samples, n_tokens, dim)

        return x
    
class MLP(nn.Module):
    """Multilayer perceptron.
    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Number of nodes in the hidden layer.
    out_features : int
        Number of output features.
    p : float
        Dropout probability.
    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.
    act : nn.GELU
        GELU activation function.
    fc2 : nn.Linear
        The second linear layer.
    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(p)

    def forward(self,x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        #x: (n_samples, n_patches + 1, in_features)
        x = self.fc1(x)# (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)# (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)# (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)# (n_samples, n_patches + 1, out_features)
        x = self.drop(x)# (n_samples, n_patches + 1, out_features)

        return x
    
#---------------------------------------------------  Positional Encoding  -----------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()  
        self.dropout = nn.Dropout(p=dropout)  # 初始化dropout层
        
        # 计算位置编码并将其存储在pe张量中
        pe = torch.zeros(max_len, embed_dim)                #Dim:(max_len , embed_dim) 创建一个max_len x embed_dim的全零张量
        token_pos = torch.arange(0, max_len).unsqueeze(1)  # 生成0到max_len-1的整数序列，并添加一个维度
        # 计算div_term，用于缩放不同位置的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             -(math.log(10000.0) / embed_dim))
 
        # 使用正弦和余弦函数生成位置编码，对于d_model的偶数索引，使用正弦函数；对于奇数索引，使用余弦函数。
        pe[:, 0::2] = torch.sin(token_pos * div_term) #Dim:(max_len , embed_dim)
        pe[:, 1::2] = torch.cos(token_pos * div_term) #Dim:(max_len , embed_dim)
        pe = pe.unsqueeze(0)                  # 在第一个维度添加一个维度，以便进行批处理
        self.register_buffer('pe', pe)        # 将位置编码张量注册为缓冲区，以便在不同设备之间传输模型时保持其状态
        
    # 定义前向传播函数
    def forward(self, x):
        # 将输入x与对应的位置编码相加
        x = x + self.pe[:, :x.size(1)].to(x.device)
        # 应用dropout层并返回结果
        return self.dropout(x)
#---------------------------------------------------Test the model-----------------------------------------------------
if __name__ == "__main__":
    # Define the model parameters
    args = {
        "depth": 2,
        "n_heads": 4,
        "entity_dim": 64,
        "num_input_tokens": 10,
        "out_dim": 10
    }
    # Instantiate the model
    model = TransformerEncoder(args)

    # Create sample input data
    batch_size = 8
    num_input_tokens = args["num_input_tokens"]
    entity_dim = args["entity_dim"]

    x = torch.randn(batch_size, num_input_tokens, entity_dim)

    # Run the model
    output = model(x)

    # Print the output shape
    print("Output shape:", output.shape)  # Should be (batch_size, out_dim)