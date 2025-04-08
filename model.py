import torch
import torch.nn as nn
from einops import rearrange

class patch_embedding(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_h = config['img_h']
        self.img_w = config['img_w']
        self.patch_h = config['patch_h']
        self.patch_w = config['patch_w']
        self.in_channels = config['in_channels'] 
        self.embedding_dim = config['embedding_dim'] # D
        
        self.sequence_length = (self.img_h // self.patch_h) * (self.img_w // self.patch_w) # N
        
        self.patch_dim = self.patch_h * self.patch_w * self.in_channels
        
        
        # FC that coverts the output of patch embedding from patch dimension into embedding dimension
        self.patch_embedding_fc = nn.Sequential(nn.LayerNorm(self.patch_dim),
                                                nn.Linear(in_features = self.patch_dim,
                                                          out_features = self.embedding_dim),
                                                nn.LayerNorm(self.embedding_dim))
        
        # Classification tokens
        self.cls_tokens = nn.Parameter(torch.randn((1, self.embedding_dim))) # 1xD
        
        # Positional Embedding tokens
        self.embedding_tokens = nn.Parameter(torch.zeros((self.sequence_length + 1, self.embedding_dim))) # N+1 x D
        
        # Dropout layers
        self.patch_embedding_dropout = nn.Dropout(config['patch_embedding_dropout'])
    
    def forward(self, x):
        """
            x: BxCxHxW, torch tensor, -1 to 1, float, RGB
        """
        B, C, H, W = x.shape
        out = x # BxCxHxW
        
        # Patch Embedding block
        out = rearrange(out, 'B C (nh h) (nw w) -> B (nh nw) (h w C)',
                        h = self.patch_h, w = self.patch_w) # B x N x (hxwxC)
        
        # Convert to Embedding Dimensions
        out = self.patch_embedding_fc(out) # B x N x D
        
        # Add Classification tokens
        out = torch.cat([self.cls_tokens.repeat(B, 1, 1), out], dim=1) # B x (N+1) x D
        
        # Add Positional Embedding
        out = out + self.embedding_tokens # B x (N+1) x D
        
        # Dropout
        out = self.patch_embedding_dropout(out)
        
        return out
    







class attention_block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim'] # D
        self.num_heads = config['num_heads']
        self.head_dim = config['head_dim']
        
        # Attention block to compute K, Q and V
        self.attention_fc = nn.Linear(in_features = self.embedding_dim,
                                      out_features = self.num_heads * 3 * self.head_dim)
        
        # Dropout block for the attention maps
        self.att_map_dropout = nn.Dropout(config['att_map_dropout'])
        
        # FC to revert the dimensions of the output into embedding_dim
        self.attention_embedding_fc = nn.Sequential(nn.Linear(in_features = self.num_heads * self.head_dim,
                                                              out_features = self.embedding_dim),
                                                    nn.Dropout(config['att_map_dropout']))
    
    def forward(self, x):
        """
            x: B x (N+1) x D
        """
        
        out = x # B x (N+1) x D
        
        # Compute K, Q and V
        out = self.attention_fc(out) # B x (N+1) x (num_heads x 3 x H)
        
        # Split into K, Q, V
        K, Q, V = torch.split(out, self.num_heads*self.head_dim, dim=-1) # B x (N+1) x (num_heads x H)
        
        # Rearrange the tensors
        K = rearrange(K, 'B N (num_heads H) -> B num_heads N H', num_heads = self.num_heads) # B x num_heads x (N+1) x H
        Q = rearrange(Q, 'B N (num_heads H) -> B num_heads N H', num_heads = self.num_heads) # B x num_heads x (N+1) x H
        V = rearrange(V, 'B N (num_heads H) -> B num_heads N H', num_heads = self.num_heads) # B x num_heads x (N+1) x H
        
        # Compute attention map
        att_map = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim ** 0.5) # B x num_heads x (N+1) x (N+1)
        att_map = torch.nn.functional.softmax(att_map, dim=-1) # B x num_heads x (N+1) x (N+1)
        att_map = self.att_map_dropout(att_map)
        
        # Compute output
        output = torch.matmul(att_map, V) # B x num_heads x (N+1) x H
        output = rearrange(output, 'B num_heads N H -> B N (num_heads H)') # B x (N+1) x (num_heads x H)
        output = self.attention_embedding_fc(output) # B x (N+1) x D
        
        return output
    
    
    




class transformer_layer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.transformer_hidden_dim = config['transformer_hidden_dim']
        
        
        self.multi_head_attention = nn.Sequential(nn.LayerNorm(self.embedding_dim),
                                                  attention_block(config = self.config))
        
        self.transformer_fc = nn.Sequential(nn.LayerNorm(self.embedding_dim),
                                            nn.Linear(in_features = self.embedding_dim,
                                                      out_features = self.transformer_hidden_dim),
                                            nn.GELU(),
                                            nn.Dropout(config['transformer_dropout']),
                                            nn.Linear(in_features = self.transformer_hidden_dim,
                                                      out_features = self.embedding_dim),
                                            nn.Dropout(config['transformer_dropout']))
    
    def forward(self, x):
        
        """
            x: B x (N+1) x D
        """
        out = x 
        out = self.multi_head_attention(out) + out
        out = self.transformer_fc(out) + out
        
        return out





class vision_transformer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.num_classes = config['num_classes']
        
        self.patch_embedding = patch_embedding(config = config)
        self.transformer_blocks = nn.ModuleList([transformer_layer(config = config)
                                                 for _ in range(config['num_transformer_blocks'])])
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        self.cls_fc = nn.Linear(in_features = self.embedding_dim,
                                out_features = self.num_classes)
        
    
    def forward(self, x):
        
        """
            x: BxCxHxW
        """
        
        out = x
        out = self.patch_embedding(out) # B x (N+1) x D
        
        for block in self.transformer_blocks:
            out = block(out) # B x (N+1) x D
        
        out = self.layer_norm(out) # B x (N+1) x D
        
        cls_token = out[:, 0, :]  # B x D
        
        result = self.cls_fc(cls_token) # B x num_classes
        
        return result
        
    
    
        
    
        

