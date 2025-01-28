import torch
import torch.nn as nn
import torch.nn.functional as F
# import math
import numpy as np

from modules.quantization_cpu_np_infer import QLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from InferenceConfig import args

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, args, embed_dim, num_heads, mlp_dim, layer_id, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Layer normalization before and after attention
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            # nn.Linear(embed_dim, mlp_dim),
            QLinear(in_features=embed_dim, out_features=mlp_dim,
                    wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                    RRAM=args.RRAM,
                    subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                    name='FC'+str(layer_id)+'_', model=args.model),

            nn.GELU(),
            nn.Dropout(dropout),
            # nn.Linear(mlp_dim, embed_dim),
            QLinear(in_features=mlp_dim, out_features=embed_dim,
                    wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                    RRAM=args.RRAM,
                    subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                    name='FC'+str(layer_id+1)+'_', model=args.model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization before and after MLP
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)  # Residual connection
        
        # Feed-forward network
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)  # Residual connection
        
        return x

class ViT(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_channels=3, embed_dim=256, num_heads=4, num_classes=10, depth=8, mlp_dim=256):
        super(ViT, self).__init__()
        
        # Patch embedding: divide image into patches and map to embedding space
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * (patch_size ** 2)
        # self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.proj = QLinear(in_features=self.patch_dim, out_features=embed_dim,
                    wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                    RRAM=args.RRAM,
                    subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                    name='FC0_', model=args.model)
        
        # Learnable class token and position embeddings
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        self.EncoderLayers = nn.ModuleList()
        self.layer_id = 1
        for _ in range(depth):
            self.EncoderLayers.append(CustomTransformerEncoderLayer(args, embed_dim, num_heads, mlp_dim, self.layer_id))
            self.layer_id += 2
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            # nn.Linear(embed_dim, num_classes)
            QLinear(in_features=embed_dim, out_features=num_classes,
                    wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                    RRAM=args.RRAM,
                    subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                    name='FC'+str(self.layer_id)+'_', model=args.model),
        )

        # self

    def forward(self, x):
        bs, _, h, w = x.size()
        assert h == w == 224, "Input size must be (bs, 3, 224, 224)"
        
        # Extract patches and flatten
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(bs, -1, self.patch_dim)
        
        # Linear projection to embedding space
        embeddings = self.proj(patches)
        
        # Concatenate class token and add positional encoding
        class_token = self.class_token.expand(bs, -1, -1)  # (bs, 1, embed_dim)
        embeddings = torch.cat((class_token, embeddings), dim=1)  # (bs, num_patches+1, embed_dim)
        embeddings += self.pos_embed
        embeddings = self.dropout(embeddings)
        
        # Pass through Transformer encoder
        for layer in self.EncoderLayers:
            embeddings = layer(embeddings)
        
        # Classification token output
        class_out = embeddings[:, 0]  # Extract class token
        logits = self.mlp_head(class_out)
        return logits

