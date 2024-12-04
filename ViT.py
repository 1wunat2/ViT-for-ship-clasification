"""
ViT.py contains the architecture for the Visual Transformer
The structure and style follows from the code posted during tutorial for AMATH 445
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

@torch.inference_mode()
def perform_inference(model, dataloader, device: str, loss_fn=None):
    """
    Perform inference on given dataset using given model on the specified device. If loss_fn is
     provided, it also computes the loss and returns [y_preds, y_true, losses].
    """
    model.eval()  # Set the model to evaluation mode, this disables training specific operations
    y_preds = []
    y_true = []
    losses = []

    print("[inference.py]: Running inference...")
    for i, batch in tqdm(enumerate(dataloader)):
        inputs = batch["img"].to(device)
        outputs = model(inputs)
        if loss_fn is not None:
            labels = batch["label"].to(device)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            y_true.append(labels.cpu().numpy())

        preds = F.softmax(outputs.detach().cpu(), dim=1).argmax(dim=1)
        y_preds.append(preds.numpy())

    model.train()  # Set the model back to training mode
    y_true, y_preds = np.concatenate(y_true), np.concatenate(y_preds)
    return y_true, y_preds, np.mean(losses) if losses else None

def init_weights(module):
    """
    Initialise weights of given module using Kaiming Normal initialisation for linear and
    convolutional layers, and zeros for bias.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Head(nn.Module):
    """
    Single head for attention based learning
    """
    def __init__(self, head_size, n_embed, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k, q, v = self.key(x), self.query(x), self.value(x)
        out = F.softmax(q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5, dim=-1)
        out = self.dropout(out)
        out = out @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Combines multiple heads to get multihead attention based learning
    """
    def __init__(self, head_size, n_heads, n_embed, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MLP(nn.Module):
    """
    Basic MLP, designed with Visual Transformer in mind by increading and then decreasing the number of neurons
    """
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(inplace=True),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Block combines the multihead attention and MLP defined
    """
    def __init__(self, n_heads, n_embed, dropout=0.2):
        super().__init__()
        head_size = n_embed // n_heads
        self.attention = MultiHeadAttention(head_size, n_heads, n_embed, dropout)
        self.ffwd = MLP(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class PatchEmbedding(nn.Module):
    """
    Patch Embedding Module:
    - Splits an image into patches
    - Projects patches into a higher-dimensional embedding space
    """
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    VisonTransformer combines the Block and embedding to create the VisualTransformer
    """
    def __init__(
        self,
        in_channels,
        patch_size,
        n_patches,
        emb_size,
        n_heads,
        n_layers,
        n_classes,
        class_freqs,
        dropout=0.2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_heads, emb_size, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, n_classes)

        self.head.bias = nn.Parameter(torch.log(class_freqs))
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x