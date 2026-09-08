"""
Timestep Embedding Modules for DSRL

These embeddings are used to condition the critic on diffusion timesteps
in the TMRL (Timestep-Modulated RL) variant of DSRL.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.

    Similar to transformer positional encodings, maps a scalar timestep
    to a high-dimensional embedding using sine and cosine functions.
    """

    def __init__(self, dim: int):
        """
        Initialize sinusoidal embedding.

        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal embedding.

        Args:
            x: Timesteps of shape (B,) or (B, 1)

        Returns:
            emb: Embeddings of shape (B, dim)
        """
        if x.dim() > 1:
            x = x.squeeze(-1)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class DualTimestepEncoder(nn.Module):
    """
    Encoder for two timesteps (e.g., language timestep and observation timestep).

    Used in the TMRL variant when conditioning on multiple diffusion timesteps.
    Encodes each timestep separately with sinusoidal embeddings, concatenates them,
    and projects through an MLP.
    """

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        """
        Initialize dual timestep encoder.

        Args:
            embed_dim: Embedding dimension for each timestep
            mlp_ratio: Ratio for MLP hidden dimension
        """
        super().__init__()

        self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)

        # MLP to project concatenated embeddings
        hidden_dim = int(embed_dim * mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Encode two timesteps.

        Args:
            t1: First timestep of shape (B,) or (B, 1)
            t2: Second timestep of shape (B,) or (B, 1)

        Returns:
            emb: Combined embedding of shape (B, embed_dim)
        """
        # Encode each timestep separately
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)

        # Concatenate and project
        temb = torch.cat([temb1, temb2], dim=-1)
        return self.proj(temb)
