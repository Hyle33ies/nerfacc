"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
Modified version that doesn't rely on tinycudann.
"""

from typing import Callable, List, Union, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = 2,
    eps: float = 1e-6,
    derivative: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)  # normalize to [0, 1]
    
    # Contract to [-1, 1]
    x = x * 2 - 1  # scale to [-1, 1]
    
    # Contract to unit sphere
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag > 1
    
    if derivative:
        dev = torch.ones_like(mag)
        dev[mask] = (1 / mag[mask] - 1 / (mag[mask] ** 3 + eps))
        dev = dev[..., None]
        return torch.where(mask[..., None], x / (mag + eps) * dev, x), dev
    
    x[mask] = (2 - 1 / (mag[mask] + eps)) * (x[mask] / mag[mask])
    return x


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) * self.x_dim
            + (self.max_deg - self.min_deg) * 2 * self.x_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class NGPRadianceField(nn.Module):
    """Simplified NGP model without tinycudann dependency."""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        hidden_dim: int = 256,
        geo_feat_dim: int = 15,
    ) -> None:
        super().__init__()
        if isinstance(aabb, (list, tuple)):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.geo_feat_dim = geo_feat_dim
        
        # Position encoder
        self.pos_encoder = SinusoidalEncoder(3, 0, 10, True)
        
        # Direction encoder (if using view directions)
        if self.use_viewdirs:
            self.dir_encoder = SinusoidalEncoder(3, 0, 4, True)
        
        # MLP for density
        self.density_net = nn.Sequential(
            nn.Linear(self.pos_encoder.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 + self.geo_feat_dim),
        )
        
        # MLP for color
        if self.use_viewdirs:
            self.color_net = nn.Sequential(
                nn.Linear(self.geo_feat_dim + self.dir_encoder.latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3),
            )
        else:
            self.color_net = nn.Sequential(
                nn.Linear(self.geo_feat_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3),
            )

    def query_density(self, x, return_feat: bool = False):
        """Query density values for positions x.
        
        Args:
            x: [..., 3] positions
            return_feat: whether to return features as well
        
        Returns:
            density: [..., 1] density values
            features: [..., geo_feat_dim] features (optional)
        """
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)  # normalize to [0, 1]
            x = x * 2 - 1  # scale to [-1, 1]
        
        x_encoded = self.pos_encoder(x)
        h = self.density_net(x_encoded)
        
        density, features = h[..., 0:1], h[..., 1:]
        density = self.density_activation(density)
        
        if return_feat:
            return density, features
        else:
            return density

    def _query_rgb(self, dir: Optional[torch.Tensor], embedding: torch.Tensor, apply_act: bool = True) -> torch.Tensor:
        """Query RGB values for directions and features.
        
        Args:
            dir: [..., 3] view directions
            embedding: [..., geo_feat_dim] geometric features
            apply_act: whether to apply activation to output
        
        Returns:
            rgb: [..., 3] RGB values
        """
        if self.use_viewdirs:
            if dir is None:
                raise ValueError("Directions must be provided if use_viewdirs is True")
            # Normalize directions
            dir = F.normalize(dir, p=2, dim=-1)
            dir_encoded = self.dir_encoder(dir)
            h = torch.cat([embedding, dir_encoded], dim=-1)
        else:
            h = embedding
        
        rgb = self.color_net(h)
        
        if apply_act:
            rgb = torch.sigmoid(rgb)
        
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
    ):
        """Forward pass for the NGP model.
        
        Args:
            positions: [..., 3] positions
            directions: [..., 3] view directions (optional)
        
        Returns:
            rgb: [..., 3] RGB values
            density: [..., 1] density values
        """
        if self.use_viewdirs and directions is None:
            raise ValueError("Directions must be provided if use_viewdirs is True")
        
        density, embedding = self.query_density(positions, return_feat=True)
        
        rgb = self._query_rgb(directions, embedding)
        
        return rgb, density


class NGPDensityField(nn.Module):
    """Simplified NGP density field without tinycudann dependency."""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if isinstance(aabb, (list, tuple)):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.unbounded = unbounded
        
        # Position encoder
        self.pos_encoder = SinusoidalEncoder(3, 0, 10, True)
        
        # MLP for density
        self.density_net = nn.Sequential(
            nn.Linear(self.pos_encoder.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass for the NGP density field.
        
        Args:
            positions: [..., 3] positions
        
        Returns:
            density: [..., 1] density values
        """
        if self.unbounded:
            positions = contract_to_unisphere(positions, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)  # normalize to [0, 1]
            positions = positions * 2 - 1  # scale to [-1, 1]
        
        positions_encoded = self.pos_encoder(positions)
        density = self.density_net(positions_encoded)
        density = self.density_activation(density)
        
        return density 
