import flax.linen as nn
import jax.numpy as jnp

from models import models_resnet

class SimCLR(nn.Module):
    
    net_type: str
    hidden_dim: int
    out_dim: int=128
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        encoder_model = getattr(models_resnet, self.net_type)
        self.encoder = encoder_model()
        self.projection_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.out_dim)
        ])

    def forward(self, x):
        """For extracting representation"""
        x = self.encoder(x)
        # x = self.projection_head(x)
        return x

    def __call__(self, x):
        """For training"""
        x = self.encoder(x)
        x = self.projection_head(x)
        return x