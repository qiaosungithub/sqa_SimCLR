import flax.linen as nn
import jax.numpy as jnp

from models import models_resnet
from models.models_tf import tf_linear, tf_batch_norm

class SimCLR(nn.Module):
    
    net_type: str
    hidden_dim: int
    out_dim: int=128
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        encoder_model = getattr(models_resnet, self.net_type)
        self.encoder = encoder_model()
        self.projection_head = nn.Sequential([
            tf_linear(self.hidden_dim),
            tf_batch_norm(),
            nn.relu,
            tf_linear(self.hidden_dim),
            tf_batch_norm(),
            nn.relu,
            tf_linear(self.out_dim, use_bias=False),
            tf_batch_norm(use_bias=False),
        ])

    def forward(self, x):
        """For extracting representation"""
        x = self.encoder(x, train=False)
        # x = self.projection_head(x)
        return x

    def __call__(self, x):
        """For training"""
        x = self.encoder(x, train=True)
        x = self.projection_head(x)
        return x