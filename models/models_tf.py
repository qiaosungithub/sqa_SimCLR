import flax.linen as nn
import jax.numpy as jnp

def tf_linear(out_dim, use_bias=True):
    return nn.Dense(
        out_dim, 
        use_bias=use_bias, 
        kernel_init=nn.initializers.normal(stddev=0.01)
    )

def tf_batch_norm(use_bias=True):
    return nn.BatchNorm(
        use_running_average=False, 
        # momentum=0.9, # useless if only for projection head
        axis_name="batch",
        use_bias=use_bias,
    )