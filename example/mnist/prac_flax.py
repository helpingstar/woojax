from flax import linen as nn
import jax.numpy as jnp
import jax

x = jnp.ones((1, 11, 11, 64))


class AvgPool(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))


rng = jax.random.key(0)

avg_pool = AvgPool()
param = avg_pool.init(rng, x)
print(avg_pool.apply(param, x).shape)
