import jax
import jax.numpy as jnp

t = 2


@jax.jit
def f(x, y):
    z = x / y

    def breakpoint_if_nonfinite(x):
        is_finite = jnp.isfinite(x).all()

        def true_fn(x):
            pass

        def false_fn(x):
            # breakpoint에서 z, y, t의 참조가 가능할까?
            jax.debug.breakpoint()

        jax.lax.cond(is_finite, true_fn, false_fn, x)

    breakpoint_if_nonfinite(z)
    return z


f(2.0, 0.0)  # ==> No breakpoint
