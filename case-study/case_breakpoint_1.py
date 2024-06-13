import jax
import jax.numpy as jnp

t = 2


def breakpoint_if_nonfinite(x):
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        # (jdb) 에서 f의 z, y 와 전역변수 t 참조는 불가능하다.
        jax.debug.breakpoint()

    jax.lax.cond(is_finite, true_fn, false_fn, x)


@jax.jit
def f(x, y):
    z = x / y
    breakpoint_if_nonfinite(z)
    return z


f(2.0, 0.0)  # ==> No breakpoint
