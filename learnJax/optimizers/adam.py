"""

"""

from optimizer import Optimizer

import jax
import jax.numpy as jnp

class Adam(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)


def main():
    print("Hello World!")
    x = jnp.arange(10)
    print(x)

if __name__ == "__main__":
    main()