"""
Stochastic Gradient Descent (SGD) optimizer class.
"""

from optimizer import Optimizer

import jax
import jax.numpy as jnp


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)
        
    def step(self):
        loss = None
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad
                
        return loss