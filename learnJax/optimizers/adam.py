"""

"""

from optimizer import Optimizer

import jax
import jax.numpy as jnp

from typing import Any

class Adam(Optimizer):
    """
    params: list of parameters to optimize
    lr: learning rate
    beta_1: exponential decay rate for the first moment estimates
    beta_2: exponential decay rate for the second moment estimates
    eps: term added to the denominator to improve numerical stability
    All parameters are set to the default of the Adam paper suggested values.
    """
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        
        self.m = [jnp.zeros_like(param) for param in self.params]
        self.v = [jnp.zeros_like(param) for param in self.params]
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        self.m = [self.beta_1 * m + (1 - self.beta_1) * grad for m, grad in zip(self.m, grads)]
        self.v = [self.beta_2 * v + (1 - self.beta_2) * grad**2 for v, grad in zip(self.v, grads)]
        m_hat = [m / (1 - self.beta_1**self.t) for m in self.m]
        v_hat = [v / (1 - self.beta_2**self.t) for v in self.v]
        self.params = [param - self.lr * m / (jnp.sqrt(v) + self.eps) for param, m, v in zip(self.params, m_hat, v_hat)]
        return self.params

if __name__ == "__main__":
    pass