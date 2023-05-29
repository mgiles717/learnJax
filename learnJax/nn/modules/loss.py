"""
Loss functions for neural networks.
"""

from .module import Module

import jax.numpy as jnp
from typing import Callable, optional

__all__ = ['CrossEntropyLoss', 'MeanSquaredError']

class Loss(Module):
    def __init__(self) -> None:
       super().__init__()
        
class MeanSquaredError(Loss):
    """
    Mean Squared Error is the sum of all errors squared and divided by the number of samples.
    math: 1/n 8 Sum (y_true - y_pred)^2
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, y_true, y_pred):
        return jnp.mean(jnp.square(y_true - y_pred))
    
class CrossEntropy(Loss):
    """
    Cross-Entropy is a loss function, which calculates the 
    math: -Sum (y_true * log(y_pred))
    """
    
    def __init__(self):
        super().__init__()
        raise NotImplementedError
        
    def forward(self, y_true, y_pred):
        return -jnp.sum(y_true * jnp.log(y_pred))