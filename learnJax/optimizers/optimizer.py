"""
Base class optimizer

TODO:  
1. State saving and loading
2. Learning rate scheduling
3. Weight decay
4. Clip Gradients
"""

from abc import abstractmethod

class Optimizer:
    @abstractmethod
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        raise NotImplementedError 
    
if __name__ == "__main__":
    pass
