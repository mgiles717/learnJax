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