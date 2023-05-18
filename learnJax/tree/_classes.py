"""
Base classes for decision tree classifiers and regressors.
"""

from abc import abstractmethod

import jax
import jax.numpy as jnp

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor"
]

class BaseDecisionTree:
    @abstractmethod
    def __init__(
        self,
        criterion,
        splitter,
        max_depth
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        
    def get_depth(self):
       raise NotImplementedError
   
    def get_n_leaves(self):
       raise NotImplementedError

class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(
        self,
        criterion,
        splitter,
        max_depth=None
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth
        )
        
class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(
        self,
        criterion,
        splitter,
        max_depth=None
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth
        )

def main():
    return

if __name__ == "__main__":
    main()