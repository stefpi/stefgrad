import random

from stefgrad.tensor import Tensor

class Neuron:
    def __init__(self, nin):
        self.w = [random.uniform(-1,1) for _ in range(nin)]
        self.b = random.uniform(-1,1)

class Linear:
    def __init__(self, nin, nout, activation_fn=None):
        import numpy as np
        self.W = Tensor(np.random.uniform(-1, 1, (nin, nout)))
        self.B = Tensor(np.random.uniform(-1, 1, nout))
        self.activation_fn = activation_fn

    def __call__(self, x):
        output = x.mat_mul(self.W) + self.B
        if self.activation_fn is not None:
            return self.activation_fn(output)
        return output
    
    def parameters(self):
        return [self.W, self.B]