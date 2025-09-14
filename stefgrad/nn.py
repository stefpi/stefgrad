import random
from stefgrad.tensor import Tensor

class Neuron:
    def __init__(self, nin):
        self.w = [random.uniform(-1,1) for _ in range(nin)]
        self.b = random.uniform(-1,1)
    
    def parameters(self):
        return [self.w, self.b]
    
class Layer:
    def __init__(self, nin, nout, label):
        self.neurons = [Neuron(nin) for i in range(nout)]
        self.W = Tensor(self.neurons[0].w if len(self.neurons) == 1 else [n.w for n in self.neurons])
        self.B = Tensor(self.neurons[0].b if len(self.neurons) == 1 else [n.b for n in self.neurons])

    def __call__(self, x):
        activation = self.W.mat_mul(x) + self.B
        return activation.tanh()
    
    def parameters(self):
        return [self.W, self.B]
    
class MLP:
    def __init__(self, nin, nouts: list):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i+1], label=f"L{i}") for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]