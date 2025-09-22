from stefgrad.nn.layers import Linear

class MLP:
    def __init__(self, nin, nouts: list, ):
        size = [nin] + nouts
        self.layers = [Linear(size[i], size[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]