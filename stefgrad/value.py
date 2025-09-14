import math

class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0

        self._backward = lambda: None # store the chain of derivatives
        self._prev = set(_children) # save the children of the Value to go backwards to them
        self._op = _op # save what operation made this value
    
    def __repr__(self):
        return f"{self.data}"
    
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += result.grad 
            other.grad += result.grad
        result._backward = _backward

        return result

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.grad * result.grad # (d objective function / d result) * (d result / d self)
            other.grad += self.grad * result.grad
        result._backward = _backward

        return result
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        result = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * result.grad
        result._backward = _backward

        return result
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        t = (math.exp(2*self.data - 1)/(math.exp(2*self.data) + 1))
        result = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - self.grad**2) * result.grad
        result._backward = _backward

        return result
    
    def exp(self):
        result = Value(math.exp(self.data), (self, ), 'exp')

        def _backward():
            self.grad += result.data * result.grad
        self._backward = _backward

        return result