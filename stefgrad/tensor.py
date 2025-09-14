import numpy as np

class Tensor:

    def __init__(self, data, _children=(), _op='', label=""):
        if isinstance(data, list):
            self.data = np.array(data)
            self.grad = np.zeros_like(self.data)
            self.type = "list"
        else:
            self.data = data if isinstance(data, np.ndarray) else np.array(data)
            self.grad = np.zeros_like(self.data)
            self.type = "array"

        self.label = ""

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def get_prev(self):
        return list(self._prev)
    
    def __repr__(self):
        return f"tensor({self.data})"

    def __len__(self):
        return len(self.data) if self.type == "list" else 1

    def __add__(self, other):
        """element-wise addition"""
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(np.add(self.data, other.data), (self, other), "+")

        def _backward():
            self.grad = self.grad + result.grad
            other.grad = other.grad + result.grad
        result._backward = _backward

        return result

    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self - other
    
    def __mul__(self, other):
        """element-wise multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.type == "list" and other.type == "list":
            assert self.data.shape == other.data.shape

        result = Tensor(np.multiply(self.data, other.data), (self, other), "*")

        def _backward():
            self.grad = self.grad + other.data * result.grad
            other.grad = other.grad + self.data * result.grad
        result._backward = _backward

        return result
    
    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1
    
    def transpose(self):
        result = Tensor(np.transpose(self.data), (self, ), "T")

        def _backward():
            self.grad = self.grad + np.transpose(result.grad)
        result._backward = _backward

        return result
    
    def mat_mul(self, other):
        """matrix multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        result = Tensor(np.matmul(self.data, other.data), (self, other), "@")

        def _backward():
            # For matrix multiplication gradient: if C = A @ B, then:
            # dA = dC @ B.T and dB = A.T @ dC
            if result.grad.ndim == 0:
                # Scalar result - use outer product approach
                self.grad = self.grad + result.grad * other.data
                other.grad = other.grad + result.grad * self.data
            else:
                # Vector/Matrix result
                if other.data.ndim == 1:
                    # Vector case: reshape for proper matrix multiplication
                    self.grad = self.grad + np.outer(result.grad, other.data)
                    other.grad = other.grad + np.dot(self.data.T, result.grad)
                else:
                    # Matrix case
                    self.grad = self.grad + np.matmul(result.grad, other.data.T)
                    other.grad = other.grad + np.matmul(self.data.T, result.grad)
        result._backward = _backward

        return result

    def tanh(self):
        result = Tensor(np.tanh(self.data), (self, ), 'tanh')

        def _backward():
            self.grad = self.grad + (1 - np.power(result.data, 2)) * result.grad
        result._backward = _backward

        return result
    
    # def sum(self):
    #     """sum of all elements in the tensor"""
    #     return Tensor([sum([v for v in self.data])])
    
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
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()