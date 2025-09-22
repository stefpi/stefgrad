import numpy as np

class Tensor:

    def __init__(self, data, _children=(), _op='', label=""):
        if isinstance(data, list): self.type = "list"
        elif isinstance(data, np.ndarray): self.type = "array"
        else: self.type = "" 
       
        self.data = data if isinstance(data, np.ndarray) else np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
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
            # Handle broadcasting by summing gradients over broadcasted dimensions
            self_grad = result.grad
            other_grad = result.grad
            
            # Sum over dimensions that were broadcasted for self
            for i in range(result.grad.ndim - self.data.ndim):
                self_grad = np.sum(self_grad, axis=0)
            for i, (result_dim, self_dim) in enumerate(zip(self_grad.shape, self.data.shape)):
                if self_dim == 1 and result_dim > 1:
                    self_grad = np.sum(self_grad, axis=i, keepdims=True)
            
            # Sum over dimensions that were broadcasted for other
            for i in range(result.grad.ndim - other.data.ndim):
                other_grad = np.sum(other_grad, axis=0)
            for i, (result_dim, other_dim) in enumerate(zip(other_grad.shape, other.data.shape)):
                if other_dim == 1 and result_dim > 1:
                    other_grad = np.sum(other_grad, axis=i, keepdims=True)
            
            self.grad = self.grad + self_grad
            other.grad = other.grad + other_grad
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
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, int)
        other = Tensor(other)

        result = Tensor(np.power(self.data, other.data), (self, other), "**")

        def _backward():
            self.grad = self.grad + other.data * np.power(self.data, (other.data - 1)) * result.grad
        result._backward = _backward

        return result
    
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
            
            # Handle different dimensionalities properly
            if self.data.ndim == 1 and other.data.ndim == 1:
                # Vector dot product case
                self.grad = self.grad + result.grad * other.data
                other.grad = other.grad + result.grad * self.data
            elif self.data.ndim == 2 and other.data.ndim == 1:
                # Matrix-vector multiplication case
                self.grad = self.grad + np.outer(result.grad, other.data)
                other.grad = other.grad + np.dot(self.data.T, result.grad)
            elif self.data.ndim == 1 and other.data.ndim == 2:
                # Vector-matrix multiplication case
                self.grad = self.grad + np.dot(result.grad, other.data.T)
                other.grad = other.grad + np.outer(self.data, result.grad)
            else:
                # General matrix-matrix case
                self.grad = self.grad + np.matmul(result.grad, other.data.T)
                other.grad = other.grad + np.matmul(self.data.T, result.grad)
        result._backward = _backward

        return result
    
    def exp(self):
        result = Tensor(np.exp(self.data), (self, ), "e")

        def _backward():
            self.grad = self.grad + np.exp(self.data) * result.grad
        result._backward = _backward
        
        return result

    def tanh(self):
        result = Tensor(np.tanh(self.data), (self, ), 'tanh')

        def _backward():
            self.grad = self.grad + (1 - np.power(result.data, 2)) * result.grad
        result._backward = _backward

        return result
    
    def sigmoid(self):
        def _logistic(x):
            return 1 / (1 + np.exp(np.multiply(-1, x)))
        
        result = Tensor(_logistic(self.data), (self, ), 'sigmoid')

        def _backward():
            # Derivative of sigmoid: σ(x) * (1 - σ(x))
            self.grad = self.grad + result.data * (1 - result.data) * result.grad
        result._backward = _backward

        return result
    
    def relu(self):
        result = Tensor(np.maximum(0, self.data), (self, ), "ReLU")

        def _backward():
            # ReLU derivative: 1 if x > 0, else 0
            self.grad = self.grad + (self.data > 0).astype(float) * result.grad
        result._backward = _backward

        return result

    def softmax(self):
        # stable softmax along classes axis
        stable_data = self.data - np.max(self.data, axis=-1, keepdims=True)
        exp_values = np.exp(stable_data)
        exp_values_sum = np.sum(exp_values, axis=-1, keepdims=True)

        result = Tensor(exp_values/exp_values_sum, (self,), "softmax")

        def _backward():
            K = self.data.shape[-1]
            I = np.eye(K, dtype=self.data.dtype)                      # (K, K)
            diag = self.data[..., :, None] * I                        # (..., K, K) == diag(s)
            outer = np.einsum('...i,...j->...ij', self.data, self.data)       # (..., K, K) == s s^T
            out = diag - outer 
            self.grad = self.grad + np.einsum('bij,bi->bj', out, result.grad) 

        result._backward = _backward
        return result
    
    def logsumexp(self, axis=-1):
        m = Tensor(np.max(self.data, axis=axis, keepdims=True))
        return self - ((self - m).exp().sum(axis=axis, keepdims=True).log() + m)
    
    def log(self):
        """Natural logarithm with numerical stability"""
        result = Tensor(np.log(self.data), (self, ), "log")

        def _backward():
            self.grad = self.grad + (self.data**-1) * result.grad

        result._backward = _backward

        return result
    
    def sum(self, axis=-1, keepdims=False):
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")
        result._sum_axis = axis
        result._sum_keepdims = keepdims

        def _backward():
            g = result.grad  # upstream grad
            ax = result._sum_axis
            if ax is None:
                # summed all elements
                grad = np.ones_like(self.data) * g
            else:
                if not isinstance(ax, tuple):
                    ax = (ax,)
                # normalize negative axes
                ax = tuple(a % self.data.ndim for a in ax)
                # if keepdims=False in forward, add back singleton dims
                if not result._sum_keepdims:
                    for a in sorted(ax):
                        g = np.expand_dims(g, axis=a)
                # now g is broadcastable to self.data
                grad = np.ones_like(self.data) * g
            self.grad = self.grad + grad

        result._backward = _backward
        return result
    
    def mean(self):
        result = Tensor(np.mean(self.data), (self, ), "mean")

        def _backward():
            # For mean, gradient is broadcasted to all elements and divided by total number of elements
            numel = self.data.size  # total number of elements
            grad_broadcasted = np.full_like(self.data, result.grad / numel)
            self.grad = self.grad + grad_broadcasted
        result._backward = _backward

        return result
    
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