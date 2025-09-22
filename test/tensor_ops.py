import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from stefgrad.tensor import Tensor


class TestTensorShapes:
    """Test tensor creation and operations with different shapes."""
    
    def test_scalar_tensor(self):
        """Test scalar tensor creation and basic properties."""
        t = Tensor(5.0)
        assert t.data.shape == ()
        assert t.grad.shape == ()
        assert t.data.item() == 5.0
        
    def test_vector_tensor(self):
        """Test vector tensor creation."""
        t = Tensor([1.0, 2.0, 3.0])
        assert t.data.shape == (3,)
        assert t.grad.shape == (3,)
        np.testing.assert_array_equal(t.data, np.array([1.0, 2.0, 3.0]))
        
    def test_matrix_tensor(self):
        """Test matrix tensor creation."""
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.data.shape == (2, 2)
        assert t.grad.shape == (2, 2)
        np.testing.assert_array_equal(t.data, np.array([[1.0, 2.0], [3.0, 4.0]]))
        
    def test_3d_tensor(self):
        """Test 3D tensor creation."""
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        t = Tensor(data)
        assert t.data.shape == (2, 2, 2)
        assert t.grad.shape == (2, 2, 2)


class TestArithmeticOperations:
    """Test basic arithmetic operations and their gradients."""
    
    def test_addition_scalars(self):
        """Test scalar addition and gradients."""
        a = Tensor(3.0)
        b = Tensor(2.0)
        c = a + b
        c.backward()
        
        assert c.data.item() == 5.0
        assert a.grad.item() == 1.0
        assert b.grad.item() == 1.0
        
    def test_addition_vectors(self):
        """Test vector addition and gradients."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a + b
        c.backward()
        
        np.testing.assert_array_equal(c.data, np.array([5.0, 7.0, 9.0]))
        np.testing.assert_array_equal(c.grad, np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(a.grad, np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(b.grad, np.array([1.0, 1.0, 1.0]))
        
    def test_subtraction(self):
        """Test subtraction and gradients."""
        a = Tensor([5.0, 3.0])
        b = Tensor([2.0, 1.0])
        c = a - b
        c.backward()
        
        np.testing.assert_array_equal(c.data, np.array([3.0, 2.0]))
        np.testing.assert_array_equal(c.grad, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(a.grad, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(b.grad, np.array([-1.0, -1.0]))
        
    def test_multiplication_elementwise(self):
        """Test element-wise multiplication and gradients."""
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        c.backward()
        
        np.testing.assert_array_equal(c.data, np.array([8.0, 15.0]))
        np.testing.assert_array_equal(a.grad, np.array([4.0, 5.0]))
        np.testing.assert_array_equal(b.grad, np.array([2.0, 3.0]))
        
    def test_power_operation(self):
        """Test power operation and gradients."""
        a = Tensor([2.0, 3.0])
        c = a ** 2
        c.backward()
        
        np.testing.assert_array_equal(c.data, np.array([4.0, 9.0]))
        np.testing.assert_array_equal(a.grad, np.array([4.0, 6.0]))

class TestMatrixOperations:
    """Test matrix operations and their gradients."""
    
    def test_matrix_multiplication_2d(self):
        """Test 2D matrix multiplication and gradients."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a.mat_mul(b)
        loss = c.sum()
        loss.backward()
        
        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Gradient check
        def f(x): return x.mat_mul(b).sum()
        num_grad = numerical_gradient(f, a)
        assert_gradients_close(a.grad, num_grad)
        
    def test_matrix_vector_multiplication(self):
        """Test matrix-vector multiplication and gradients."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([5.0, 6.0])
        c = a.mat_mul(b)
        loss = c.sum()
        loss.backward()
        
        # Expected result: [17, 39]
        expected = np.array([17.0, 39.0])
        np.testing.assert_array_equal(c.data, expected)
        
        # Gradient check
        def f(x): return x.mat_mul(b).sum()
        num_grad = numerical_gradient(f, a)
        assert_gradients_close(a.grad, num_grad)
        
    def test_vector_vector_multiplication(self):
        """Test vector dot product and gradients."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a.mat_mul(b)
        c.backward()
        
        # Expected result: 32 (1*4 + 2*5 + 3*6)
        assert c.data.item() == 32.0
        np.testing.assert_array_equal(a.grad, b.data)
        np.testing.assert_array_equal(b.grad, a.data)
        
    def test_transpose(self):
        """Test transpose operation and gradients."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a.transpose()
        loss = b.sum()
        loss.backward()
        
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_equal(b.data, expected)
        
        # Gradient should flow back correctly
        assert_gradients_close(a.grad, np.ones_like(a.data))


class TestActivationFunctions:
    """Test activation functions and their gradients."""
    
    def test_relu(self):
        """Test ReLU activation and gradients."""
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        b = a.relu()
        loss = b.sum()
        loss.backward()
        
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(b.data, expected)
        
        # Gradient check
        def f(x): return x.relu().sum()
        num_grad = numerical_gradient(f, a)
        assert_gradients_close(a.grad, num_grad)
        
    def test_sigmoid(self):
        """Test sigmoid activation and gradients."""
        a = Tensor([-1.0, 0.0, 1.0])
        b = a.sigmoid()
        loss = b.sum()
        loss.backward()
        
        # Check that sigmoid values are in (0, 1)
        assert np.all(b.data > 0) and np.all(b.data < 1)
        
        # Gradient check
        def f(x): return x.sigmoid().sum()
        num_grad = numerical_gradient(f, a)
        assert_gradients_close(a.grad, num_grad, rtol=1e-3)
        
    def test_tanh(self):
        """Test tanh activation and gradients."""
        a = Tensor([-1.0, 0.0, 1.0])
        b = a.tanh()
        loss = b.sum()
        loss.backward()
        
        # Check that tanh values are in (-1, 1)
        assert np.all(b.data > -1) and np.all(b.data < 1)
        
        # Gradient check
        def f(x): return x.tanh().sum()
        num_grad = numerical_gradient(f, a)
        assert_gradients_close(a.grad, num_grad)
        
    def test_softmax(self):
        """Test softmax activation and gradients."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a.softmax(axis=-1)
        loss = b.sum()
        loss.backward()
        
        # Check that softmax sums to 1 along axis
        sums = np.sum(b.data, axis=-1)
        np.testing.assert_allclose(sums, np.ones(2), rtol=1e-6)
        
    def test_exp(self):
        """Test exponential function and gradients."""
        a = Tensor([0.0, 1.0, 2.0])
        b = a.exp()
        loss = b.sum()
        loss.backward()
        
        expected = np.exp(a.data)
        np.testing.assert_allclose(b.data, expected)
        
        # Gradient check: d/dx(exp(x)) = exp(x)
        def f(x): return x.exp().sum()
        num_grad = numerical_gradient(f, a)
        assert_gradients_close(a.grad, num_grad)
        
    def test_log(self):
        """Test natural logarithm and gradients."""
        a = Tensor([1.0, 2.0, 3.0])
        b = a.log()
        loss = b.sum()
        loss.backward()
        
        expected = np.log(a.data)
        np.testing.assert_allclose(b.data, expected)
        
        # Gradient check: d/dx(log(x)) = 1/x
        def f(x): return x.log().sum()
        num_grad = numerical_gradient(f, a)
        assert_gradients_close(a.grad, num_grad)


class TestReductionOperations:
    """Test reduction operations and their gradients."""
    
    def test_sum_no_axis(self):
        """Test sum over all elements."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.sum()
        b.backward()
        
        assert b.data.item() == 10.0
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data) * b.grad)
        
    def test_sum_with_axis(self):
        """Test sum along specific axis."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.sum(axis=0)
        b.backward()
        
        expected = np.array([4.0, 6.0])
        np.testing.assert_array_equal(b.data, expected)
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data) * b.grad)

    def test_sum_with_axis(self):
        """Test sum along specific axis."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.sum(axis=1)
        b.backward()
        
        expected = np.array([3.0, 7.0])
        np.testing.assert_array_equal(b.data, expected)
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data) * b.grad)
        
    def test_mean(self):
        """Test mean operation and gradients."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.mean()
        b.backward()
        
        assert b.data.item() == 2.5
        expected_grad = np.full_like(a.data, 1.0 / a.data.size)
        np.testing.assert_array_equal(a.grad, expected_grad)


# class TestBroadcasting:
#     """Test broadcasting behavior in operations."""
    
#     def test_scalar_vector_addition(self):
#         """Test addition of scalar and vector."""
#         a = Tensor(5.0)
#         b = Tensor([1.0, 2.0, 3.0])
#         c = a + b
#         loss = c.sum()
#         loss.backward()
        
#         expected = np.array([6.0, 7.0, 8.0])
#         np.testing.assert_array_equal(c.data, expected)
        
#         # Gradient check
#         def f_a(x): return (x + b).sum()
#         def f_b(x): return (a + x).sum()
        
#         num_grad_a = numerical_gradient(f_a, a)
#         num_grad_b = numerical_gradient(f_b, b)
        
#         assert_gradients_close(a.grad, num_grad_a)
#         assert_gradients_close(b.grad, num_grad_b)
        
#     def test_scalar_matrix_multiplication(self):
#         """Test multiplication of scalar and matrix."""
#         a = Tensor(2.0)
#         b = Tensor([[1.0, 2.0], [3.0, 4.0]])
#         c = a * b
#         loss = c.sum()
#         loss.backward()
        
#         expected = np.array([[2.0, 4.0], [6.0, 8.0]])
#         np.testing.assert_array_equal(c.data, expected)
        
#         # Gradient check
#         def f_a(x): return (x * b).sum()
#         def f_b(x): return (a * x).sum()
        
#         num_grad_a = numerical_gradient(f_a, a)
#         num_grad_b = numerical_gradient(f_b, b)
        
#         assert_gradients_close(a.grad, num_grad_a)
#         assert_gradients_close(b.grad, num_grad_b)


# class TestComplexComputations:
#     """Test complex computational graphs."""
    
#     def test_mlp_forward_backward(self):
#         """Test a simple multi-layer perceptron computation."""
#         # Input
#         x = Tensor([[1.0, 2.0]])  # (1, 2)
        
#         # First layer: 2 -> 3
#         W1 = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # (2, 3)
#         b1 = Tensor([0.1, 0.2, 0.3])  # (3,)
        
#         # Second layer: 3 -> 1
#         W2 = Tensor([[0.7], [0.8], [0.9]])  # (3, 1)
#         b2 = Tensor([0.5])  # (1,)
        
#         # Forward pass
#         h1 = x.mat_mul(W1) + b1  # (1, 3)
#         h1_act = h1.relu()
#         output = h1_act.mat_mul(W2) + b2  # (1, 1)
        
#         loss = output.sum()
#         loss.backward()
        
#         # Check that all gradients exist and have correct shapes
#         assert x.grad.shape == x.data.shape
#         assert W1.grad.shape == W1.data.shape
#         assert b1.grad.shape == b1.data.shape
#         assert W2.grad.shape == W2.data.shape
#         assert b2.grad.shape == b2.data.shape
        
#         # Gradient check for weights
#         def f_W1(w):
#             # Reset gradients
#             for tensor in [x, w, b1, W2, b2]:
#                 tensor.grad = np.zeros_like(tensor.data)
#             h1 = x.mat_mul(w) + b1
#             h1_act = h1.relu()
#             output = h1_act.mat_mul(W2) + b2
#             return output.sum()
        
#         num_grad_W1 = numerical_gradient(f_W1, W1)
#         assert_gradients_close(W1.grad, num_grad_W1, rtol=1e-3)


# if __name__ == "__main__":
#     # Run a few basic tests manually
#     print("Running basic tensor tests...")
    
#     # Test scalar addition
#     a = Tensor(3.0)
#     b = Tensor(2.0)
#     c = a + b
#     c.backward()
#     print(f"Scalar addition: {a.data} + {b.data} = {c.data}")
#     print(f"Gradients: a.grad = {a.grad}, b.grad = {b.grad}")
    
#     # Test matrix multiplication
#     a = Tensor([[1.0, 2.0], [3.0, 4.0]])
#     b = Tensor([[5.0, 6.0], [7.0, 8.0]])
#     c = a.mat_mul(b)
#     loss = c.sum()
#     loss.backward()
#     print(f"\nMatrix multiplication result:\n{c.data}")
#     print(f"Matrix A gradient:\n{a.grad}")
    
#     # Test ReLU activation
#     a = Tensor([-1.0, 0.0, 1.0, 2.0])
#     b = a.relu()
#     loss = b.sum()
#     loss.backward()
#     print(f"\nReLU: {a.data} -> {b.data}")
#     print(f"ReLU gradients: {a.grad}")
    
#     # Test softmax
#     a = Tensor([[1.0, 2.0, 3.0]])
#     b = a.softmax(axis=-1)
#     loss = b.sum()
#     loss.backward()
#     print(f"\nSoftmax: {a.data} -> {b.data}")
#     print(f"Softmax sum: {np.sum(b.data, axis=-1)}")
    
#     # Test broadcasting
#     a = Tensor(2.0)  # scalar
#     b = Tensor([1.0, 2.0, 3.0])  # vector
#     c = a * b
#     loss = c.sum()
#     loss.backward()
#     print(f"\nBroadcasting: {a.data} * {b.data} = {c.data}")
#     print(f"Broadcast gradients: scalar={a.grad}, vector={b.grad}")
    
#     print("\n" + "="*50)
#     print("All basic tests completed successfully!")
#     print("Run 'python -m pytest test/tensor_ops.py -v' for full test suite")