import numpy as np

from stefgrad.tensor import Tensor
from stefgrad.nn import Neuron, Layer, MLP

from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='png', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %s | grad %s }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

xs = [
    Tensor([2.0, 3.0, -1]),
    Tensor([3.0, -1.0, 0.5]),
    Tensor([0.5, 1.0, 1.0]),
    Tensor([1.0, 1.0, -1.0])
]
ys = Tensor([1.0, -1.0, -1.0, 1.0])


model = MLP(3, [4, 4, 1]) # 2-layer neural network

for k in range(1):
    # Reset gradients to zero before each iteration
    for p in model.parameters():
        p.grad = np.zeros_like(p.data)
    
    # Compute predictions for each input
    ypred_outputs = [model(x) for x in xs]

    # Compute loss for each prediction and accumulate  
    total_loss = Tensor(0.0)
    for i, y_out in enumerate(ypred_outputs):
        # Use squared error loss: (target - prediction)^2
        individual_loss = (ys.data[i] - y_out) * (ys.data[i] - y_out)
        total_loss = total_loss + individual_loss

    total_loss.backward()
    # dot = draw_dot(total_loss)
    # dot.render("gout.png")

    # Update parameters using gradient descent
    learning_rate = 0.01
    for p in model.parameters():
        p.data = p.data - learning_rate * p.grad

    if k % 10 == 0:
        print(f"Iteration {k}: loss = {total_loss.data:.6f}")