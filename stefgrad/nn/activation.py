from stefgrad.tensor import Tensor

def softmax(t: Tensor):
    return t.softmax()

def sigmoid(t: Tensor):
    return t.sigmoid()

def relu(t: Tensor):
    return t.relu()

def tanh(t: Tensor):
    return t.tanh()