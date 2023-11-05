#Adapted from micrograd by Andrej Karpathy
#https://github.com/karpathy/micrograd
import random
from autograd import Value

class Neuron:
  def __init__(self, nin, activation='tanh'):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value((random.uniform(-1,1)))
    self.activation = activation

  def __call__(self, x):
    act = sum((xi*wi for xi, wi in zip(self.w, x)), self.b)

    if self.activation == 'relu':
      out = act.relu()
    else:
      out = act.tanh()

    return out

  def parameters(self):
    return self.w + [self.b]
  
  def __repr__(self):
    return f"{self.activation} Neuron({len(self.w)})"

class Layer:
  def __init__(self, nin, noutput, activation='tanh'):
    self.neurons = [Neuron(nin, activation) for _ in range(noutput)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
  
  def __repr__(self):
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:
  def __init__(self, nin, noutputs, activations=[]):
    sz = [nin] + noutputs
    if len(activations) > 0:
      self.layers = [Layer(sz[i], sz[i+1], activations[i]) for i in range(len(noutputs))]
    else:
      self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(noutputs))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

  def __repr__(self):
    return f"Model type: MLP [{', '.join(str(layer) for layer in self.layers)}]"
