#
#Based on micrograd by Andrej Karpath
#https://github.com/karpathy/micrograd
import math
#Draw the computation directed graph
from graphviz import Digraph

class Value:
  def __init__(self, data, children=(), _op='', name=''):
    self.data = data
    self._op = _op
    self._prev = children
    self.name = name
    self.grad=0.0
    self._backward = lambda: None

  def __repr__(self):
    return f'Value ({self.name=}, {self.data=}, {self._op=})'

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad

    out._backward = _backward
    return out

  def __radd__(self,other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self,other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward

    return out

  def __rmul__(self, other): #this is to accommodate int * Value
    return self * other

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "Float and int supported in power"
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
      self.grad += other*(self.data**(other-1)) * out.grad

    out._backward = _backward

    return out

  def relu(self):
      out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

      def _backward():
          self.grad += (out.data > 0) * out.grad
      out._backward = _backward

      return out

  def tanh(self):
    n = self.data
    _exp = math.exp(2*n)
    t = (_exp - 1)/(_exp + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
      self.grad += (1 - t**2)* out.grad

    out._backward = _backward

    return out

  def __truediv__ (self, other):
    return self * other**-1

  def __neg__ (self):
    return self*-1

  def __sub__(self, other):
    return self + (-other)

  def exp(self):
    x=self.data
    out = Value(math.exp(x), (self,), 'exp')

    def _backward():
      self.grad += out.data * out.grad

    out._backward = _backward

    return out

  def backward(self):
    #We want to do a topological sort before we can process the backward pass automatically
    #This is also used for foward pass as well
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    #Build the topological graph
    build_topo(self)
    topo

    self.grad = 1.0
    #Run the backwards pass in the topological graph
    for node in reversed(topo):
      node._backward()

  def _trace(self):
    edges, nodes = set(), set()
    def build(v):
      if v not in nodes:
        nodes.add(v)
        for child in v._prev:
          edges.add((child, v))
          build(child)
    build(self)
    return nodes, edges

  def draw_dot(self):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) #Left to right
    nodes, edges = self._trace(self)
    for n in nodes:
      uid = str(id(n))
      dot.node(name = uid, label= "{%s | data %.4f | grad %.4f}" % (n.name, n.data, n.grad, ), shape="record")
      if n._op:
        #Define the op node
        dot.node(name = uid + n._op, label=n._op)
        #Connect the edge to the previous node
        dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
      dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot