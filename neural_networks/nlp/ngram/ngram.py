#Libraries
#Based on the paper by https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')

#Create linear class

class Linear:
  def __init__(self, fanin, fanout, bias=True, debug=False):
    #Fine Tuning 1:
    #At initialization we want the probabilities to uniformly distributed. In our case it would be
    #1/(total_chars) with a loss of torch.log(torch.tensor(1/(total_chars))) = 3.2958
    #But in our case we get an initial loss of 27.xxx. It means that the weights initialization is giving a lot more
    #probability to some characters while ignoring others. Hence we want to adjust the initialization weights
    #The way to ensure uniform output is to divide by the sqrt of the fanin (input to the layer)
    #Also check the kaiminginit documentation in pytorch to see if we need to add again depending on the type of
    #activation.
    self.weights = torch.randn((fanin,fanout)) /  fanin**0.5
    self.bias = torch.zeros(fanout) if bias else None

    self.debug=debug
    self.training=True

  def __call__(self, X):
    self.out = X @ self.weights
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weights] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
  def __init__(self, dim, eps=1e-05, momentum=0.1, debug=False):
    self.eps = eps
    self.momentum = momentum
    #Parameters trained with back propagation
    self.gamma = torch.ones((1, dim))
    self.beta = torch.zeros((1, dim))

    #Running parameters used during training
    self.bnmean = torch.zeros(dim)
    self.bnvar = torch.ones(dim)
    self.training = True

    #Others
    self.debug=debug

  def __call__(self, X):
    if self.training:
      xmean = X.mean(0,keepdim=True)
      xvar = X.var(0, keepdim=True, unbiased=True)
    else:
      xmean = self.bnmean
      xvar = self.bnvar

    self.out = self.gamma * (X - xmean) / (torch.sqrt(xvar+ self.eps)) + self.beta

    if self.training:
      with torch.no_grad():
        self.bnmean = (1 - self.momentum)*self.bnmean + self.momentum*xmean
        self.bnvar = (1 - self.momentum)*self.bnvar + self.momentum*xvar

    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __init__(self):
    self.training=True

  def __call__(self, X):
    self.out = torch.tanh(X)
    return self.out

  def parameters(self):
    return []

class Tokenizer:
  def __init__(self, block_size, data, debug=False):
    self.data=data
    self.block_size=block_size
    self.chars = list(sorted(set(''.join(data))))
    self.stoi = {chr:i+1 for i, chr in enumerate(self.chars)}
    self.stoi['.'] = 0
    self.itos = {i:chr for chr, i in self.stoi.items()}
    self.debug = debug
    self.vocab_size = len(self.stoi)

  def splitData(self, split=[0.8,0.9,1]):
    X_tr, Y_tr = self.build_data_set(self.data[:int(0.8*len(self.data))])
    X_dev, Y_dev = self.build_data_set(self.data[int(0.8*len(self.data)):int(0.9*len(self.data))])
    X_test, Y_test = self.build_data_set(self.data[int(0.9*len(self.data)):])

    return X_tr, Y_tr, X_dev, Y_dev, X_test, Y_test

  def build_data_set(self,data):
    #Data Set
    X=[]
    Y=[]

    for w in data:
      context = [0] * self.block_size
      for ch in w + '.':
        ix = self.stoi[ch]
        X.append(context)
        Y.append(ix)
        if self.debug: print(''.join([self.itos[i] for i in context]),'-->', self.itos[ix])
        context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X,Y


class NgramModel:
  def __init__(self, layers, embed_vector_size, tokenizer, lr_start=0.05, lr_end=0.0001, debug=False):
    self.layers = layers
    self.embed_vector_size = embed_vector_size
    self.training=True
    self.tokenizer=tokenizer
    self.vocab_size = tokenizer.vocab_size
    self.C = torch.randn((self.vocab_size,embed_vector_size))
    self.flat_embedding_size = tokenizer.block_size*embed_vector_size
    self.debug=debug
    self.lr_start=lr_start
    self.lr_end=lr_end
    self.stats={"ud": [],
                "loss": []}

    with torch.no_grad():
      #Make the last layer less confident
      #self.layers[-1].weights *= 0.1 #Uncomment if not using BN
      #self.layers[-1].gamma *= 0.1 #Comment if not using BN
      if self.layers[-1].__class__.__name__ == 'BatchNorm1d':
        self.layers[-1].gamma *= 0.1 #Make gamme less confident if Batch norm
      else:
        self.layers[-1].weights *= 0.1 #Make weights less confident if not Batch norm


      #For all other layers apply 5/3 gain
      #Since they will be using tanh (this model only supports tanh)
      for layer in self.layers[:-1]:
        if isinstance(layer, Linear):
          layer.weights *= 5/3

    #Set all the parameters to have gradient
    self.parameters = [self.C] + [p for layer in self.layers for p in layer.parameters()]
    for p in self.parameters:
      p.requires_grad = True

  def predict(self,X):
    embedding = self.C[X]
    #refactor x in terms of the flattened embedding size
    x = embedding.view(-1, self.flat_embedding_size)

    #No complete the forward pass
    for layer in self.layers:
      layer.training=False
      x=layer(x)
    return x

  def forward(self,X):
    embedding = self.C[X]
    #refactor x in terms of the flattened embedding size
    x = embedding.view((-1, self.flat_embedding_size))

    #No complete the forward pass
    for layer in self.layers:
      layer.training=True
      x=layer(x)
    return x

  def train(self, X, Y, batch_size=32, epochs=10):

    for i in range(epochs):
      ix = torch.randint(0, len(X), (batch_size,))

      #Run the forward pass
      logits = self.forward(X[ix])

      #Calculate the loss
      #More efficient way to calculate loss is to use the torch cross entropy loss
      #Torch also handles the nans elegantly
      loss = F.cross_entropy(logits, Y[ix])

      #Backward pass
      if self.debug:
        #Retain the gradients for debugging
        for layer in self.layers:
          layer.out.retain_grad()
      for p in self.parameters:
        p.grad = None
      loss.backward()


      #Update gradients
      lr=self.lr_start if i < 100000 else self.lr_end
      for p in self.parameters:
        p.data += - lr * p.grad

      #Collect statistics
      if i%1000 == 0: print(f'Steps= {i} Loss= {loss.item()}')
      with torch.no_grad():
        self.stats['loss'].append(loss.item())
        self.stats['ud'].append([(lr*p.grad.std()/p.data.std()).mean().log10().item() for p in self.parameters])


    print(f'Steps= {i} Loss= {loss.item()}')

  def visualizeLayers(self, type='Tanh', gradient=False):
    #We only retain the gradients in debug mode
    if gradient and not self.debug:
      print(f'Error Visualizing: Gradient can be visualized in debug mode only.')
      return

    plt.figure(figsize=(20,4)) #Width and height of the plot
    legends = []
    for i,layer in enumerate(self.layers[:-1]):

      if (type == layer.__class__.__name__):
        t = layer.out if not gradient else layer.out.grad
        print('Layer %d %s mean: %2f std: %2f saturation: %.2f%%' % (i, type, t.mean(), t.std(), (t.abs()> 0.97).float().mean()*100))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'Layer {i} {layer.__class__.__name__}')

    plt.legend(legends)
    plt.title(f'{type} {"Gradient" if gradient else "Output"}')

  def visualizeWeights(self):
    #We only retain the gradients in debug mode
    if not self.debug:
      print(f'Error Visualizing Weight Gradients.')
      return

    plt.figure(figsize=(20,4)) #Width and height of the plot
    legends = []
    for i,p in enumerate(self.parameters):
        t = p.grad
        if p.ndim == 2:
          print('Layer %10s | mean: %2f | std: %2f | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
          hy, hx = torch.histogram(t, density=True)
          plt.plot(hx[:-1].detach(), hy.detach())
          legends.append(f'Layer {i} {tuple(p.shape)}')

    plt.legend(legends)
    plt.title(f'{"Weights Gradient"}')

  def visualizeGradientUpdates(self):
    #We only retain the gradients in debug mode
    plt.figure(figsize=(20,4)) #Width and height of the plot
    legends = []
    for i,p in enumerate(self.parameters):
        if p.ndim == 2:
          plt.plot([self.stats['ud'][j][i] for j in range(len(self.stats['ud']))])
          legends.append(f'Param {i}')
    plt.plot((0, len(self.stats['ud'])), [-3,-3], 'k')
    plt.legend(legends)
    plt.title(f'{"Gradient update to Data Ratio"}')



#Read the data
#Import the data
names = open('../names.txt','r').read().splitlines()

embedding_vector_size=5

#Create the tokenizer
tokenizer = Tokenizer(block_size=3,data=names)
hidden_layer_neurons=300

#Define model arctitecture - exclised the embedding layer
layers=[Linear(tokenizer.block_size*embedding_vector_size,hidden_layer_neurons),BatchNorm1d(hidden_layer_neurons),  Tanh(),
        Linear(hidden_layer_neurons,hidden_layer_neurons), BatchNorm1d(hidden_layer_neurons), Tanh(),
        Linear(hidden_layer_neurons,hidden_layer_neurons), BatchNorm1d(hidden_layer_neurons), Tanh(),
        Linear(hidden_layer_neurons,hidden_layer_neurons), BatchNorm1d(hidden_layer_neurons), Tanh(),
        Linear(hidden_layer_neurons,tokenizer.vocab_size, BatchNorm1d(hidden_layer_neurons,tokenizer.vocab_size))
]

#Initialize the model
model = NgramModel(layers, embedding_vector_size,tokenizer, debug=True)
X_tr, Y_tr, X_dev, Y_dev, X_test, Y_test = tokenizer.splitData()

model.train(X_tr, Y_tr, epochs=200000)
model.visualizeLayers('Tanh')
model.visualizeLayers('Tanh', gradient=True)
model.visualizeLayers('Linear')
model.visualizeLayers('Linear', gradient=True)
model.visualizeWeights()
model.visualizeGradientUpdates()

#Sample 20 from the trained network
for _ in range(2):
    out = []
    context = [0] * tokenizer.block_size # initialize with all "..."
    while True:
      #emb = C[torch.tensor([context])] # (1,block_size,d)
      logits = model.predict(torch.tensor([context]))
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break

    print(''.join(tokenizer.itos[i] for i in out))