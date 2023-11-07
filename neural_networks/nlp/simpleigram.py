#This is a simple bigram network neural network using Pytorch and its back progagation
#It uses a single layer and with no non-linearity and basic L2 regularization
#This network is used to generate names.
#

import torch.nn.functional as F

####Load the data
#Open the data file
names = open('names.txt', 'r').read().split()


####Network


#Creating the simplest Neural Network
#First create a training set of bigrams
xs=[]
ys=[]

for w in names:
  ch = '.' + w + '.'
  x=[]
  y=[]
  for ch1, ch2 in zip(ch, ch[1:]):
    xs.append(stoi[ch1])
    ys.append(stoi[ch2])


print(f'Number of bigrams: {len(xs)} \n{xs=} \n{ys=}')

#Pass it through a single Neuron 27 inputs and 1 output
#W = torch.randn(27,1)
#Input 5,27 -> Neuron 27,1 gives 5,1. PyTorch is processing 5 inputs in parallel
#(x_enc @ W)

#Pass through 27 neurons. Each neuron accepts 27 inputs representing the first word in the bigram and the final output is 27
#This is a fully linear layer with no non-linearity
W = torch.randn((27,27), requires_grad=True)


debug=False

for step in range(50):
  ###Forward Pass
  x_enc = F.one_hot(torch.tensor(xs), num_classes=27).float()
  logits = (x_enc @ W) #Log Counts
  #Create a softmax to get the probabilities
  counts = torch.exp(logits) # Counts that are similar to N
  probs = counts / counts.sum(1, keepdim=True)

  ####Calculate the loss
  nlls = torch.zeros(5)

  #Print more details
  if debug:
    for i in range(len(xs)):
      bigram = (itos[xs[i]],itos[ys[i]])
      nlls[i] =  - torch.log(probs[i, ys[i]]) #Negative Log Likelihood
      print(f'Bigram: {bigram}, Input: {xs[i]}, Label: {ys[i]}, Prob. next character after {itos[xs[i]]} is {itos[ys[i]]} = {probs[i, ys[i]]}, Negative Loss Likelihood: {- torch.log(probs[i, ys[i]])} ')

    print(f'Model NLL: {nlls.mean().item()}')

  #More efficient way of doing it is
  loss = - probs[torch.arange(len(xs)), ys].log().mean() + 0.01*(W**2).mean() #Regularization term forces weights to go to zero and make the distribution uniform or smooth
  print(f'Step no: {step} loss: {loss.item()}')

  ####Run the Backward Pass
  #Set the gradient to 0
  W.grad = None #Set the gradient to 0
  loss.backward()

  ####Update the weights
  W.data += - 10*W.grad


##Generate the word predictions

g = torch.Generator()

for i in range(5):
  ix=0
  out=[]
  while True:
    p = P[ix]
    ###Forward Pass
    x_enc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = (x_enc @ W) #Log Counts
    #Create a softmax to get the probabilities
    counts = torch.exp(logits) # Counts that are similar to N
    probs = counts / counts.sum(1, keepdim=True)

    ix=torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
