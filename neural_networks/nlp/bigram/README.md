# Bigram Model
## Concept
Take a list of names and break up the name into sets of 2 characters each known as bigrams. For example: Pankaj is broken into ".P", "Pa", "an", "nk", "ka", "aj" and "j.". The "." represents start and end token for a word or name. 

For each bigram, the first character is fed as input to a single layer (27x27) linear layer and the second character is used as label. Each character is one hot encoded. The logits of the layer are then transformed into probabilities using softmax.

The probabilities are then used to generate names.

## Performance
The model is very basic and does not perform very well as a generative model. But serves as a good foundation for NLP and its basics.