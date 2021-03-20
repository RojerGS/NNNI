# NNNI

### Neural Networks with No Imports (in Python)

As a proof of concept, this repository contains a *single*
Python script with a full implementation of framework
for artificial neural networks while making use of **no**
imports whatsoever.
This means that the implementation is done purely in
vanilla Python, without even resorting to the standard
library.

Here are some of the things that had to be implemented
by hand because no `import` were allowed:

 - pseudo-random number generation;
 - matrices and matrix algebra:
   - matrix addition and subtraction
   - matrix multiplication and division with scalars
   - matrix comparison with scalars
   - matrix transpose
 - classes for abstract activation and loss functions
 - Leaky ReLU activation function
 - MSE loss function
 - layer and neural network classes
 - forward pass and backpropagation

 > As of now, the code may have a bug that is preventing the
 neural network from learning on the MNIST dataset.
 To be sorted out soon, live on [mathspp.com/twitch](https://mathspp.com/twitch).
