Implementation of label neural network
===========================================

This is (so far partial) a Julia implementation of the neural network that was originally written for MATLAB/Octave by Andrew Ng for Coursera Machine Learning Class. My implementation is hopefully more versatile for more than one layer.

What works
-------------------------------------------
* There is a `nnCostFunction` (see the sourcefile for documentation) which seems to be working, output is (cost, array of gradients)

TO DO
-------------------------------------------
* I was unable to use the optim.jl to make it iterate
* Add the checking gradient

Usage
-------------------------------------------
* Just download and include `load("neuralnetworks.jl")` in your source
