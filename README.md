# Tiny ml

Basic neural networks for rust!

## What is this for?
It is meant for small machine learning applications, such as evolution simulators. It can technically be used for other things, but is not intended to be used that way.

## Speed
Using the NeuralNetwork::bulk_run function, on a network with 301 neurons, 1 input, 1 output the crate can handle some ~1000000 runs / second.
For smaller models speed is signicicantly better. (31 neurons, 1 input, 1 output, ~10000000 runs / second)     
Considering this crate is called tiny ml, i would say that is quite fast. However i will continue to improve the performance as time goes on.