# Tiny ml

Basic neural networks for rust!

## What is this for?
It is meant for small machine learning applications, such as evolution simulators. It can technically be used for other things, but is not intended to be used that way.
## Things to know:
- This is _very_ early in development, everything is subject to change.
- Use --release or opt-level=3 or this will be very slow.
- The traininf algorithm currently is very simple and naive, and probably will take imense amounts of time on more compley datasets.
## Speed
Using the NeuralNetwork::par_run function, on a network with 301 neurons, 1 input, 1 output the crate can handle some ~800000 runs / second.
For smaller models speed is signicicantly better. (31 neurons, 1 input, 1 output, ~8000000 runs / second)     
Considering this crate is called tiny ml, i would say that is quite fast. However i will continue to improve the performance as time goes on.