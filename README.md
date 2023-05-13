
# tiny ml

A simple, fast rust crate for simple basic neural networks. 

### What is this for?
- Learning about ML 
- Evolution simulations
- Whaterver else you want to use this for

#### what is this **not**?
- A large scale ML libary like Tensorflow or PyTorch. This is simple and basic, or just 'tiny'.

## How to use this?
As an example, here is how to make a model that can tell if a point is in a circle or not!
```rust
use tiny_ml::prelude::*;

// how many input datapoints the model has
const NET_INPUTS: usize = 2;
// how many datampoints the model outputs
const NET_OUTPUTS: usize = 1;
// radius of the circle
const RADIUS: f32 = 30.0;


fn main() {
    // create a network   
    let mut net: NeuralNetwork<NET_INPUTS, NET_OUTPUTS> = NeuralNetwork::new()
        .add_layer(3, ActivationFunction::ReLU)
        .add_layer(3, ActivationFunction::ReLU)
        .add_layer(1, ActivationFunction::Linear);
    // this network has no weights yet but we can fix that by training it
    
    // for training we first need a  dataset
    let mut inputs = vec![];
    let mut outputs = vec![];

    // well just generate some samples
    for x in 0..100 {
        for y in 0..100 {
            inputs.push([x as f32, y as f32]);
            // we want this to be a classifier, so we will give -1 for in the circle 
            // and +1 for in the circle 
            outputs.push(
                if (x as f32).abs() + (y as f32).abs() < RADIUS{
                    Expectation { expected: [1.0]}
                } else {
                    Expectation { expected: [-1.0]}
                }
            )
        }
    }


    let data = DataSet {
        inputs,
        outputs,
    };

    // get ourselves a trainer
    let trainer = BasicTrainer::new(data);
    // let it train 10 times, 50 iterations each 
    for _ in 0..10 {
        trainer.train(&mut net, 50);
        // print the total error, lower is better
        println!("{}", trainer.get_total_error(&net))
    }
}
```

## Speed?
Here some benchmarks on an AMD Ryzen 5 2600X (12) @ 3.6 GHz with the 'bench' example.
Build with `--release`-flag enabled.
Benchmark is 10 Million runs on this network, and then sum the results: 
```rust
    let mut net: NeuralNetwork<1, 1> = NeuralNetwork::new()
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(1, ActivationFunction::Linear);
```
| method | time     | Description                |
| :-------- | :------- | :------------------------- |
| `run` | 1.045s | Single threaded, but buffers some vecs |
| `unbufferd_run` | 1.251s | Can be ran in multiple threads at the same time, however has to allocate more often |
| `par_run` | 240ms | Takes a multiple inputs at once. Parallelizes computation with `rayon`, uses `unbufferd_run` under the hood |