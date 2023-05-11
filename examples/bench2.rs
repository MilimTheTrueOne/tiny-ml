use tiny_ml::prelude::*;

fn main() {
    // let's check how fast this thing gets
    let net = NeuralNetwork::new(1)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(1, ActivationFunction::Linear);

    let mut inputs = vec![];
    for i in 0..10_000_000 {
        inputs.push(vec![i as f32]);
    }

    net.par_run(&inputs);
}
