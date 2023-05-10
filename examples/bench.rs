use tiny_ml::prelude::*;

fn main() {
    // this network is completley overkill, but it does the job
    let net = NeuralNetwork::new(1)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        // this layer makles sure we get one input
        .add_layer(1, ActivationFunction::Linear);

    for i in 0..10_000_000 {
        net.run(&vec![i as f32]);
    }
}
