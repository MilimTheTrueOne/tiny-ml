use tiny_ml::prelude::*;

fn main() {
    // let's check how fast this thing gets
    let net: NeuralNetwork<1, 1> = NeuralNetwork::new()
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(1, ActivationFunction::Linear);

    let mut sum = 0.0;
    for i in 0..10_000_000 {
        sum += net.run(&[i as f32])[0];
    }

    println!("{sum}");
}
