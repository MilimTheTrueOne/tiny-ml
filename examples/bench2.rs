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

    let mut inputs = vec![];
    for i in 0..10_000_000 {
        inputs.push([i as f32]);
    }

    println!("{}", net.par_run(&inputs).iter().map(|x| x[0]).sum::<f32>());
}
