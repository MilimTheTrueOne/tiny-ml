use tiny_ml::prelude::*;

fn main() {
    let mut net = NeuralNetwork::new(2)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(5, ActivationFunction::ReLU)
        .add_layer(1, ActivationFunction::Linear);

    let mut inputs = vec![];
    let mut output = vec![];
    for x in 0..=100 {
        for y in 0..=100 {
            inputs.push(vec![x as f32, y as f32]);
            output.push(if (x as f32).abs() + (y as f32).abs() < 30.0 {
                Expectation::SmallerZero
            } else {
                Expectation::GreaterZero
            })
        }
    }
    
    let data = DataSet {
        inputs: inputs,
        outputs: output,
    };
    let trainer = BasicTrainer::new(data);
    for _ in 0..5 {
        trainer.train(&mut net, 500);
        println!("{}", trainer.get_distance(&net))
    }
}