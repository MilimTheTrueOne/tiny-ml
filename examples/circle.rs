use tiny_ml::prelude::*;

fn main() {
    // this network is completely overkill, but it does the job
    let mut net: NeuralNetwork<2, 1> = NeuralNetwork::new()
        .add_layer(3, ActivationFunction::ReLU)
        .add_layer(3, ActivationFunction::ReLU)
        // this layer reduced everything to one input!
        .add_layer(1, ActivationFunction::Linear);

    let mut inputs = vec![];
    let mut output = vec![];
    for x in 0..=100 {
        for y in 0..=100 {
            inputs.push([x as f32, y as f32]);
            // we want this to be a classifier, so we ask it for a result greater zero or smaller zero
            output.push(if (x as f32).abs() + (y as f32).abs() < 30.0 {
                [1.0]
            } else {
                [-1.0]
            })
        }
    }

    let data = DataSet {
        inputs,
        outputs: output,
    };

    let trainer = BasicTrainer::new(data);
    for _ in 0..50 {
        trainer.train(&mut net, 10);
        println!("{}", trainer.get_total_error(&net))
    }
}
