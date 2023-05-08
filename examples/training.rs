use tiny_ml::prelude::*;

fn main() {
    let mut net = NeuralNetwork::new(1).random_layer(1, ActivationFunction::Linear).random_layer(1, ActivationFunction::Linear);
    let mut inputs = vec![];
    let mut output = vec![];
    for i in -50..50 {
        inputs.push(vec![i as f32]);
        output.push(Expectation::Value {
            expected: vec![i as f32 * 3.0],
        });
    }
    let data = DataSet::new().set_inputs(inputs).set_output(output);

    let trainer = BasicTrainer::new(data);
    trainer.train(&mut net, 100000);

    for i in -5..5 {
        println!("{}", &net.run(&vec![i as f32 + 0.5]).unwrap()[0]);
    }
}
