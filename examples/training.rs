use tiny_ml::prelude::*;

fn main() {
    // create a neural network
    let mut net = NeuralNetwork::new().add_layer(1, ActivationFunction::Linear);

    // create training data
    let mut inputs = vec![];
    let mut output = vec![];
    for i in -50..50 {
        inputs.push([i as f32]);
        output.push(Expectation::Value {
            expected: [i as f32 * 3.0],
        });
    }
    let data = DataSet::new().set_inputs(inputs).set_output(output);

    let trainer = BasicTrainer::new(data);

    // train the model
    for _ in 0..10 {
        trainer.train(&mut net, 10);
        // lower is better
        println!("{}", trainer.get_total_error(&net))
    }

    // show that this actually works!
    println!("########");
    for i in -5..5 {
        println!("{}", &net.run(&[i as f32 + 0.5])[0]);
    }
}
