use crate::networks::NeuralNetwork;

pub struct BasicTrainer {
    training_data: DataSet,
}

impl BasicTrainer {
    pub fn new(data: DataSet) -> Self {
        Self {
            training_data: data,
        }
    }
    pub fn train(&self, net: &mut NeuralNetwork, iterations: usize) {
        let training_data = &self.training_data;

        for _ in 0..=iterations {
            let pre_dist = compute_distance(net, training_data);
            net.random_edit();
            let after_dist = compute_distance(net, training_data);
            if pre_dist < after_dist {
                net.reverse_edit();
            }
        }
    }
}

fn compute_distance(net: &NeuralNetwork, data: &DataSet) -> f32 {
    data.inputs
        .iter()
        .zip(&data.outputs)
        .fold(0.0, |acc, (input, output)| {
            let result = net.run(input).unwrap();

            acc + match output {
                Expectation::GreaterZero => {
                    if result[0] > 0.0 {
                        0.0
                    } else {
                        result[0] * -1.0
                    }
                }
                Expectation::SmallerZero => {
                    if result[0] < 0.0 {
                        0.0
                    } else {
                        result[0]
                    }
                }
                Expectation::Value { expected } => expected
                    .iter()
                    .zip(result)
                    .fold(0.0, |dist, x| dist + (x.0 - x.1).abs()),
            }
        })
}

#[derive(Default)]
pub struct DataSet {
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Expectation>,
}

pub enum Expectation {
    GreaterZero,
    SmallerZero,
    Value { expected: Vec<f32> },
}
impl DataSet {
    pub fn new() -> Self {
        Self {
            inputs: vec![],
            outputs: vec![],
        }
    }

    pub fn set_inputs(mut self, inputs: Vec<Vec<f32>>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn set_output(mut self, output: Vec<Expectation>) -> Self {
        self.outputs = output;
        self
    }
}