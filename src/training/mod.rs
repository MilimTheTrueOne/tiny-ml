use std::usize;

use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::networks::NeuralNetwork;

/// A simple struct for Training Neural Networks
pub struct BasicTrainer<const N: usize, const O: usize> {
    training_data: DataSet<N, O>,
}

impl<const N: usize, const O: usize> BasicTrainer<N, O> {
    pub fn new(data: DataSet<N, O>) -> Self {
        Self {
            training_data: data,
        }
    }
    pub fn train(&self, net: &mut NeuralNetwork<N, O>, iterations: usize) {
        let training_data = &self.training_data;
        let mut pre_dist = self.compute_distance(net, training_data);
        for _ in 0..=iterations {
            net.random_edit();
            let after_dist = self.compute_distance(net, training_data);
            if pre_dist < after_dist {
                net.reverse_edit();
            } else {
                pre_dist = after_dist;
            }
        }
    }

    pub fn get_distance(&self, net: &NeuralNetwork<N, O>) -> f32 {
        self.compute_distance(net, &self.training_data)
    }

    fn compute_distance(&self, net: &NeuralNetwork<N, O>, data: &DataSet<N, O>) -> f32 {
        data.inputs
            .par_iter()
            .zip(&data.outputs)
            .map(|(input, output)| {
                let result = net.unbufferd_run(input);

                match output {
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
            .sum()
    }
}
/// A set of inputs and the expected Outputs
#[derive(Default)]
pub struct DataSet<const N: usize, const O: usize> {
    pub inputs: Vec<[f32; N]>,
    pub outputs: Vec<Expectation<O>>,
}

pub enum Expectation<const O: usize> {
    GreaterZero,
    SmallerZero,
    Value { expected: [f32; O] },
}
impl<const N: usize, const O: usize> DataSet<N, O> {
    pub fn new() -> Self {
        Self {
            inputs: vec![],
            outputs: vec![],
        }
    }

    pub fn set_inputs(mut self, inputs: Vec<[f32; N]>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn set_output(mut self, output: Vec<Expectation<O>>) -> Self {
        self.outputs = output;
        self
    }
}
