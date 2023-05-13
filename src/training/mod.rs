use std::usize;

#[cfg(feature = "parallization")]
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::networks::NeuralNetwork;

/// A simple struct for Training Neural Networks
pub struct BasicTrainer<const N: usize, const O: usize> {
    /// the data the model is to be trained on
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
        let mut pre_dist = self.compute_total_error(net, training_data);
        for _ in 0..=iterations {
            net.random_edit();
            let after_dist = self.compute_total_error(net, training_data);
            if pre_dist < after_dist {
                net.reverse_edit();
            } else {
                pre_dist = after_dist;
            }
        }
    }

    pub fn get_total_error(&self, net: &NeuralNetwork<N, O>) -> f32 {
        self.compute_total_error(net, &self.training_data)
    }

    fn compute_total_error(&self, net: &NeuralNetwork<N, O>, data: &DataSet<N, O>) -> f32 {
        #[cfg(feature = "parallization")]
        let it = { data.inputs.par_iter() };

        #[cfg(not(feature = "parallization"))]
        let it = { data.inputs.iter() };

        it.zip(&data.outputs)
            .map(|(input, output)| {
                let result = net.unbufferd_run(input);
                output.expected    
                        .iter()
                        .zip(result)
                        .fold(0.0, |dist, x| dist + (x.0 - x.1).abs())
                
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

pub struct  Expectation<const O: usize> {
    pub expected: [f32; O],
}
