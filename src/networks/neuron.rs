use rand::random;

use super::*;

#[derive(Default, Debug, Clone)]
pub struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    activiation: ActivationFunction,
}

impl Neuron {
    pub fn new(n_inputs: usize, bias: f32, func: ActivationFunction) -> Self {
        Self {
            weights: vec![1.0; n_inputs],
            bias,
            activiation: func,
        }
    }

    pub fn random(n_inputs: usize , func: ActivationFunction) -> Self {
        let mut weights = vec![];
        for _ in 0..n_inputs {
            weights.push(random())
        }
        Self { weights, bias: random(),  activiation: func }
    }

    pub fn set_weights(&mut self, weights: Vec<f32>) {
        self.weights = weights
    }

    pub fn change_weight(&mut self, index: usize, change: f32) {
        self.weights[index] += change;
    }

    pub fn get_weights_len(&self) -> usize {
        self.weights.len()
    }

    pub fn set_bias(&mut self, bias: f32) {
        self.bias = bias
    }

    pub fn change_bias(&mut self, change: f32) {
        self.bias += change;
    }

    pub fn compute(&self, x: &[f32]) -> f32 {
        let res: f32 = x.iter().zip(&self.weights).map(|(a, b)| a * b).sum::<f32>() + self.bias;
        match self.activiation {
            ActivationFunction::Linear => res,
            ActivationFunction::ReLU => res.max(0.0),
        }
    }
}
