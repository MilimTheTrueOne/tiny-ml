use rand::random;

use super::*;

/// Struct representing one Neuron
#[cfg(feature = "serialization")]
#[derive(Serialize, Deserialize)]
#[derive(Default, Debug, Clone)]
pub struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    activation: ActivationFunction,
}

impl Neuron {
    pub fn new(n_inputs: usize, bias: f32, func: ActivationFunction) -> Self {
        Self {
            weights: vec![1.0; n_inputs],
            bias,
            activation: func,
        }
    }

    pub fn random(n_inputs: usize, func: ActivationFunction) -> Self {
        let mut weights = vec![];
        for _ in 0..n_inputs {
            weights.push(random())
        }
        Self {
            weights,
            bias: random(),
            activation: func,
        }
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

    #[inline]
    pub fn compute(&self, x: &[f32]) -> f32 {
        let mut res = self.bias;
        let mut i = 0;
        while i < self.weights.len() {
            res += self.weights[i] * x[i];
            i += 1;
        }
        match self.activation {
            ActivationFunction::Linear => res,
            ActivationFunction::ReLU => res.max(0.0),
        }
    }
}
