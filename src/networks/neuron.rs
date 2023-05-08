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
            bias: bias,
            activiation: func,
        }
    }

    pub fn set_weights(&mut self, weights: Vec<f32>) {
        self.weights = weights
    }

    pub fn set_bais(&mut self, bias: f32) {
        self.bias = bias
    }

    pub fn compute(&self, x: &Vec<f32>) -> f32 {
        let res: f32 = x.iter().zip(&self.weights).map(|(a, b)| a * b).sum::<f32>() + self.bias;
        match self.activiation {
            ActivationFunction::Linear => res,
            ActivationFunction::ReLU => res.max(0.0),
        }
    }
}
