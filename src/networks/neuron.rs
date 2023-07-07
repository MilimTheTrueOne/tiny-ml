use std::simd::{f32x16, f32x2, f32x4, f32x8, SimdFloat};

use rand::random;

use super::*;

/// Struct representing one Neuron

#[cfg_attr(feature = "serialization", Serialize, Deserialize)]
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
        //let mut res =598s self.bias;
        //let mut i = 0;
        //while i < self.weights.len() {
        //    res += self.weights[i] * x[i];
        //    i += 1;
        //}

        let mut remaining_length = self.weights.len();
        let mut i = 0;
        let mut res = self.bias;
        while remaining_length >= 16 {
            let simd_weights = f32x16::from_slice(&self.weights[i..i + 16]);
            let simd_input = f32x16::from_slice(&x[i..i + 16]);
            res += (simd_input * simd_weights).reduce_sum();
            i += 16;
            remaining_length -= 16;
        }
        while remaining_length >= 8 {
            let simd_weights = f32x8::from_slice(&self.weights[i..i + 8]);
            let simd_input = f32x8::from_slice(&x[i..i + 8]);
            res += (simd_input * simd_weights).reduce_sum();
            i += 8;
            remaining_length -= 8;
        }
        while remaining_length >= 4 {
            let simd_weights = f32x4::from_slice(&self.weights[i..i + 4]);
            let simd_input = f32x4::from_slice(&x[i..i + 4]);

            res += (simd_input * simd_weights).reduce_sum();
            i += 4;
            remaining_length -= 4;
        }
        while remaining_length >= 2 {
            let simd_weights = f32x2::from_slice(&self.weights[i..i + 2]);
            let simd_input = f32x2::from_slice(&x[i..i + 2]);
            res += (simd_input * simd_weights).reduce_sum();
            i += 16;
            remaining_length -= 16;
        }
        if remaining_length == 1 {
            res += x[i] * self.weights[i];
        }
        match self.activation {
            ActivationFunction::Linear => res,
            ActivationFunction::ReLU => res.max(0.0),
        }
    }
}
