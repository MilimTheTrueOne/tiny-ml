use self::neuron::Neuron;
use rand::Rng;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

mod neuron;

pub struct NeuralNetwork {
    layers: Vec<Vec<neuron::Neuron>>,
    n_inputs: usize,
    last_edit: Option<Edit>,
}

struct Edit {
    old: Neuron,
    layer: usize,
    row: usize,
}

impl NeuralNetwork {
    pub fn new(n_inputs: usize) -> Self {
        Self {
            layers: Vec::new(),
            n_inputs,
            last_edit: None,
        }
    }

    pub fn add_layer(mut self, n: usize, func: ActivationFunction) -> Self {
        let n_inputs = self.get_layer_inputs();
        self.layers.push(vec![Neuron::new(n_inputs, 0.0, func); n]);
        self
    }

    pub fn random_layer(mut self, n: usize, func: ActivationFunction) -> Self {
        let mut layer: Vec<Neuron> = vec![];
        for _ in 0..n {
            layer.push(Neuron::random(self.get_layer_inputs(), func))
        }
        self.layers.push(layer);
        self
    }

    pub fn random_edit(&mut self) {
        let mut rng = rand::thread_rng();
        let layer = rng.gen_range(0..self.layers.len());
        let row = rng.gen_range(0..self.layers[layer].len());
        let mut change: f32 = rng.gen::<f32>() / 10.0;

        if rng.gen_bool(0.5) {
            change *= -1.0;
        }

        let neuron = &mut self.layers[layer][row];
        self.last_edit = Some(Edit {
            old: neuron.clone(),
            layer,
            row,
        });

        if rng.gen_bool(0.95) {
            let index = rng.gen_range(0..neuron.get_weights_len());
            neuron.change_weight(index, change);
        } else {
            neuron.change_bias(change);
        }
    }

    pub fn reverse_edit(&mut self) {
        match &self.last_edit {
            Some(edit) => {
                self.layers[edit.layer][edit.row] = edit.old.clone();
            }
            None => {}
        }
    }

    fn get_layer_inputs(&self) -> usize {
        match self.layers.len() {
            0 => self.n_inputs,
            _ => self.layers[self.layers.len() - 1].len(),
        }
    }
    pub fn with_weights(mut self, weights: Vec<Vec<f32>>) -> Self {
        match self.layers.last_mut() {
            None => panic!("tried to add weights before layers!"),
            Some(layer) => layer
                .iter_mut()
                .zip(weights)
                .for_each(|(neuron, weight)| neuron.set_weights(weight)),
        }
        self
    }

    pub fn with_bias(mut self, baises: Vec<f32>) -> Self {
        match self.layers.last_mut() {
            None => panic!("tried to add biases before layers!"),
            Some(layer) => layer
                .iter_mut()
                .zip(baises)
                .for_each(|(neuron, bais)| neuron.set_bias(bais)),
        }
        self
    }

    pub fn run(&self, input: &Vec<f32>) -> Result<Vec<f32>, NNError> {
        if input.len() != self.n_inputs {
            return Err(NNError::IncorectInputLength);
        }
        let mut data = input.clone();
        for layer in &self.layers {
            data = layer
                .iter()
                .map(|neuron| neuron.compute(&data))
                .collect::<Vec<f32>>();
        }

        Ok(data)
    }

    pub fn bulk_run(&self, inputs: &Vec<Vec<f32>>) -> Vec<Result<Vec<f32>, NNError>> {
        inputs.par_iter().map(|input| self.run(input)).collect()
    }
}

#[derive(Debug)]
pub enum NNError {
    IncorectInputLength,
}
#[derive(Debug, Default, Clone, Copy)]
pub enum ActivationFunction {
    #[default]
    ReLU,
    Linear,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn minimal_test() {
        let net = NeuralNetwork::new(1)
            .add_layer(10, ActivationFunction::ReLU)
            .add_layer(5, ActivationFunction::Linear);
        assert_eq!(net.run(&vec![1.0]).unwrap(), vec![10.0; 5])
    }

    #[test]
    fn better_test() {
        let net = NeuralNetwork::new(2)
            .add_layer(4, ActivationFunction::ReLU)
            .with_weights(vec![
                vec![1.1, -0.93],
                vec![-0.9, -0.96],
                vec![1.2, 0.81],
                vec![-0.91, 0.95],
            ])
            .with_bias(vec![0.048, 0.12, 0.083, -0.02])
            .add_layer(1, ActivationFunction::Linear)
            .with_weights(vec![vec![-1.4, 1.3, 1.4, -1.3]]);

        assert_eq!(net.run(&vec![3.0, 3.0]).unwrap()[0] > 0.0, true);
        assert_eq!(net.run(&vec![-3.0, -3.0]).unwrap()[0] > 0.0, true);
        assert_eq!(net.run(&vec![3.0, -3.0]).unwrap()[0] < 0.0, true);
        assert_eq!(net.run(&vec![-3.0, 3.0]).unwrap()[0] < 0.0, true);
    }
}
