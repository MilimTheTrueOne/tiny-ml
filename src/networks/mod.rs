use self::neuron::Neuron;
use rand::Rng;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

mod neuron;

/// A simple Neural Network
pub struct NeuralNetwork {
    layers: Vec<Vec<Neuron>>,
    n_inputs: usize,
    last_edit: Option<Edit>,
    longest_layer: usize,
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
            longest_layer: 0,
        }
    }

    /// adds a layer to the model
    pub fn add_layer(mut self, n: usize, func: ActivationFunction) -> Self {
        let n_inputs = self.get_layer_inputs();
        self.layers.push(vec![Neuron::new(n_inputs, 0.0, func); n]);
        if n > self.longest_layer {
            self.longest_layer = n + 1;
        }
        self
    }

    /// adds a randomized layer to the model
    pub fn random_layer(mut self, n: usize, func: ActivationFunction) -> Self {
        let mut layer: Vec<Neuron> = vec![];
        for _ in 0..n {
            layer.push(Neuron::random(self.get_layer_inputs(), func))
        }
        self.layers.push(layer);
        if n > self.longest_layer {
            self.longest_layer = n + 1;
        }
        self
    }

    /// randomly edits some neuron
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

    /// Reverses the last random edit
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

    /// Adds custom weights to the last layer of the model
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

    /// adds custom biases to the last layer of the model
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

    /// runs the model on the given input, fails if the input had incorrect size
    pub fn run(&self, input: &Vec<f32>) -> Vec<f32>{
        let mut data = Vec::with_capacity(self.longest_layer);
        unsafe { data.set_len(input.len())}
        for i in 0..input.len() {
            data[i] = input[i];
        }

        let mut temp = Vec::with_capacity(self.longest_layer);
        for layer in &self.layers {
            unsafe { temp.set_len(layer.len()) }

            for (i, neuron) in layer.iter().enumerate() {
                temp[i] = neuron.compute(&data);
            }

            (data, temp) = (temp, data);
        }

        data
    }

    /// runs the model on a large set of data. Uses rayon for faster computation
    pub fn par_run(&self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        inputs.par_iter().map(|input| self.run(input)).collect()
    }
}

/// The activation function neurons are to use
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
        assert_eq!(net.run(&vec![1.0]), vec![10.0; 5])
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

        assert_eq!(net.run(&vec![3.0, 3.0])[0] > 0.0, true);
        assert_eq!(net.run(&vec![-3.0, -3.0])[0] > 0.0, true);
        assert_eq!(net.run(&vec![3.0, -3.0])[0] < 0.0, true);
        assert_eq!(net.run(&vec![-3.0, 3.0])[0] < 0.0, true);
    }
}
