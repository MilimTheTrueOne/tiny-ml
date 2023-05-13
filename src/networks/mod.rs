use self::neuron::Neuron;
use rand::Rng;

#[cfg(feature = "parallization")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

mod neuron;

/// A simple Neural Network
pub struct NeuralNetwork<const I: usize, const O: usize> {
    layers: Vec<Vec<Neuron>>,
    last_edit: Option<Edit>,
    /// this exist for preallocation
    longest_layer: usize,
    /// preallocated buffers
    buffers: (Vec<f32>, Vec<f32>),
}

/// struct that represents the last edit done to the NN
struct Edit {
    old: Neuron,
    layer: usize,
    row: usize,
}

impl<const I: usize, const O: usize> Default for NeuralNetwork<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const I: usize, const O: usize> NeuralNetwork<I, O> {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            last_edit: None,
            longest_layer: 0,
            buffers: (vec![], vec![]),
        }
    }

    /// adds a layer with `n` neurons and the specified activation function
    pub fn add_layer(mut self, n: usize, func: ActivationFunction) -> Self {
        let n_inputs = self.get_layer_inputs();
        self.layers.push(vec![Neuron::new(n_inputs, 0.0, func); n]);
        self.check_max_layer(n);
        self
    }

    /// checks whenever the newly added layer is longer then previus longest layer
    fn check_max_layer(&mut self, n: usize) {
        if n > self.longest_layer {
            self.longest_layer = n;
            self.buffers = (Vec::with_capacity(n), Vec::with_capacity(n))
        }
    }

    /// adds a layer with `n` neurons and randomized weights/bias to the model
    pub fn random_layer(mut self, n: usize, func: ActivationFunction) -> Self {
        let mut layer: Vec<Neuron> = vec![];
        for _ in 0..n {
            layer.push(Neuron::random(self.get_layer_inputs(), func))
        }
        self.layers.push(layer);
        Self::check_max_layer(&mut self, n);
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

    /// get the inputs lenght of the last layer
    fn get_layer_inputs(&self) -> usize {
        if self.layers.is_empty() {
            return I;
        }
        self.layers[self.layers.len() - 1].len()
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

    /// runs the model on the given input.
    #[inline]
    pub fn run(&mut self, input: &[f32; I]) -> [f32; O] {
        // data and temp vec
        let mut data = &mut self.buffers.0;
        let mut temp = &mut self.buffers.1;

        // we only read values we initialise in this loop, so this is 100% safe
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(input.len())
        }

        // put input data in temp vec
        data[..input.len()].copy_from_slice(&input[..]);

        for layer in &self.layers {
            // we only read values we initialise in this loop, so this is 100% safe
            #[allow(clippy::uninit_vec)]
            unsafe {
                temp.set_len(layer.len())
            }

            for (i, neuron) in layer.iter().enumerate() {
                temp[i] = neuron.compute(data);
            }

            (data, temp) = (temp, data);
        }

        let mut out = [0.0; O];
        out[..O].copy_from_slice(&data[..O]);
        out
    }

    /// runs the model on the given input. This does not buffer some things, and thus is slower
    /// as ist does extra allocations work.
    #[inline]
    pub fn unbufferd_run(&self, input: &[f32; I]) -> [f32; O] {
        let mut data = Vec::with_capacity(self.longest_layer);

        // we only read values we initialise in this loop, so this is 100% safe
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(input.len())
        }

        data[..input.len()].copy_from_slice(&input[..]);

        let mut temp = Vec::with_capacity(self.longest_layer);
        for layer in &self.layers {
            // we only read values we initialise in this loop, so this is 100% safe
            #[allow(clippy::uninit_vec)]
            unsafe {
                temp.set_len(layer.len())
            }

            for (i, neuron) in layer.iter().enumerate() {
                temp[i] = neuron.compute(&data);
            }

            (data, temp) = (temp, data);
        }

        let mut out = [0.0; O];
        out[..O].copy_from_slice(&data[..O]);
        out
    }

    /// runs the model on a large set of data. Uses rayon for faster computation
    #[cfg(feature = "parallization")]
    pub fn par_run(&self, inputs: &Vec<[f32; I]>) -> Vec<[f32; O]> {
        inputs
            .par_iter()
            .map(|input| self.unbufferd_run(input))
            .collect()
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
        let mut net = NeuralNetwork::new()
            .add_layer(10, ActivationFunction::ReLU)
            .add_layer(5, ActivationFunction::Linear);
        assert_eq!(net.run(&[1.0]), [10.0; 5])
    }

    #[test]
    fn better_test() {
        let mut net: NeuralNetwork<2, 1> = NeuralNetwork::new()
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

        assert!(net.run(&[3.0, 3.0])[0] > 0.0);
        assert!(net.run(&[-3.0, -3.0])[0] > 0.0);
        assert!(net.run(&[3.0, -3.0])[0] < 0.0);
        assert!(net.run(&[-3.0, 3.0])[0] < 0.0);
    }
}
