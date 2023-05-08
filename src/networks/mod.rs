mod neuron;

pub struct Network {
    layers: Vec<Vec<neuron::Neuron>>,
    n_inputs: usize,
}

impl Network {
    pub fn new(n_inputs: usize) -> Self {
        Self {
            layers: Vec::new(),
            n_inputs,
        }
    }

    pub fn add_layer(mut self, n: usize, func: ActivationFunction) -> Self {
        let n_inputs = match self.layers.len() {
            0 => self.n_inputs,
            _ => self.layers[self.layers.len() - 1].len(),
        };
        self.layers
            .push(vec![neuron::Neuron::new(n_inputs, 1.0, func); n]);
        self
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum ActivationFunction {
    #[default]
    ReLU,
    Linear,
}
