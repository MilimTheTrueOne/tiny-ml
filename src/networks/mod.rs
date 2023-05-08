mod neuron;

pub struct NeuralNetwork {
    layers: Vec<Vec<neuron::Neuron>>,
    n_inputs: usize,
}

impl NeuralNetwork {
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
            .push(vec![neuron::Neuron::new(n_inputs, 0.0, func); n]);
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
        // 
        let net = NeuralNetwork::new(1).add_layer(10, ActivationFunction::ReLU).add_layer(5, ActivationFunction::Linear);
        assert_eq!(net.run(&vec![1.0]).unwrap(), vec![10.0; 10])
    }
}

