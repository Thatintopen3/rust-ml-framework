
use ndarray::{array, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::utils::{sigmoid, sigmoid_prime};

pub struct NeuralNetwork {
    input_nodes: usize,
    hidden_nodes: usize,
    output_nodes: usize,
    weights_ih: Array2<f64>,
    weights_ho: Array2<f64>,
    bias_h: Array2<f64>,
    bias_o: Array2<f64>,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(input_nodes: usize, hidden_nodes: usize, output_nodes: usize) -> Self {
        let weights_ih = Array2::random((hidden_nodes, input_nodes), Uniform::new(-1.0, 1.0));
        let weights_ho = Array2::random((output_nodes, hidden_nodes), Uniform::new(-1.0, 1.0));

        let bias_h = Array2::random((hidden_nodes, 1), Uniform::new(-1.0, 1.0));
        let bias_o = Array2::random((output_nodes, 1), Uniform::new(-1.0, 1.0));

        NeuralNetwork {
            input_nodes,
            hidden_nodes,
            output_nodes,
            weights_ih,
            weights_ho,
            bias_h,
            bias_o,
            learning_rate: 0.1,
        }
    }

    pub fn predict(&self, input_array: &Array1<f64>) -> Array1<f64> {
        let inputs = input_array.to_owned().insert_axis(Axis(1));

        let hidden = self.weights_ih.dot(&inputs) + &self.bias_h;
        let hidden_activated = hidden.mapv(sigmoid);

        let output = self.weights_ho.dot(&hidden_activated) + &self.bias_o;
        output.mapv(sigmoid).remove_axis(Axis(1))
    }

    pub fn train(&mut self, input_data: &Array2<f64>, target_data: &Array2<f64>, epochs: usize) {
        for _ in 0..epochs {
            for i in 0..input_data.shape()[0] {
                let inputs = input_data.row(i).to_owned().insert_axis(Axis(1));
                let targets = target_data.row(i).to_owned().insert_axis(Axis(1));

                // Feedforward
                let hidden = self.weights_ih.dot(&inputs) + &self.bias_h;
                let hidden_activated = hidden.mapv(sigmoid);

                let output = self.weights_ho.dot(&hidden_activated) + &self.bias_o;
                let outputs = output.mapv(sigmoid);

                // Backpropagation
                let output_errors = &targets - &outputs;
                let gradients = outputs.mapv(sigmoid_prime) * &output_errors * self.learning_rate;
                let delta_weights_ho = gradients.dot(&hidden_activated.t());

                self.weights_ho += &delta_weights_ho;
                self.bias_o += &gradients;

                let hidden_errors = self.weights_ho.t().dot(&output_errors);
                let hidden_gradients = hidden_activated.mapv(sigmoid_prime) * &hidden_errors * self.learning_rate;
                let delta_weights_ih = hidden_gradients.dot(&inputs.t());

                self.weights_ih += &delta_weights_ih;
                self.bias_h += &hidden_gradients;
            }
        }
    }
}
