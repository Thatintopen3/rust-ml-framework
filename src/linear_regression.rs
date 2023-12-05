
use ndarray::{Array1, Array2, Axis};

pub struct LinearRegression {
    weights: Option<Array2<f64>>,
    bias: Option<f64>,
    learning_rate: f64,
    iterations: usize,
}

impl LinearRegression {
    pub fn new(learning_rate: f64, iterations: usize) -> Self {
        LinearRegression {
            weights: None,
            bias: None,
            learning_rate,
            iterations,
        }
    }

    pub fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>) {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        self.weights = Some(Array2::zeros((n_features, 1)));
        self.bias = Some(0.0);

        for _ in 0..self.iterations {
            let y_predicted = x.dot(self.weights.as_ref().unwrap()) + self.bias.unwrap();

            let dw = (2.0 / n_samples as f64) * x.t().dot(&(y_predicted - y));
            let db = (2.0 / n_samples as f64) * (y_predicted - y).sum();

            *self.weights.as_mut().unwrap() -= self.learning_rate * dw;
            *self.bias.as_mut().unwrap() -= self.learning_rate * db;
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array2<f64> {
        x.dot(self.weights.as_ref().unwrap()) + self.bias.unwrap()
    }
}
