
mod linear_regression;
mod neural_network;
mod utils;

use linear_regression::LinearRegression;
use neural_network::NeuralNetwork;
use ndarray::{array, Array1, Array2};

fn main() {
    println!("Running Rust ML Framework Demo...");

    // Linear Regression Demo
    println!("
--- Linear Regression Demo ---");
    let mut lr = LinearRegression::new(0.01, 1000);
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![[2.0], [4.0], [5.0], [4.0]];
    lr.train(&x, &y);
    let prediction = lr.predict(&array![[5.0]]);
    println!("Prediction for 5.0: {:.4}", prediction[[0, 0]]);

    // Neural Network Demo
    println!("
--- Neural Network Demo ---");
    let mut nn = NeuralNetwork::new(2, 3, 1);
    let nn_inputs = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let nn_targets = array![[0.0], [1.0], [1.0], [0.0]];
    nn.train(&nn_inputs, &nn_targets, 10000);

    println!("Prediction for [0, 0]: {:.4}", nn.predict(&array![0.0, 0.0])[[0]]);
    println!("Prediction for [0, 1]: {:.4}", nn.predict(&array![0.0, 1.0])[[0]]);
    println!("Prediction for [1, 0]: {:.4}", nn.predict(&array![1.0, 0.0])[[0]]);
    println!("Prediction for [1, 1]: {:.4}", nn.predict(&array![1.0, 1.0])[[0]]);
}
