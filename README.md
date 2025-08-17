# Optimizer Benchmark

A benchmarking suite for evaluating and comparing **PyTorch optimization algorithms** on 2D mathematical functions.
This project uses **[pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)** and **Optuna** for hyperparameter tuning, and generates visualizations of optimizer trajectories across optimization test functions.

## 🚀 Features

* Benchmarks **all supported optimizers** in `pytorch-optimizer`.
* Hyperparameter search with **Optuna (TPE sampler)**.
* Visualization of optimization trajectories on:

  * Ackley
  * Cross-in-Tray
  * Eggholder
  * Langermann
  * Lévy
  * Rastrigin
  * Rosenbrock
  * Schaffer 2
  * Shubert
  * Styblinski–Tang
* Configurable **search spaces**, **iteration counts**, **ignored optimizers** and... via `config.toml`.
* Saves results and plots for later analysis.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## 📚 References

- Virtual Library of Simulation Experiments: *Test Functions and Datasets for Optimization Algorithms*.
  Source: Simon Fraser University
  [https://www.sfu.ca/~ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html)
  Curated by Derek Bingham — For inquiries: dbingham@stat.sfu.ca


## 📝 License

<pre>
 Copyright (c) 2025 Aidin Hamedi

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
</pre>
