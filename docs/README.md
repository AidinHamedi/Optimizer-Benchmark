# Optimizer Benchmark

A benchmarking suite for evaluating and comparing **PyTorch optimization algorithms** on 2D mathematical functions.
This project uses **[pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer)** and **Optuna** for hyperparameter tuning, and generates visualizations of optimizer trajectories across optimization test functions.

> [!WARNING]
> **Important Limitations**: These benchmark results are based on synthetic 2D functions and may not reflect real-world performance when training actual neural networks. The rankings should only be used as a reference, not as definitive guidance for choosing optimizers in practical applications.


## 🌶️ Features

* Benchmarks **most of the supported optimizers** in `pytorch_optimizer`.
* Hyperparameter search with **Optuna (TPE sampler)**.
* Visualization of optimization trajectories on:
  * Ackley
  * Cross-in-Tray
  * Drop-Wave
  * Eggholder
  * Gramacy & Lee (2D) *(Not yet added to the results)*
  * Griewank
  * Holder Table *(Not yet added to the results)*
  * Langermann
  * Lévy
  * Lévy 13 *(Not yet added to the results)*
  * Rastrigin
  * Rosenbrock
  * Schaffer 2
  * Schaffer 4 *(Not yet added to the results)*
  * Shubert
  * Styblinski–Tang
* Configurable **search spaces**, **iteration counts**, **ignored optimizers** and... via `config.toml`.
* Saves results and plots for later analysis.

## 🚀 Quick Start

```bash
# Clone repository
git clone --depth 1 https://github.com/AidinHamedi/ML-Optimizer-Benchmark.git
cd ML-Optimizer-Benchmark

# Install dependencies
uv sync

# Run the benchmark
python runner.py
```

The script will:

1. Load settings from `config.toml`.
2. Iterate through available optimizers.
3. Run hyperparameter tuning with Optuna.
4. Save results and visualizations under `./results/`.

## 📊 Visualizations
> ### Newest release 📦
> #### [Go to newest release](https://github.com/Aidinhamedi/ML-Optimizer-Benchmark/releases/latest)

{% include comparison.md %}

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## 📚 References

- Virtual Library of Simulation Experiments: *Test Functions and Datasets for Optimization Algorithms*.
  Source: Simon Fraser University
  [https://www.sfu.ca/~ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html)
  Curated by Derek Bingham — For inquiries: dbingham@stat.sfu.ca

- Kim, H. (2021). *pytorch_optimizer: optimizer & lr scheduler & loss function collections in PyTorch* (Version 2.12.0) [Computer software].
  https://github.com/kozistr/pytorch_optimizer


## 📝 License

<pre>
 Copyright (c) 2025 Aidin Hamedi

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
</pre>
