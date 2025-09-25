# Optimizer Benchmark

A benchmarking suite for evaluating and comparing PyTorch optimization algorithms on 2D mathematical functions.

## 🌟 Highlights

*   Benchmarks optimizers from the `pytorch_optimizer` library.
*   Uses Optuna for hyperparameter tuning.
*   Generates trajectory visualizations for each optimizer and function.
*   Presents performance rankings on a project website.
*   Configurable via a `config.toml` file.

## ℹ️ Overview

This project provides a framework to evaluate and compare the performance of various PyTorch optimizers. It uses algorithms from `pytorch_optimizer` and performs hyperparameter searches with Optuna. The benchmark is run on a suite of standard 2D mathematical test functions, and the results, including optimization trajectories, are visualized and ranked.

> [!WARNING]
> **Important Limitations**: These benchmark results are based on synthetic 2D functions and may not reflect real-world performance when training actual neural networks. The rankings should only be used as a reference, not as definitive guidance for choosing optimizers in practical applications.

## 📌 Benchmark Functions

The optimizers are evaluated on the following standard 2D test functions. Click on a function's name to learn more about it.

| Function                                                                 | Function                                                                       |
| :----------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| [Ackley](https://www.sfu.ca/~ssurjano/ackley.html)                       | [Cross-in-Tray](https://www.sfu.ca/~ssurjano/crossit.html)                     |
| [Drop-Wave](https://www.sfu.ca/~ssurjano/drop.html)                      | [Eggholder](https://www.sfu.ca/~ssurjano/egg.html)                             |
| [Gramacy & Lee](https://www.sfu.ca/~ssurjano/grlee12.html)               | [Griewank](https://www.sfu.ca/~ssurjano/griewank.html)                         |
| [Holder Table](https://www.sfu.ca/~ssurjano/holder.html)                 | [Langermann](https://www.sfu.ca/~ssurjano/langer.html)                         |
| [Lévy](https://www.sfu.ca/~ssurjano/levy.html)                           | [Lévy N. 13](https://www.sfu.ca/~ssurjano/levy13.html)                         |
| [Rastrigin](https://www.sfu.ca/~ssurjano/rastr.html)                     | [Rosenbrock](https://www.sfu.ca/~ssurjano/rosen.html)                          |
| [Schaffer N. 2](https://www.sfu.ca/~ssurjano/schaffer2.html)             | [Schaffer N. 4](https://www.sfu.ca/~ssurjano/schaffer4.html)                   |
| [Shubert](https://www.sfu.ca/~ssurjano/shubert.html)                     | [Styblinski–Tang](https://www.sfu.ca/~ssurjano/stybtang.html)                  |
| [Weierstrass](https://en.wikipedia.org/wiki/Weierstrass_function)        | [Schwefel](https://www.sfu.ca/~ssurjano/schwef.html)                           |


## 📊 Results & Visualizations

The full benchmark results, including performance rankings and detailed trajectory plots for each optimizer, are available on the project website.

#### ➡️ [**View the Optimizer Benchmark Website**](https://aidinhamedi.github.io/Optimizer-Benchmark/)
#### ➡️ [**Download the Benchmark Results**](https://github.com/Aidinhamedi/Optimizer-Benchmark/releases/latest)

## 🚀 Quick Start

```bash
# Clone repository
git clone --depth 1 https://github.com/AidinHamedi/Optimizer-Benchmark.git
cd Optimizer-Benchmark

# Install dependencies
uv sync

# Run the benchmark
python runner.py
```

The script will load settings from `config.toml`, run hyperparameter tuning for each optimizer, and save the results and visualizations to the `./results/` directory.

## 🤝 Contributing

Contributions are welcome! In particular, I’m looking for help improving and expanding the **web page**.

If you’d like to contribute, please feel free to submit a pull request or open an issue to discuss your ideas.

## 📚 References

*   Virtual Library of Simulation Experiments: *Test Functions and Datasets for Optimization Algorithms*.
    Source: Simon Fraser University
    [https://www.sfu.ca/~ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html)
    Curated by Derek Bingham — For inquiries: dbingham@stat.sfu.ca

*   Kim, H. (2021). *pytorch\_optimizer: optimizer & lr scheduler & loss function collections in PyTorch* (Version 2.12.0) \[Computer software].
    [https://github.com/kozistr/pytorch\_optimizer](https://github.com/kozistr/pytorch_optimizer)

## 📝 License

<pre>
 Copyright (c) 2025 Aidin Hamedi

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
</pre>
