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
  * Langermann
  * Lévy
  * Rastrigin
  * Rosenbrock
  * Schaffer 2
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

|   Rank | Optimizer          |   Average Error Rate | Vis                                                                                 |
|--------|--------------------|----------------------|-------------------------------------------------------------------------------------|
|      1 | emonavi            |              1.12538 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emonavi)            |
|      2 | adammini           |              1.27945 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adammini)           |
|      3 | emofact            |              1.32818 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emofact)            |
|      4 | emozeal            |              1.32818 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emozeal)            |
|      5 | stableadamw        |              1.42537 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/stableadamw)        |
|      6 | adamw              |              1.42792 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamw)              |
|      7 | adagc              |              1.49691 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adagc)              |
|      8 | soap               |              1.50217 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/soap)               |
|      9 | sgdw               |              1.58334 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgdw)               |
|     10 | apollo             |              1.84813 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/apollo)             |
|     11 | fira               |              1.84813 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fira)               |
|     12 | galore             |              1.84813 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/galore)             |
|     13 | aida               |              1.8762  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/aida)               |
|     14 | laprop             |              1.92798 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/laprop)             |
|     15 | stablespam         |              2.02185 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/stablespam)         |
|     16 | amos               |              2.04766 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/amos)               |
|     17 | adam               |              2.09923 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adam)               |
|     18 | spam               |              2.17744 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/spam)               |
|     19 | yogi               |              2.20304 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/yogi)               |
|     20 | rmsprop            |              2.22484 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/rmsprop)            |
|     21 | swats              |              2.30101 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/swats)              |
|     22 | emolynx            |              2.35384 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emolynx)            |
|     23 | ademamix           |              2.39265 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ademamix)           |
|     24 | adamp              |              2.42379 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamp)              |
|     25 | adanorm            |              2.55958 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adanorm)            |
|     26 | fromage            |              2.6313  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fromage)            |
|     27 | emoneco            |              2.66198 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emoneco)            |
|     28 | diffgrad           |              2.72364 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/diffgrad)           |
|     29 | adamax             |              2.90853 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamax)             |
|     30 | signsgd            |              3.02079 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/signsgd)            |
|     31 | adapnm             |              3.03175 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adapnm)             |
|     32 | sophiah            |              3.20866 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sophiah)            |
|     33 | asgd               |              3.25593 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/asgd)               |
|     34 | focus              |              3.26186 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/focus)              |
|     35 | lion               |              3.26327 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lion)               |
|     36 | lamb               |              3.27226 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lamb)               |
|     37 | tiger              |              3.28567 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/tiger)              |
|     38 | adabelief          |              3.35307 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adabelief)          |
|     39 | adahessian         |              3.57163 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adahessian)         |
|     40 | scionlight         |              3.6695  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scionlight)         |
|     41 | adasmooth          |              3.75005 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adasmooth)          |
|     42 | adatam             |              4.08313 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adatam)             |
|     43 | kron               |              4.11133 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/kron)               |
|     44 | adan               |              4.18387 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adan)               |
|     45 | nadam              |              4.38707 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/nadam)              |
|     46 | exadam             |              4.50333 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/exadam)             |
|     47 | sgdsai             |              4.5617  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgdsai)             |
|     48 | ranger25           |              4.71608 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger25)           |
|     49 | adashift           |              4.7847  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adashift)           |
|     50 | nero               |              4.84514 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/nero)               |
|     51 | ranger21           |              5.09579 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger21)           |
|     52 | dadaptadan         |              5.22616 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptadan)         |
|     53 | grams              |              5.56661 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/grams)              |
|     54 | prodigy            |              5.80589 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/prodigy)            |
|     55 | tam                |              5.9481  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/tam)                |
|     56 | novograd           |              5.9543  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/novograd)           |
|     57 | lars               |              6.14319 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lars)               |
|     58 | sgd                |              6.32309 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgd)                |
|     59 | sgdp               |              6.32309 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgdp)               |
|     60 | adadelta           |              6.38321 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adadelta)           |
|     61 | shampoo            |              6.56501 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/shampoo)            |
|     62 | sm3                |              6.5936  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sm3)                |
|     63 | ftrl               |              6.72551 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ftrl)               |
|     64 | schedulefreeadamw  |              6.76049 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreeadamw)  |
|     65 | adalite            |              7.17815 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adalite)            |
|     66 | scalableshampoo    |              7.21953 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scalableshampoo)    |
|     67 | dadaptlion         |              7.43713 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptlion)         |
|     68 | grokfastadamw      |              7.73076 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/grokfastadamw)      |
|     69 | adabound           |              7.99936 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adabound)           |
|     70 | pid                |              8.01779 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/pid)                |
|     71 | vsgd               |              8.10525 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/vsgd)               |
|     72 | dadaptadagrad      |              8.10725 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptadagrad)      |
|     73 | simplifiedademamix |              8.30243 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/simplifiedademamix) |
|     74 | fadam              |              8.33242 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fadam)              |
|     75 | srmm               |              8.48812 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/srmm)               |
|     76 | adopt              |              9.19321 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adopt)              |
|     77 | accsgd             |              9.49563 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/accsgd)             |
|     78 | kate               |              9.52691 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/kate)               |
|     79 | adams              |              9.64474 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adams)              |
|     80 | padam              |             10.228   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/padam)              |
|     81 | qhm                |             11.006   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/qhm)                |
|     82 | apollodqn          |             11.3717  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/apollodqn)          |
|     83 | schedulefreesgd    |             11.3966  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreesgd)    |
|     84 | aggmo              |             11.6408  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/aggmo)              |
|     85 | adai               |             12.1643  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adai)               |
|     86 | avagrad            |             12.2265  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/avagrad)            |
|     87 | pnm                |             12.2658  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/pnm)                |
|     88 | madgrad            |             12.6119  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/madgrad)            |
|     89 | racs               |             12.8971  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/racs)               |
|     90 | qhadam             |             13.3887  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/qhadam)             |
|     91 | dadaptsgd          |             13.5656  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptsgd)          |
|     92 | radam              |             13.7083  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/radam)              |
|     93 | gravity            |             14.6517  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/gravity)            |
|     94 | schedulefreeradam  |             14.9588  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreeradam)  |
|     95 | ranger             |             15.1098  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger)             |
|     96 | came               |             15.3956  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/came)               |
|     97 | adamg              |             15.7262  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamg)              |
|     98 | adamod             |             15.8158  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamod)             |
|     99 | adafactor          |             16.1021  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adafactor)          |
|    100 | mars               |             16.5068  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/mars)               |
|    101 | dadaptadam         |             17.0097  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptadam)         |
|    102 | scion              |           1583.72    | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scion)              |

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
