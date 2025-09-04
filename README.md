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
  * Gramacy & Lee (2D) (Not yet added to the results)
  * Griewank
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
|      1 | emonavi            |              2.57147 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emonavi)            |
|      2 | emofact            |              2.8488  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emofact)            |
|      3 | emozeal            |              2.8488  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emozeal)            |
|      4 | yogi               |              2.97492 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/yogi)               |
|      5 | signsgd            |              3.04044 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/signsgd)            |
|      6 | sophiah            |              3.0733  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sophiah)            |
|      7 | focus              |              3.08895 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/focus)              |
|      8 | tiger              |              3.10801 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/tiger)              |
|      9 | stablespam         |              3.60424 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/stablespam)         |
|     10 | soap               |              3.69787 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/soap)               |
|     11 | apollo             |              3.70088 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/apollo)             |
|     12 | fira               |              3.70088 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fira)               |
|     13 | galore             |              3.70088 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/galore)             |
|     14 | adam               |              4.03249 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adam)               |
|     15 | adanorm            |              4.1164  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adanorm)            |
|     16 | adamp              |              4.18049 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamp)              |
|     17 | adagc              |              4.24922 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adagc)              |
|     18 | kron               |              4.34267 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/kron)               |
|     19 | aida               |              4.38717 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/aida)               |
|     20 | swats              |              4.40458 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/swats)              |
|     21 | spam               |              4.48192 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/spam)               |
|     22 | ademamix           |              4.61658 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ademamix)           |
|     23 | emolynx            |              4.62471 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emolynx)            |
|     24 | lion               |              4.62471 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lion)               |
|     25 | adatam             |              4.80643 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adatam)             |
|     26 | adashift           |              5.12098 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adashift)           |
|     27 | nadam              |              5.13738 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/nadam)              |
|     28 | adammini           |              5.14225 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adammini)           |
|     29 | fromage            |              5.19338 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fromage)            |
|     30 | ranger25           |              5.34414 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger25)           |
|     31 | novograd           |              5.36078 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/novograd)           |
|     32 | adadelta           |              5.4427  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adadelta)           |
|     33 | adabelief          |              5.46737 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adabelief)          |
|     34 | adalite            |              5.69652 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adalite)            |
|     35 | adasmooth          |              5.71434 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adasmooth)          |
|     36 | exadam             |              5.98366 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/exadam)             |
|     37 | nero               |              6.13398 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/nero)               |
|     38 | fadam              |              6.3439  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fadam)              |
|     39 | adahessian         |              6.50315 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adahessian)         |
|     40 | laprop             |              6.54265 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/laprop)             |
|     41 | adapnm             |              6.66368 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adapnm)             |
|     42 | scionlight         |              6.67052 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scionlight)         |
|     43 | dadaptadan         |              6.7595  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptadan)         |
|     44 | ranger21           |              7.04377 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger21)           |
|     45 | rmsprop            |              7.14312 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/rmsprop)            |
|     46 | vsgd               |              7.28179 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/vsgd)               |
|     47 | adopt              |              7.40785 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adopt)              |
|     48 | adamax             |              7.6868  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamax)             |
|     49 | simplifiedademamix |              7.81188 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/simplifiedademamix) |
|     50 | sm3                |              8.17468 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sm3)                |
|     51 | ftrl               |              8.18237 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ftrl)               |
|     52 | schedulefreeadamw  |              8.27146 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreeadamw)  |
|     53 | asgd               |              8.47956 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/asgd)               |
|     54 | came               |              8.70882 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/came)               |
|     55 | lamb               |              8.77041 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lamb)               |
|     56 | tam                |              8.94983 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/tam)                |
|     57 | adan               |              9.30751 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adan)               |
|     58 | emoneco            |              9.58197 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emoneco)            |
|     59 | apollodqn          |              9.63575 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/apollodqn)          |
|     60 | padam              |             10.4026  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/padam)              |
|     61 | diffgrad           |             10.7386  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/diffgrad)           |
|     62 | grokfastadamw      |             10.9842  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/grokfastadamw)      |
|     63 | lars               |             12.0577  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lars)               |
|     64 | racs               |             12.0613  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/racs)               |
|     65 | sgd                |             12.6166  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgd)                |
|     66 | sgdp               |             12.6166  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgdp)               |
|     67 | prodigy            |             13.0301  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/prodigy)            |
|     68 | pid                |             13.1682  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/pid)                |
|     69 | scion              |             13.769   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scion)              |
|     70 | scalableshampoo    |             13.9738  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scalableshampoo)    |
|     71 | dadaptlion         |             14.0853  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptlion)         |
|     72 | radam              |             14.4628  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/radam)              |
|     73 | adabound           |             14.5819  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adabound)           |
|     74 | avagrad            |             15.9035  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/avagrad)            |
|     75 | qhm                |             16.0966  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/qhm)                |
|     76 | ranger             |             16.7279  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger)             |
|     77 | pnm                |             16.8118  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/pnm)                |
|     78 | aggmo              |             16.8834  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/aggmo)              |
|     79 | schedulefreesgd    |             17.2629  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreesgd)    |
|     80 | adai               |             17.3909  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adai)               |
|     81 | accsgd             |             17.5586  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/accsgd)             |
|     82 | madgrad            |             17.8124  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/madgrad)            |
|     83 | dadaptadagrad      |             18.0346  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptadagrad)      |
|     84 | qhadam             |             18.6783  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/qhadam)             |
|     85 | dadaptsgd          |             18.9387  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptsgd)          |
|     86 | gravity            |             19.0526  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/gravity)            |
|     87 | schedulefreeradam  |             20.9976  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreeradam)  |
|     88 | kate               |             21.4434  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/kate)               |
|     89 | adamg              |             21.4805  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamg)              |
|     90 | adamod             |             21.6752  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamod)             |
|     91 | grams              |             21.7464  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/grams)              |
|     92 | sgdsai             |             21.7469  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgdsai)             |
|     93 | adafactor          |             22.9181  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adafactor)          |
|     94 | mars               |             26.4904  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/mars)               |
|     95 | shampoo            |             28.3101  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/shampoo)            |
|     96 | srmm               |             39.6709  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/srmm)               |
|     97 | adams              |            125.205   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adams)              |

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
