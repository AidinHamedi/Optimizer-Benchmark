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
|      1 | adammini           |              1.87047 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adammini)           |
|      2 | amos               |              2.21237 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/amos)               |
|      3 | soap               |              2.65934 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/soap)               |
|      4 | stableadamw        |              2.66532 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/stableadamw)        |
|      5 | adagc              |              2.69131 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adagc)              |
|      6 | emonavi            |              3.31596 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emonavi)            |
|      7 | emofact            |              3.4946  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emofact)            |
|      8 | emozeal            |              3.4946  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emozeal)            |
|      9 | adamw              |              3.65314 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamw)              |
|     10 | emolynx            |              3.74591 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emolynx)            |
|     11 | sophiah            |              3.9992  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sophiah)            |
|     12 | tiger              |              4.02113 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/tiger)              |
|     13 | signsgd            |              4.11017 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/signsgd)            |
|     14 | emoneco            |              4.11926 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emoneco)            |
|     15 | apollo             |              4.22118 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/apollo)             |
|     16 | fira               |              4.22118 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fira)               |
|     17 | galore             |              4.22118 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/galore)             |
|     18 | spam               |              4.42994 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/spam)               |
|     19 | yogi               |              4.44999 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/yogi)               |
|     20 | swats              |              4.89521 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/swats)              |
|     21 | kron               |              4.95861 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/kron)               |
|     22 | focus              |              5.07233 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/focus)              |
|     23 | sgdsai             |              5.29109 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgdsai)             |
|     24 | lion               |              5.47808 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lion)               |
|     25 | ademamix           |              5.52255 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ademamix)           |
|     26 | stablespam         |              5.58092 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/stablespam)         |
|     27 | adamp              |              5.95326 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamp)              |
|     28 | adam               |              5.9833  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adam)               |
|     29 | adanorm            |              6.05375 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adanorm)            |
|     30 | fromage            |              6.37057 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fromage)            |
|     31 | novograd           |              6.44134 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/novograd)           |
|     32 | adatam             |              6.46243 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adatam)             |
|     33 | aida               |              6.74602 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/aida)               |
|     34 | adalite            |              6.77547 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adalite)            |
|     35 | adashift           |              6.77979 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adashift)           |
|     36 | ranger25           |              6.83287 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger25)           |
|     37 | nero               |              7.08884 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/nero)               |
|     38 | adadelta           |              7.29552 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adadelta)           |
|     39 | scionlight         |              7.4069  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scionlight)         |
|     40 | nadam              |              7.47419 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/nadam)              |
|     41 | exadam             |              7.68435 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/exadam)             |
|     42 | adasmooth          |              7.72783 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adasmooth)          |
|     43 | adapnm             |              8.09707 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adapnm)             |
|     44 | fadam              |              8.28703 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fadam)              |
|     45 | adabelief          |              8.3582  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adabelief)          |
|     46 | adahessian         |              8.67695 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adahessian)         |
|     47 | laprop             |              9.19151 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/laprop)             |
|     48 | adopt              |              9.34072 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adopt)              |
|     49 | adamax             |              9.48154 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamax)             |
|     50 | simplifiedademamix |              9.83305 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/simplifiedademamix) |
|     51 | lamb               |              9.85373 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lamb)               |
|     52 | ranger21           |              9.98826 | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger21)           |
|     53 | vsgd               |             10.0006  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/vsgd)               |
|     54 | rmsprop            |             10.097   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/rmsprop)            |
|     55 | schedulefreeadamw  |             10.5053  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreeadamw)  |
|     56 | sm3                |             11.0035  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sm3)                |
|     57 | dadaptadan         |             11.1451  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptadan)         |
|     58 | ftrl               |             11.1837  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ftrl)               |
|     59 | apollodqn          |             11.8016  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/apollodqn)          |
|     60 | tam                |             12.0729  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/tam)                |
|     61 | asgd               |             12.5142  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/asgd)               |
|     62 | diffgrad           |             12.5589  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/diffgrad)           |
|     63 | adan               |             13.4256  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adan)               |
|     64 | prodigy            |             13.5376  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/prodigy)            |
|     65 | padam              |             13.7359  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/padam)              |
|     66 | racs               |             15.9594  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/racs)               |
|     67 | grokfastadamw      |             16.013   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/grokfastadamw)      |
|     68 | sgd                |             16.3659  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgd)                |
|     69 | sgdp               |             16.3659  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgdp)               |
|     70 | radam              |             16.7106  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/radam)              |
|     71 | lars               |             16.8783  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lars)               |
|     72 | pid                |             17.501   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/pid)                |
|     73 | dadaptadagrad      |             18.2722  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptadagrad)      |
|     74 | scalableshampoo    |             18.4725  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scalableshampoo)    |
|     75 | adabound           |             18.6868  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adabound)           |
|     76 | ranger             |             19.0237  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger)             |
|     77 | qhm                |             19.5376  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/qhm)                |
|     78 | pnm                |             20.752   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/pnm)                |
|     79 | avagrad            |             21.089   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/avagrad)            |
|     80 | accsgd             |             21.0943  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/accsgd)             |
|     81 | adai               |             21.1059  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adai)               |
|     82 | schedulefreesgd    |             21.2877  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreesgd)    |
|     83 | aggmo              |             21.64    | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/aggmo)              |
|     84 | gravity            |             22.0485  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/gravity)            |
|     85 | dadaptsgd          |             22.1239  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptsgd)          |
|     86 | grams              |             22.1362  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/grams)              |
|     87 | dadaptlion         |             22.2188  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptlion)         |
|     88 | kate               |             22.3818  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/kate)               |
|     89 | madgrad            |             22.6336  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/madgrad)            |
|     90 | qhadam             |             22.7482  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/qhadam)             |
|     91 | schedulefreeradam  |             22.7926  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreeradam)  |
|     92 | came               |             23.3895  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/came)               |
|     93 | adamg              |             23.6004  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamg)              |
|     94 | adamod             |             23.6852  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamod)             |
|     95 | adafactor          |             23.9996  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adafactor)          |
|     96 | mars               |             26.0127  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/mars)               |
|     97 | shampoo            |             29.6867  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/shampoo)            |
|     98 | srmm               |             43.8504  | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/srmm)               |
|     99 | adams              |            127.388   | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adams)              |
|    100 | scion              |           1236.98    | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scion)              |

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
