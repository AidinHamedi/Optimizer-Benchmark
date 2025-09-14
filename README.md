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
  * Gramacy & Lee (2D)
  * Griewank
  * Holder Table
  * Langermann
  * Lévy
  * Lévy 13
  * Rastrigin
  * Rosenbrock
  * Schaffer 2
  * Schaffer 4
  * Shubert
  * Styblinski–Tang
* Configurable **search spaces**, **iteration counts**, **ignored optimizers** and... via `config.toml`.
* Saves results and plots for later analysis.

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

The script will:

1. Load settings from `config.toml`.
2. Iterate through available optimizers.
3. Run hyperparameter tuning with Optuna.
4. Save results and visualizations under `./results/`.

## 📊 Visualizations
> ### Newest release 📦
> #### [Go to newest release](https://github.com/Aidinhamedi/Optimizer-Benchmark/releases/latest)

|   Rank | Optimizer          | Average Error Rate   | Vis                                                                                 |
|--------|--------------------|----------------------|-------------------------------------------------------------------------------------|
|      1 | adatam             | 3.3906               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adatam)             |
|      2 | emonavi            | 3.4562               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emonavi)            |
|      3 | emofact            | 3.4753               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emofact)            |
|      4 | emozeal            | 3.4753               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emozeal)            |
|      5 | tiger              | 3.9969               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/tiger)              |
|      6 | signsgd            | 4.2924               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/signsgd)            |
|      7 | sophiah            | 4.723                | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sophiah)            |
|      8 | emolynx            | 5.1373               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emolynx)            |
|      9 | lion               | 5.1373               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lion)               |
|     10 | focus              | 5.3087               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/focus)              |
|     11 | kron               | 6.0274               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/kron)               |
|     12 | adabelief          | 6.4662               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adabelief)          |
|     13 | novograd           | 6.5473               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/novograd)           |
|     14 | exadam             | 6.7497               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/exadam)             |
|     15 | fromage            | 7.0991               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fromage)            |
|     16 | lamb               | 7.5312               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lamb)               |
|     17 | ademamix           | 7.948                | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ademamix)           |
|     18 | adammini           | 8.1328               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adammini)           |
|     19 | adadelta           | 8.2486               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adadelta)           |
|     20 | adagc              | 8.4104               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adagc)              |
|     21 | apollo             | 8.4469               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/apollo)             |
|     22 | fira               | 8.4469               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fira)               |
|     23 | galore             | 8.4469               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/galore)             |
|     24 | adapnm             | 8.5628               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adapnm)             |
|     25 | scionlight         | 8.6491               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scionlight)         |
|     26 | adalite            | 8.7071               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adalite)            |
|     27 | emoneco            | 8.7699               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/emoneco)            |
|     28 | stablespam         | 8.7751               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/stablespam)         |
|     29 | adam               | 8.8905               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adam)               |
|     30 | soap               | 8.9126               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/soap)               |
|     31 | nadam              | 9.2417               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/nadam)              |
|     32 | adamp              | 9.5865               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamp)              |
|     33 | adanorm            | 9.7223               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adanorm)            |
|     34 | rmsprop            | 9.7358               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/rmsprop)            |
|     35 | aida               | 9.7844               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/aida)               |
|     36 | nero               | 9.8053               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/nero)               |
|     37 | diffgrad           | 9.8235               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/diffgrad)           |
|     38 | laprop             | 10.0266              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/laprop)             |
|     39 | spam               | 10.0438              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/spam)               |
|     40 | adasmooth          | 10.158               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adasmooth)          |
|     41 | tam                | 10.4039              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/tam)                |
|     42 | swats              | 10.5598              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/swats)              |
|     43 | asgd               | 10.8174              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/asgd)               |
|     44 | came               | 10.9854              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/came)               |
|     45 | adahessian         | 11.1066              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adahessian)         |
|     46 | schedulefreeadamw  | 12.1115              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreeadamw)  |
|     47 | adashift           | 12.7504              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adashift)           |
|     48 | adamax             | 12.8721              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamax)             |
|     49 | yogi               | 12.9361              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/yogi)               |
|     50 | vsgd               | 12.9511              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/vsgd)               |
|     51 | adan               | 13.0089              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adan)               |
|     52 | schedulefreesgd    | 13.0602              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreesgd)    |
|     53 | fadam              | 13.0729              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/fadam)              |
|     54 | adopt              | 13.4732              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adopt)              |
|     55 | padam              | 14.0248              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/padam)              |
|     56 | dadaptadan         | 14.4575              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptadan)         |
|     57 | grokfastadamw      | 15.0406              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/grokfastadamw)      |
|     58 | dadaptadagrad      | 15.2034              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptadagrad)      |
|     59 | simplifiedademamix | 15.7192              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/simplifiedademamix) |
|     60 | ranger21           | 16.0177              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger21)           |
|     61 | scion              | 16.119               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scion)              |
|     62 | sgd                | 16.6986              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgd)                |
|     63 | sgdp               | 16.6986              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgdp)               |
|     64 | ftrl               | 17.6352              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ftrl)               |
|     65 | sm3                | 17.7487              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sm3)                |
|     66 | adabound           | 18.7915              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adabound)           |
|     67 | pid                | 18.9555              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/pid)                |
|     68 | kate               | 19.6937              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/kate)               |
|     69 | apollodqn          | 19.7824              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/apollodqn)          |
|     70 | dadaptlion         | 20.2519              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptlion)         |
|     71 | lars               | 20.3135              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/lars)               |
|     72 | prodigy            | 20.6609              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/prodigy)            |
|     73 | grams              | 20.8008              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/grams)              |
|     74 | scalableshampoo    | 21.1188              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/scalableshampoo)    |
|     75 | shampoo            | 21.6842              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/shampoo)            |
|     76 | accsgd             | 23.1303              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/accsgd)             |
|     77 | qhm                | 23.5284              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/qhm)                |
|     78 | ranger             | 23.6326              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger)             |
|     79 | racs               | 23.9282              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/racs)               |
|     80 | adai               | 23.9893              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adai)               |
|     81 | srmm               | 24.2722              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/srmm)               |
|     82 | aggmo              | 24.276               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/aggmo)              |
|     83 | pnm                | 25.1595              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/pnm)                |
|     84 | madgrad            | 26.0235              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/madgrad)            |
|     85 | schedulefreeradam  | 26.418               | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/schedulefreeradam)  |
|     86 | adamg              | 26.5044              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamg)              |
|     87 | gravity            | 27.1906              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/gravity)            |
|     88 | radam              | 27.3285              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/radam)              |
|     89 | avagrad            | 27.36                | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/avagrad)            |
|     90 | sgdsai             | 28.2559              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/sgdsai)             |
|     91 | dadaptsgd          | 28.4846              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/dadaptsgd)          |
|     92 | adamod             | 29.2631              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adamod)             |
|     93 | adafactor          | 32.3336              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adafactor)          |
|     94 | mars               | 38.0967              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/mars)               |
|     95 | adams              | 56.9071              | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/adams)              |
|     96 | qhadam             | 386.1081             | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/qhadam)             |
|     97 | ranger25           | Failed ⚠️            | [Open](https://aidinhamedi.github.io/ML-Optimizer-Benchmark/vis/ranger25)           |

## 🤝 Contributing

Contributions are welcome!
In particular, I’m looking for help improving and expanding the **web page**

If you’d like to contribute, please feel free to submit a pull request or open an issue to discuss your ideas.

## 📚 References

- Virtual Library of Simulation Experiments: *Test Functions and Datasets for Optimization Algorithms*.
  Source: Simon Fraser University
  [https://www.sfu.ca/~ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html)
  Curated by Derek Bingham — For inquiries: dbingham@stat.sfu.ca

- Kim, H. (2021). *pytorch_optimizer: optimizer & lr scheduler & loss function collections in PyTorch* (Version 2.12.0) [Computer software].
  [https://github.com/kozistr/pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer)


## 📝 License

<pre>
 Copyright (c) 2025 Aidin Hamedi

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
</pre>
