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

<h4>
<details open>
<summary>Ranking by Avg Function Rank ⚡</summary>
<h6>

|   Rank (Avg Function Rank) | Optimizer          |   Average Rank | Vis                                                                              |
|----------------------------|--------------------|----------------|----------------------------------------------------------------------------------|
|                          1 | emofact            |          13    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emofact)            |
|                          2 | emozeal            |          13    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emozeal)            |
|                          3 | emonavi            |          14.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emonavi)            |
|                          4 | adatam             |          18.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adatam)             |
|                          5 | adabelief          |          20.5  | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adabelief)          |
|                          6 | signsgd            |          24.19 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/signsgd)            |
|                          7 | soap               |          25    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/soap)               |
|                          8 | tiger              |          26.44 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/tiger)              |
|                          9 | stablespam         |          26.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/stablespam)         |
|                         10 | novograd           |          26.81 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/novograd)           |
|                         11 | ademamix           |          27.19 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ademamix)           |
|                         12 | adagc              |          27.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adagc)              |
|                         13 | adam               |          28    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adam)               |
|                         14 | focus              |          28.44 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/focus)              |
|                         15 | sophiah            |          28.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sophiah)            |
|                         16 | adamp              |          29    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamp)              |
|                         17 | aida               |          29.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/aida)               |
|                         18 | adammini           |          30.12 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adammini)           |
|                         19 | kron               |          31.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/kron)               |
|                         20 | apollo             |          31.88 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/apollo)             |
|                         21 | fira               |          31.88 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/fira)               |
|                         22 | galore             |          31.88 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/galore)             |
|                         23 | emolynx            |          32.19 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emolynx)            |
|                         24 | lion               |          32.19 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/lion)               |
|                         25 | adalite            |          32.5  | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adalite)            |
|                         26 | fromage            |          34.44 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/fromage)            |
|                         27 | spam               |          34.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/spam)               |
|                         28 | adanorm            |          34.94 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adanorm)            |
|                         29 | swats              |          35.12 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/swats)              |
|                         30 | ranger25           |          37    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ranger25)           |
|                         31 | grokfastadamw      |          38.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/grokfastadamw)      |
|                         32 | laprop             |          38.62 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/laprop)             |
|                         33 | emoneco            |          38.81 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emoneco)            |
|                         34 | yogi               |          38.94 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/yogi)               |
|                         35 | exadam             |          39.62 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/exadam)             |
|                         36 | adamax             |          40.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamax)             |
|                         37 | came               |          40.88 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/came)               |
|                         38 | nadam              |          40.94 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/nadam)              |
|                         39 | tam                |          41    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/tam)                |
|                         40 | adasmooth          |          41.25 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adasmooth)          |
|                         41 | adan               |          41.69 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adan)               |
|                         42 | adahessian         |          42.38 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adahessian)         |
|                         43 | adadelta           |          42.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adadelta)           |
|                         44 | schedulefreeadamw  |          42.69 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/schedulefreeadamw)  |
|                         45 | adapnm             |          42.94 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adapnm)             |
|                         46 | scionlight         |          43.12 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/scionlight)         |
|                         47 | sgd                |          43.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sgd)                |
|                         48 | sgdp               |          43.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sgdp)               |
|                         49 | lamb               |          44.62 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/lamb)               |
|                         50 | scion              |          45.38 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/scion)              |
|                         51 | diffgrad           |          45.5  | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/diffgrad)           |
|                         52 | nero               |          45.62 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/nero)               |
|                         53 | dadaptadan         |          46.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptadan)         |
|                         54 | asgd               |          46.62 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/asgd)               |
|                         55 | rmsprop            |          46.94 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/rmsprop)            |
|                         56 | vsgd               |          47.31 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/vsgd)               |
|                         57 | pid                |          48.38 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/pid)                |
|                         58 | adashift           |          48.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adashift)           |
|                         59 | adabound           |          49.19 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adabound)           |
|                         60 | padam              |          49.38 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/padam)              |
|                         61 | fadam              |          49.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/fadam)              |
|                         62 | prodigy            |          52.12 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/prodigy)            |
|                         63 | simplifiedademamix |          53.94 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/simplifiedademamix) |
|                         64 | lars               |          54.62 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/lars)               |
|                         65 | ranger21           |          55.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ranger21)           |
|                         66 | adopt              |          57.31 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adopt)              |
|                         67 | dadaptadagrad      |          57.69 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptadagrad)      |
|                         68 | apollodqn          |          57.81 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/apollodqn)          |
|                         69 | dadaptlion         |          58.19 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptlion)         |
|                         70 | ftrl               |          58.44 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ftrl)               |
|                         71 | sm3                |          58.69 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sm3)                |
|                         72 | scalableshampoo    |          59.44 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/scalableshampoo)    |
|                         73 | shampoo            |          63.62 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/shampoo)            |
|                         74 | aggmo              |          63.94 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/aggmo)              |
|                         75 | schedulefreesgd    |          64    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/schedulefreesgd)    |
|                         76 | accsgd             |          64.31 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/accsgd)             |
|                         77 | kate               |          68.5  | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/kate)               |
|                         78 | grams              |          69.19 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/grams)              |
|                         79 | adai               |          70.44 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adai)               |
|                         80 | madgrad            |          71.94 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/madgrad)            |
|                         81 | qhm                |          72    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/qhm)                |
|                         82 | srmm               |          72.5  | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/srmm)               |
|                         83 | racs               |          74.25 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/racs)               |
|                         84 | adamg              |          74.38 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamg)              |
|                         85 | gravity            |          75.12 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/gravity)            |
|                         86 | ranger             |          75.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ranger)             |
|                         87 | schedulefreeradam  |          75.75 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/schedulefreeradam)  |
|                         88 | pnm                |          76.62 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/pnm)                |
|                         89 | dadaptsgd          |          79    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptsgd)          |
|                         90 | avagrad            |          79.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/avagrad)            |
|                         91 | sgdsai             |          79.69 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sgdsai)             |
|                         92 | radam              |          81.19 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/radam)              |
|                         93 | qhadam             |          85.88 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/qhadam)             |
|                         94 | adamod             |          86.56 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamod)             |
|                         95 | adafactor          |          91    | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adafactor)          |
|                         96 | adams              |          91.06 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adams)              |
|                         97 | mars               |          93.44 | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/mars)               |

</h6>
</details>
</h4>

<h4>
<details>
<summary>Ranking by Error Rate ⚡</summary>
<h6>

|   Rank (Error Rate) | Optimizer          | Avg Error Rate   | Vis                                                                              |
|---------------------|--------------------|------------------|----------------------------------------------------------------------------------|
|                   1 | adatam             | 3.39             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adatam)             |
|                   2 | emonavi            | 3.46             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emonavi)            |
|                   3 | emofact            | 3.48             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emofact)            |
|                   4 | emozeal            | 3.48             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emozeal)            |
|                   5 | tiger              | 4.00             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/tiger)              |
|                   6 | signsgd            | 4.29             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/signsgd)            |
|                   7 | sophiah            | 4.72             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sophiah)            |
|                   8 | emolynx            | 5.14             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emolynx)            |
|                   9 | lion               | 5.14             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/lion)               |
|                  10 | focus              | 5.31             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/focus)              |
|                  11 | kron               | 6.03             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/kron)               |
|                  12 | adabelief          | 6.47             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adabelief)          |
|                  13 | novograd           | 6.55             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/novograd)           |
|                  14 | exadam             | 6.75             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/exadam)             |
|                  15 | fromage            | 7.10             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/fromage)            |
|                  16 | lamb               | 7.53             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/lamb)               |
|                  17 | ademamix           | 7.95             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ademamix)           |
|                  18 | adammini           | 8.13             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adammini)           |
|                  19 | adadelta           | 8.25             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adadelta)           |
|                  20 | adagc              | 8.41             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adagc)              |
|                  21 | apollo             | 8.45             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/apollo)             |
|                  22 | fira               | 8.45             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/fira)               |
|                  23 | galore             | 8.45             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/galore)             |
|                  24 | adapnm             | 8.56             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adapnm)             |
|                  25 | scionlight         | 8.65             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/scionlight)         |
|                  26 | adalite            | 8.71             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adalite)            |
|                  27 | emoneco            | 8.77             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emoneco)            |
|                  28 | stablespam         | 8.78             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/stablespam)         |
|                  29 | adam               | 8.89             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adam)               |
|                  30 | soap               | 8.91             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/soap)               |
|                  31 | nadam              | 9.24             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/nadam)              |
|                  32 | adamp              | 9.59             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamp)              |
|                  33 | adanorm            | 9.72             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adanorm)            |
|                  34 | rmsprop            | 9.74             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/rmsprop)            |
|                  35 | aida               | 9.78             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/aida)               |
|                  36 | nero               | 9.81             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/nero)               |
|                  37 | diffgrad           | 9.82             | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/diffgrad)           |
|                  38 | laprop             | 10.03            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/laprop)             |
|                  39 | spam               | 10.04            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/spam)               |
|                  40 | adasmooth          | 10.16            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adasmooth)          |
|                  41 | tam                | 10.40            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/tam)                |
|                  42 | swats              | 10.56            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/swats)              |
|                  43 | asgd               | 10.82            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/asgd)               |
|                  44 | came               | 10.99            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/came)               |
|                  45 | adahessian         | 11.11            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adahessian)         |
|                  46 | schedulefreeadamw  | 12.11            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/schedulefreeadamw)  |
|                  47 | adashift           | 12.75            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adashift)           |
|                  48 | adamax             | 12.87            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamax)             |
|                  49 | yogi               | 12.94            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/yogi)               |
|                  50 | vsgd               | 12.95            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/vsgd)               |
|                  51 | adan               | 13.01            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adan)               |
|                  52 | schedulefreesgd    | 13.06            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/schedulefreesgd)    |
|                  53 | fadam              | 13.07            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/fadam)              |
|                  54 | adopt              | 13.47            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adopt)              |
|                  55 | padam              | 14.02            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/padam)              |
|                  56 | dadaptadan         | 14.46            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptadan)         |
|                  57 | grokfastadamw      | 15.04            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/grokfastadamw)      |
|                  58 | dadaptadagrad      | 15.20            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptadagrad)      |
|                  59 | simplifiedademamix | 15.72            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/simplifiedademamix) |
|                  60 | ranger21           | 16.02            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ranger21)           |
|                  61 | scion              | 16.12            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/scion)              |
|                  62 | sgd                | 16.70            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sgd)                |
|                  63 | sgdp               | 16.70            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sgdp)               |
|                  64 | ftrl               | 17.64            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ftrl)               |
|                  65 | sm3                | 17.75            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sm3)                |
|                  66 | adabound           | 18.79            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adabound)           |
|                  67 | pid                | 18.96            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/pid)                |
|                  68 | kate               | 19.69            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/kate)               |
|                  69 | apollodqn          | 19.78            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/apollodqn)          |
|                  70 | dadaptlion         | 20.25            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptlion)         |
|                  71 | lars               | 20.31            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/lars)               |
|                  72 | prodigy            | 20.66            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/prodigy)            |
|                  73 | grams              | 20.80            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/grams)              |
|                  74 | scalableshampoo    | 21.12            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/scalableshampoo)    |
|                  75 | shampoo            | 21.68            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/shampoo)            |
|                  76 | accsgd             | 23.13            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/accsgd)             |
|                  77 | qhm                | 23.53            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/qhm)                |
|                  78 | ranger             | 23.63            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ranger)             |
|                  79 | racs               | 23.93            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/racs)               |
|                  80 | adai               | 23.99            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adai)               |
|                  81 | srmm               | 24.27            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/srmm)               |
|                  82 | aggmo              | 24.28            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/aggmo)              |
|                  83 | pnm                | 25.16            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/pnm)                |
|                  84 | madgrad            | 26.02            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/madgrad)            |
|                  85 | schedulefreeradam  | 26.42            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/schedulefreeradam)  |
|                  86 | adamg              | 26.50            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamg)              |
|                  87 | gravity            | 27.19            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/gravity)            |
|                  88 | radam              | 27.33            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/radam)              |
|                  89 | avagrad            | 27.36            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/avagrad)            |
|                  90 | sgdsai             | 28.26            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sgdsai)             |
|                  91 | dadaptsgd          | 28.48            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptsgd)          |
|                  92 | adamod             | 29.26            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamod)             |
|                  93 | adafactor          | 32.33            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adafactor)          |
|                  94 | mars               | 38.10            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/mars)               |
|                  95 | adams              | 56.91            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adams)              |
|                  96 | qhadam             | 386.11           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/qhadam)             |
|                  97 | ranger25           | Failed ⚠️        | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ranger25)           |

</h6>
</details>
</h4>

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
