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
|                   1 | adatam             | 3.3906           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adatam)             |
|                   2 | emonavi            | 3.4562           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emonavi)            |
|                   3 | emofact            | 3.4753           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emofact)            |
|                   4 | emozeal            | 3.4753           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emozeal)            |
|                   5 | tiger              | 3.9969           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/tiger)              |
|                   6 | signsgd            | 4.2924           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/signsgd)            |
|                   7 | sophiah            | 4.723            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sophiah)            |
|                   8 | emolynx            | 5.1373           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emolynx)            |
|                   9 | lion               | 5.1373           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/lion)               |
|                  10 | focus              | 5.3087           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/focus)              |
|                  11 | kron               | 6.0274           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/kron)               |
|                  12 | adabelief          | 6.4662           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adabelief)          |
|                  13 | novograd           | 6.5473           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/novograd)           |
|                  14 | exadam             | 6.7497           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/exadam)             |
|                  15 | fromage            | 7.0991           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/fromage)            |
|                  16 | lamb               | 7.5312           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/lamb)               |
|                  17 | ademamix           | 7.948            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ademamix)           |
|                  18 | adammini           | 8.1328           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adammini)           |
|                  19 | adadelta           | 8.2486           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adadelta)           |
|                  20 | adagc              | 8.4104           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adagc)              |
|                  21 | apollo             | 8.4469           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/apollo)             |
|                  22 | fira               | 8.4469           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/fira)               |
|                  23 | galore             | 8.4469           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/galore)             |
|                  24 | adapnm             | 8.5628           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adapnm)             |
|                  25 | scionlight         | 8.6491           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/scionlight)         |
|                  26 | adalite            | 8.7071           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adalite)            |
|                  27 | emoneco            | 8.7699           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/emoneco)            |
|                  28 | stablespam         | 8.7751           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/stablespam)         |
|                  29 | adam               | 8.8905           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adam)               |
|                  30 | soap               | 8.9126           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/soap)               |
|                  31 | nadam              | 9.2417           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/nadam)              |
|                  32 | adamp              | 9.5865           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamp)              |
|                  33 | adanorm            | 9.7223           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adanorm)            |
|                  34 | rmsprop            | 9.7358           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/rmsprop)            |
|                  35 | aida               | 9.7844           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/aida)               |
|                  36 | nero               | 9.8053           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/nero)               |
|                  37 | diffgrad           | 9.8235           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/diffgrad)           |
|                  38 | laprop             | 10.0266          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/laprop)             |
|                  39 | spam               | 10.0438          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/spam)               |
|                  40 | adasmooth          | 10.158           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adasmooth)          |
|                  41 | tam                | 10.4039          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/tam)                |
|                  42 | swats              | 10.5598          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/swats)              |
|                  43 | asgd               | 10.8174          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/asgd)               |
|                  44 | came               | 10.9854          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/came)               |
|                  45 | adahessian         | 11.1066          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adahessian)         |
|                  46 | schedulefreeadamw  | 12.1115          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/schedulefreeadamw)  |
|                  47 | adashift           | 12.7504          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adashift)           |
|                  48 | adamax             | 12.8721          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamax)             |
|                  49 | yogi               | 12.9361          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/yogi)               |
|                  50 | vsgd               | 12.9511          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/vsgd)               |
|                  51 | adan               | 13.0089          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adan)               |
|                  52 | schedulefreesgd    | 13.0602          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/schedulefreesgd)    |
|                  53 | fadam              | 13.0729          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/fadam)              |
|                  54 | adopt              | 13.4732          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adopt)              |
|                  55 | padam              | 14.0248          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/padam)              |
|                  56 | dadaptadan         | 14.4575          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptadan)         |
|                  57 | grokfastadamw      | 15.0406          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/grokfastadamw)      |
|                  58 | dadaptadagrad      | 15.2034          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptadagrad)      |
|                  59 | simplifiedademamix | 15.7192          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/simplifiedademamix) |
|                  60 | ranger21           | 16.0177          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ranger21)           |
|                  61 | scion              | 16.119           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/scion)              |
|                  62 | sgd                | 16.6986          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sgd)                |
|                  63 | sgdp               | 16.6986          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sgdp)               |
|                  64 | ftrl               | 17.6352          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ftrl)               |
|                  65 | sm3                | 17.7487          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sm3)                |
|                  66 | adabound           | 18.7915          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adabound)           |
|                  67 | pid                | 18.9555          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/pid)                |
|                  68 | kate               | 19.6937          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/kate)               |
|                  69 | apollodqn          | 19.7824          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/apollodqn)          |
|                  70 | dadaptlion         | 20.2519          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptlion)         |
|                  71 | lars               | 20.3135          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/lars)               |
|                  72 | prodigy            | 20.6609          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/prodigy)            |
|                  73 | grams              | 20.8008          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/grams)              |
|                  74 | scalableshampoo    | 21.1188          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/scalableshampoo)    |
|                  75 | shampoo            | 21.6842          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/shampoo)            |
|                  76 | accsgd             | 23.1303          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/accsgd)             |
|                  77 | qhm                | 23.5284          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/qhm)                |
|                  78 | ranger             | 23.6326          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/ranger)             |
|                  79 | racs               | 23.9282          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/racs)               |
|                  80 | adai               | 23.9893          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adai)               |
|                  81 | srmm               | 24.2722          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/srmm)               |
|                  82 | aggmo              | 24.276           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/aggmo)              |
|                  83 | pnm                | 25.1595          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/pnm)                |
|                  84 | madgrad            | 26.0235          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/madgrad)            |
|                  85 | schedulefreeradam  | 26.418           | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/schedulefreeradam)  |
|                  86 | adamg              | 26.5044          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamg)              |
|                  87 | gravity            | 27.1906          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/gravity)            |
|                  88 | radam              | 27.3285          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/radam)              |
|                  89 | avagrad            | 27.36            | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/avagrad)            |
|                  90 | sgdsai             | 28.2559          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/sgdsai)             |
|                  91 | dadaptsgd          | 28.4846          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/dadaptsgd)          |
|                  92 | adamod             | 29.2631          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adamod)             |
|                  93 | adafactor          | 32.3336          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adafactor)          |
|                  94 | mars               | 38.0967          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/mars)               |
|                  95 | adams              | 56.9071          | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/adams)              |
|                  96 | qhadam             | 386.1081         | [Open](https://aidinhamedi.github.io/Optimizer-Benchmark/vis/qhadam)             |
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
