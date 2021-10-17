# Temp repo for apriori-rs

Benchmarks

| Min support, length | apriori-rs | [efficient-apriori](https://github.com/tommyod/Efficient-Apriori) | [mlxtend](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) | [apyori](https://github.com/ymoch/apyori) |
|:-------------------:|:----------:|:----------------:|:-------:|:----:|
|            0.100, 1 |       0.2s |             0.1s |    0.1s | 0.29s
|            0.100, 2 |       0.2s |             0.1s |    0.1s | 0.26s
|            0.100, 3 |       0.2s |             0.1s |    0.1s | 0.25s
|            0.100, 4 |       0.2s |             0.1s |    0.1s |0.25s
|            0.100, 5 |       0.2s |             0.1s |    0.1s | 0.25s
|            0.050, 1 |       0.2s |             0.1s |    0.1s | 0.25s
|            0.050, 2 |       0.2s |             0.2s |    0.1s |0.25s
|            0.050, 3 |       0.2s |             0.2s |    0.1s | 0.25s
|            0.050, 4 |       0.2s |             0.2s |    0.1s | 0.25s
|            0.050, 5 |       0.2s |             0.2s |    0.2s |0.25s
|            0.010, 1 |       0.2s |             0.1s |    0.1s | 0.32s
|            0.010, 2 | **1.6s** (prev 16s) |             261s |     73s | 2.1s
|            0.010, 3 | 10s (prev 15s) 😭 |             272s |     79s | **2.3s**
|            0.010, 4 | 15s (prev 17s) 😭 |             284s |     78s | **2.4s**
|            0.010, 5 |        14s |             279s |     92s | **2.4s**
|            0.005, 1 |       0.2s |             0.1s |    0.1s | 0.25s
|            0.005, 2 |        76s |            1190s |    327s | **5.7s**
|            0.005, 3 |        68s |            1278s |    643s | **20s**
|            0.005, 4 |        81s |            1168s |    638s | **39s**
|            0.005, 5 |        70s |            1217s |    643s | **41s**
