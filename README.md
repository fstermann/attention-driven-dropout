# Attention Dropout

This repository contains code for the Attention Dropout paper.




## Results

For SimCSE settings we used the same hyperparameters as in the original paper.

|                      | Unsup. BERT | Unsup. RoBERTa |
| :------------------- | :---------: | :------------: |
| Batch size           |     64      |      512       |
| Learning rate (base) |    3e-5     |      1e-5      |


Our reproduced results for BERT
```
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 69.90 | 82.01 | 74.94 | 82.37 | 78.48 |    77.23     |      71.19      | 76.59 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

Our reproduced results for RoBERTa

...


### Bert Base Uncased

Best configuration:

- n_dropout: 1
- min_tokens: 10


|                            | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg.  |
| -------------------------- | ----- | ----- | ----- | ----- | ----- | ------------ | --------------- | ----- |
| Our results                | 69.90 | 82.01 | 74.94 | 82.37 | 78.48 | 77.23        | 71.19           | 76.59 |
| Diff to reproduced results | +3.58 | +0.16 | +1.56 | +2.11 | +0.46 | +1.59        | +1.38           | +1.55 |
| Diff to reported results   | +1.50 | -0.40 | +0.56 | +1.46 | -0.08 | +0.38        | -1.04           | +0.34 |




### Experiment tracking

We are using [Weights & Biases](https://www.wandb.com/) to track experiments.