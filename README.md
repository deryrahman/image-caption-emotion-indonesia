# 13515097 StyleNet

Intended for final project

## Setup

```
$ pip install -r requirements
```

## Experiment

```
    -- loss: negative log-likelihood
    -- optimizer: Adam
```

#### Hyperparameter
```
    -- batch_size (factual): 64
    -- batch_size (emotion): 96
    -- epoch (factual): 20, 100
    -- epoch (emotion): 25, 100
    -- lstm_layers: 1, 2, 3
    -- lstm_units: 512
    -- factored_size: 256, 512, 1024
    -- embedding_size: 300
    -- optimizer: Adam
    -- learning_rate (factual): 0.0002
    -- learning_rate (emotion): 0.0005
    -- β1: 0.9
    -- β2: 0.999
    -- ε: 1e-08
```

#### GridSearch

| No | Epoch (factual) | Epoch (emotion) | LSTM layers | Factored size |
|----|-----------------|-----------------|-------------|---------------|
| 1  | 20              | 25              | 1           | 256           |
| 2  | 20              | 25              | 1           | 512           |
| 3  | 20              | 25              | 1           | 1024          |
| 4  | 20              | 25              | 2           | 256           |
| 5  | 20              | 25              | 2           | 512           |
| 6  | 20              | 25              | 2           | 1024          |
| 7  | 20              | 25              | 3           | 256           |
| 8  | 20              | 25              | 3           | 512           |
| 9  | 20              | 25              | 3           | 1024          |
| 10 | 100             | 100             | 1           | 256           |
| 11 | 100             | 100             | 1           | 512           |
| 12 | 100             | 100             | 1           | 1024          |
| 13 | 100             | 100             | 2           | 256           |
| 14 | 100             | 100             | 2           | 512           |
| 15 | 100             | 100             | 2           | 1024          |
| 16 | 100             | 100             | 3           | 256           |
| 17 | 100             | 100             | 3           | 512           |
| 18 | 100             | 100             | 3           | 1024          |

#### Result

| No | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|----|--------|--------|--------|--------|
| 1  |        |        |        |        |
| 2  |        |        |        |        |
| 3  |        |        |        |        |
| 4  |        |        |        |        |
| 5  |        |        |        |        |
| 6  |        |        |        |        |
| 7  |        |        |        |        |
| 8  |        |        |        |        |
| 9  |        |        |        |        |
| 10 |        |        |        |        |
| 11 |        |        |        |        |
| 12 |        |        |        |        |
| 13 |        |        |        |        |
| 14 |        |        |        |        |
| 15 |        |        |        |        |
| 16 |        |        |        |        |
| 17 |        |        |        |        |
| 18 |        |        |        |        |