# 13515097 StyleNet

Intended for final project

## Setup

```
$ pip install -r requirements
```

## Dataset pre setup

-- Invoke new dataset from imagecaption.geekstudio.id dump
```
usage: invoke_new_dataset.py [-h] [--mongo_dump_path MONGO_DUMP_PATH]
                             [--dataset DATASET] [--dataset_path DATASET_PATH]
```

-- Split dataset into factual, happy, sad, and angry
```
usage: split_dataset.py [-h] [--dataset_path DATASET_PATH]
                        [--val_percentage VAL_PERCENTAGE]
                        [--test_percentage TEST_PERCENTAGE]
```

## Experiment pre setup

-- Create image transfer value from Resnet152
```
usage: create_transfer_values.py [-h] [--dataset_path DATASET_PATH]
                                 [--batch_size BATCH_SIZE]
```
-- Create tokenizer from current dataset
```
usage: create_tokenizer.py [-h] [--dataset_path DATASET_PATH]
                           [--num_words NUM_WORDS]
```

## Experiment

```
    -- loss: negative log-likelihood
    -- optimizer: Adam
```
-- Train StyleNet
```
usage: run_stylenet.py [-h] [--emotion_training_mode EMOTION_TRAINING_MODE]
                       [--beam_search BEAM_SEARCH] [--load_model LOAD_MODEL]
                       [--dataset DATASET] [--dataset_path DATASET_PATH]
                       [--checkpoints_path CHECKPOINTS_PATH]
                       [--logs_path LOGS_PATH] [--mode MODE]
                       [--injection_mode INJECTION_MODE]
                       [--with_transfer_value WITH_TRANSFER_VALUE]
                       [--gpu_frac GPU_FRAC] [--epoch_num EPOCH_NUM]
                       [--batch_size BATCH_SIZE] [--early_stop EARLY_STOP]
                       [--lstm_layers LSTM_LAYERS] [--state_size STATE_SIZE]
                       [--embedding_size EMBEDDING_SIZE]
                       [--factored_size FACTORED_SIZE] [--dropout DROPOUT]
                       [--learning_rate LEARNING_RATE] [--beta_1 BETA_1]
                       [--beta_2 BETA_2] [--epsilon EPSILON]
```

#### Hyperparameter
```
    -- batch_size (factual): 64
    -- batch_size (emotion): 96
    -- epoch (factual): 100 (early stop, patience=10)
    -- epoch (emotion): 200 (early stop, parience=40)
    -- lstm_layers: 1, 2, 3
    -- lstm_units: 128, 256, 512
    -- factored_size: 128, 256, 512
    -- embedding_size: 100, 200, 300
    -- optimizer: Adam
    -- learning_rate (factual): 0.0002
    -- learning_rate (emotion): 0.0005
    -- β1: 0.9
    -- β2: 0.999
    -- ε: 1e-08
```

#### Pre Experimental

| No  | Epoch | Embedding | LSTM layer | State size | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| --- | ----- | --------- | ---------- | ---------- | ------ | ------ | ------ | ------ |
| 1   | 27    | 100       | 1          | 128        | 0.39   | 0.22   | 0.18   | 0.06   |
| 2   | 23    | 100       | 1          | 256        | 0.40   | 0.22   | 0.12   | 0.06   |
| 3   | 15    | 100       | 1          | 512        | 0.41   | 0.24   | 0.14   | 0.07   |
| 4   | 31    | 100       | 2          | 128        | 0.40   | 0.22   | 0.12   | 0.06   |
| 5   | 20    | 100       | 2          | 256        | 0.41   | 0.24   | 0.14   | 0.08   |
| 6   | 18    | 100       | 2          | 256        | 0.40   | 0.23   | 0.13   | 0.07   |
| 7   | 22    | 100       | 3          | 128        | 0.40   | 0.22   | 0.12   | 0.06   |
| 8   | 28    | 100       | 3          | 256        | 0.40   | 0.22   | 0.12   | 0.06   |
| 9   | 16    | 100       | 3          | 512        | 0.40   | 0.22   | 0.12   | 0.06   |
| 10  | 19    | 200       | 1          | 128        | 0.40   | 0.22   | 0.12   | 0.06   |
| 11  | 17    | 200       | 1          | 256        | 0.42   | 0.24   | 0.13   | 0.07   |
| 12  | 14    | 200       | 1          | 512        | 0.39   | 0.22   | 0.12   | 0.06   |
| 13  | 18    | 200       | 2          | 128        | 0.41   | 0.23   | 0.13   | 0.06   |
| 14  | 21    | 200       | 2          | 256        | 0.41   | 0.24   | 0.14   | 0.07   |
| 15  | 14    | 200       | 2          | 512        | 0.40   | 0.22   | 0.12   | 0.06   |
| 16  | 22    | 200       | 3          | 128        | 0.40   | 0.25   | 0.12   | 0.07   |
| 17  | 19    | 200       | 3          | 256        | 0.40   | 0.23   | 0.13   | 0.07   |
| 18  | 15    | 200       | 3          | 512        | 0.41   | 0.23   | 0.13   | 0.06   |
| 19  | 19    | 300       | 1          | 128        | 0.38   | 0.21   | 0.12   | 0.06   |
| 20  | 15    | 300       | 1          | 256        | 0.40   | 0.23   | 0.13   | 0.07   |
| 21  | 13    | 300       | 1          | 512        | 0.41   | 0.23   | 0.13   | 0.07   |
| 22  | 19    | 300       | 2          | 128        | 0.41   | 0.23   | 0.13   | 0.07   |
| 23  | 14    | 300       | 2          | 256        | 0.41   | 0.24   | 0.13   | 0.07   |
| 24  | 17    | 300       | 2          | 512        | 0.40   | 0.23   | 0.13   | 0.07   |
| 25  | 23    | 300       | 3          | 128        | 0.40   | 0.22   | 0.12   | 0.06   |
| 26  | 20    | 300       | 3          | 256        | 0.35   | 0.20   | 0.11   | 0.06   |
| 27  | 20    | 300       | 3          | 512        | 0.37   | 0.20   | 0.12   | 0.06   |


#### Experiment

```
-- NIC
    -- Factual
    -- Emotion
    -- Emotion Fine-Tuned
-- StyleNet
    -- Factual
    -- Emotion FactoredLSTM
    -- Emotion FactoredLSTM + Seq2Seq
    -- Emotion FactoredLSTM + Seq2Seq + Attention
```