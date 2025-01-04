# Investigation of the Grokking Phenomenon

This repo contains the code for reproducing our results. The required supplementary.zip is also organized and presented here.

  
## Installation

```bash
pip install -e .
```

The python file train.py is for training and the grokking phenomenon.

## Training

```bash
./scripts/train.py
```

## Training with different model architecture

```bash
python scripts/train.py --train_data_pct 20 --model transformer
python scripts/train.py --train_data_pct 50 --model mlplarge
python scripts/train.py --train_data_pct 70 --model mlpmedium
python scripts/train.py --train_data_pct 80 --model mlpsmall
python scripts/train.py --train_data_pct 80 --model lstm
```

## Training with different settings

```bash
python scripts/train.py --optimizer adamw --max_steps 100000 --weight_decay 0.05 --train_data_pct 50
```

## Training on the harder problem with K operands

```bash
python scripts/train.py --K 3
```

## Generating sharpness from stored checkpoints

```bash
python scripts/compute_sharpness.py
```

