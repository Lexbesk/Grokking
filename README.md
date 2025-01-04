# Investigation of the Grokking Phenomenon

  
## Installation

```bash
pip install -e .
```

## Training

```bash
./scripts/train.py
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

