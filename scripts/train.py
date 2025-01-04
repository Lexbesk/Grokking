#!/usr/bin/env python
# python scripts/train.py --gpu 0 --optimizer adamw --max_steps 150000 --weight_decay 0.05 --train_data_pct 50

import grok
import os 

parser = grok.training.add_args()
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)
print(grok.training.train(hparams))
