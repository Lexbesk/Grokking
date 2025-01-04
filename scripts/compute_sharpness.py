#!/usr/bin/env python

import os
import grok

parser = grok.training.add_args()
parser.set_defaults(logdir=os.environ.get("LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)

# ckpts = [f"./ckpts/L-2_H-4_D-128_T-70_DROP-0_SD-{i}_WU-10_LR-1p0.ckpt" for i in range(20)]
ckpts = [f"./checkpoints/epoch_{i}.ckpt" for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]]
print(grok.training.compute_sharpness(hparams, ckpts))
