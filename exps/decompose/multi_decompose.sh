#!/usr/bin/env bash

python decompose.py with alpha=1e-6 &
python decompose.py with alpha=5e-6 &
python decompose.py with alpha=1e-5 &
python decompose.py with alpha=5e-4 &
python decompose.py with alpha=1e-4 &