#!/usr/bin/env bash

python decompose_hcp.py with n_components=16 &
python decompose_hcp.py with n_components=64 &
python decompose_hcp.py with n_components=512 &