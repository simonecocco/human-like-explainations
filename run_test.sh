#!/bin/zsh

pip3 install -e .

python3 pathlm/models/lm/patta_main.py --eval "U219 R-1"