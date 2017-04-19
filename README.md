# ReCA
Source code for my semester project and master thesis. Reservoir Computing using Cellular Automata (ReCA).
Using the Japanese vowels data set: https://archive.ics.uci.edu/ml/datasets/Japanese+Vowels.

Keywords: Reservoir computing, cellular automata, deep learning, classification, regression.

# Results

Final results are placed in results/.
The files starts with the time they started, followed by the problem, and the specifications.
E.g. "2016-11-16T23:43:14-bitmem2res-i4-r4-rule-102": N-bit-memory task, 2 reservoirs, 4 iterations, 4 random mappings, rule 102.

# Applications

Note: The directory reca/tmp/ may have to be created before run-time.

## Arguments

Example: `python bittask.py -I 32 -R 16 -r 110 --pad 10`
* `-I` iterations
* `-R` the number of random mappings
* `-r` rule [0,255]
* `--diffuse` diffuse, it spreads the mapping to over a larger area
* `--pad` a number >= `size` if you want padding on your automata

You can omit diffuse and automaton-area.

## bittask.py

The 5-bit memory task with 2 reservoirs. This really handles 1 reservoir as well, it logs the output and results of that one too.

## parity.py

Non-temporal parity problem

## density.py

Non-temporal density classification task

# Dependencies

* scikit-learn
* numpy
* scipy
* matplotlib

Developed on Linux. For windows, see branch `windows`.

# A sort of a log

* `2017-01-31T21:07` Made the repo public. Great day for folk and fedreland.
* `2016-11-18T14:30` Sort of checkpoint of code and time.

