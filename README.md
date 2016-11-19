# semester-project-code
The code for my specialisation/semester project

# Results

Results are placed in project/results/.
The files starts with the time they started, followed by the problem, and the specifications.
E.g. "2016-11-16T23:43:14-bitmem2res-i4-r4-rule-102": N-bit-memory task, 2 reservoirs, 4 iterations, 4 random mappings, rule 102.

# Applications

## Arguments

Example: `python2 parity.py -s 6 -i 32 -r 154 --random-mappings 16 --automaton-area 10`
* `-i` iterations
* `-r` rule
* `-s` size
* `--random-mappings` the number of random mappings
* `--input-area` input area, it spreads the mapping to over a larger area
* `--automaton-area` a number >= `size` if you want padding on your automata

You can omit input-area and automaton-area.

## bitmemorytask2.py

The 5-bit memory task with 2 reservoirs. This really also handles 1 reservoir, it logs the output and results of that one too.

## bitmemorytask.py

The 5-bit memory task with 2 reservoirs.

## parity.py

Non-temporal parity problem

## density.py

Non-temporal density classification task

# Dependencies

* scikit-learn
* numpy
* scipy
* matplotlib

# A sort of a log

* 2016-11-18T14:30 Sort of checkpoint of code and time.

