#!/bin/bash

# A script for bulk/mass testing tasks

# Rules to test : 
# After Yilmaz: 90 150 182 22 30 126 110 | 54 62 60 32 160 4 108 218 250
# After Bye:    102 105 153 165 180 195 (excluding those from Yilmaz)

# Parameters to test (at least):
# I: 2, 4
# R: 4, 8
# Yilmaz tested (I,R): (4,4), (4,16), (16,4) ...

python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 90
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 150
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 182
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 22
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 30
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 126
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 110
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 54
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 62
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 60
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 32
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 160
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 4
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 108
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 218
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 2 -r 250
