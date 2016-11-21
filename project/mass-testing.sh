#!/bin/bash

# A script for bulk/mass testing tasks

# Rules to test : 
# After Yilmaz: 90 150 182 22 30 126 110 | 54 62 60 32 160 4 108 218 250
# After Bye:    102 105 153 165 180 195 (excluding those from Yilmaz)

# Parameters to test (at least):
# I: 2, 4
# R: 4, 8
# Yilmaz tested (I,R): (4,4), (4,16), (16,4) ...

python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 62
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 60
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 32
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 160
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 4
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 108
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 218
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 250
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 102
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 105
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 153
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 165
python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 180
#python2 bitmemorytask2.py --input-area 40 --random-mappings 4 -i 4 -r 195
