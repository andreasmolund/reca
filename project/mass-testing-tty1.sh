#!/bin/bash

# A script for bulk/mass testing tasks

# Rules to test : 
# After Yilmaz: 90 150 182 22 30 126 110 54 | 62 60 32 160 4 108 218 250
# After Bye:    102 105 153 165 180 195 (excluding those from Yilmaz)

# Parameters to test (at least):
# I: 2, 4
# R: 4, 8
# Yilmaz tested (I,R): (4,4), (4,16), (16,4) ...

#python2 bitmemorytask2.py --input-area 40 --random-mappings 16 -i 8 -r 90
#python2 bitmemorytask2.py --input-area 40 --random-mappings 16 -i 8 -r 150
#python2 bitmemorytask2.py --input-area 40 --random-mappings 16 -i 8 -r 60
python2 bitmemorytask2.py --input-area 40 --random-mappings 16 -i 8 -r 102
#python2 bitmemorytask2.py --input-area 40 --random-mappings 16 -i 8 -r 105
#python2 bitmemorytask2.py --input-area 40 --random-mappings 16 -i 8 -r 153
