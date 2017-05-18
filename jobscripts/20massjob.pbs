#!/bin/bash
#PBS -N deepreca20bit
#PBS -A andrmolu
#PBS -l select=1:ncpus=4:mem=35gb
#PBS -l walltime=50:00:00
#PBS -o /home/andrmolu/reca/jobscripts/pbs
#PBS -e /home/andrmolu/reca/jobscripts/pbs
#PBS -J 1-4

cd /home/andrmolu/reca/
module load python

python 20bittask.py --id $PBS_ARRAY_INDEX -r '62,62,62' -I '16,16,16' -R '100,100,100'
python 20bittask.py --id $PBS_ARRAY_INDEX -r '54,54,54' -I '16,16,16' -R '100,100,100'
