#!/bin/bash
#PBS -N deepreca
#PBS -A andrmolu
#PBS -l select=1:ncpus=4:mem=10gb
#PBS -l walltime=12:00:00
#PBS -o /home/andrmolu/reca/jobscripts/pbs
#PBS -e /home/andrmolu/reca/jobscripts/pbs
#PBS -J 1-4

cd /home/andrmolu/reca/
module load python
python japvow.py --id $PBS_ARRAY_INDEX -r 90 -I '20,20,20' -R '20,20,20'
python japvow.py --id $PBS_ARRAY_INDEX -r 110 -I '20,20,20' -R '20,20,20'
python japvow.py --id $PBS_ARRAY_INDEX -r 22 -I '20,20,20' -R '20,20,20'
python japvow.py --id $PBS_ARRAY_INDEX -r 41 -I '20,20,20' -R '20,20,20'
python japvow.py --id $PBS_ARRAY_INDEX -r 54 -I '20,20,20' -R '20,20,20'