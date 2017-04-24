#PBS -N andrmolutestjob
#PBS -A andrmolu
#PBS -l select=2:ncpus=1:mpiprocs=1:mem=100mb
#PBS -l walltime=0:01:00

module load python/2.7.3
python /home/andrmolu/supercomp/testpyscript.py