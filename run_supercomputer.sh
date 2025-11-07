#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado
#SBATCH -t 04:00:00
#SBATCH -J random_walk
#SBATCH -o random_walk-%j.out
#SBATCH -e random_walk-%j.err

if [ -f /etc/profile.d/modules-basis.sh ]; then
    source /etc/profile.d/modules-basis.sh
fi

module purge
module load python/3.9

python3 random_walk.py 10000 100000000 supercomputer 42
python3 visualize.py
