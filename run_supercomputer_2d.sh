#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado
#SBATCH -t 04:00:00
#SBATCH -J random_walk_2d
#SBATCH -o random_walk_2d-%j.out
#SBATCH -e random_walk_2d-%j.err

if [ -f /etc/profile.d/modules-basis.sh ]; then
    source /etc/profile.d/modules-basis.sh
fi

module purge
module load python/3.9


python3 random_walk_2d.py 10000 100000000 42 200000

# Альтернативный вариант с автоматическим расчетом batch_size:
# python3 random_walk_2d.py 10000 100000000 42

