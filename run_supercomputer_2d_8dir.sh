#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado
#SBATCH -t 12:00:00  # 12 часов (для M=100M нужно больше времени)
#SBATCH -J random_walk_2d_8dir
#SBATCH -o random_walk_2d_8dir-%j.out
#SBATCH -e random_walk_2d_8dir-%j.err

if [ -f /etc/profile.d/modules-basis.sh ]; then
    source /etc/profile.d/modules-basis.sh
fi

module purge
module load python/3.9

# Используем -u для отключения буферизации вывода (чтобы видеть вывод в реальном времени)
# Параметры: N=10000, M=100000000, seed=42, batch_size=200000
python3 -u random_walk_2d_8dir.py 10000 100000000 42 200000

