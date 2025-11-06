#!/bin/bash
# Скрипт для запуска на суперкомпьютере "Политехник - РСК Торнадо" через SLURM
# Использование: sbatch run_supercomputer.sh

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado
#SBATCH -t 00:30:00
#SBATCH -J random_walk
#SBATCH -o random_walk-%j.out
#SBATCH -e random_walk-%j.err

# Загрузка модулей окружения
if [ -f /etc/profile.d/modules-basis.sh ]; then
    source /etc/profile.d/modules-basis.sh
fi

module purge
module load python/3.5.2

# Активация окружения (если используется)
# conda activate random_walk
# или
# source venv/bin/activate

# Запуск скрипта моделирования (указываем платформу как 'supercomputer')
# Используем одинаковые параметры и seed для сравнения с локальной машиной
# Формат: python random_walk.py N M platform seed
echo "Начало выполнения моделирования: $(date)"
python random_walk.py 10000 100000000 supercomputer 42
echo "Завершение моделирования: $(date)"

# Создание графиков (если visualize.py доступен)
if [ -f visualize.py ]; then
    echo "Начало создания графиков: $(date)"
    python visualize.py
    echo "Завершение создания графиков: $(date)"
else
    echo "Файл visualize.py не найден, графики не созданы"
fi

# Коммит результатов в git (если репозиторий настроен)
if [ -d .git ]; then
    echo "Коммит результатов в git..."
    # Добавляем результаты (теперь они не исключены из .gitignore)
    git add results/results_N*.json results/timing_N*.json results/*.png results/positions_sample_*.npy 2>/dev/null || true
    # Коммитим только если есть изменения
    if ! git diff --staged --quiet; then
        git commit -m "Results from supercomputer run $(date '+%Y-%m-%d %H:%M:%S')" || true
        # Пытаемся запушить (может потребоваться настройка credentials)
        git push || echo "Не удалось запушить в git. Сделайте это вручную: git push"
    else
        echo "Нет новых результатов для коммита"
    fi
else
    echo "Git репозиторий не найден, результаты не закоммичены"
fi

