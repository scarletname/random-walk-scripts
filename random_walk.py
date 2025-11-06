#!/usr/bin/env python3
"""
Скрипт для моделирования одномерного случайного блуждания
для выполнения на суперкомпьютере и локальной машине.
"""

import numpy as np
import time
import json
import sys
from pathlib import Path
from datetime import datetime


class TeeOutput:
    """Класс для вывода одновременно в консоль и файл."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
        self.start_time = time.time()
    
    def write(self, message):
        """Записывает сообщение в консоль и файл."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Сразу записываем в файл
    
    def flush(self):
        """Очищает буферы."""
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        """Закрывает файл."""
        if self.log_file:
            self.log_file.close()
    
    def get_elapsed_time(self):
        """Возвращает прошедшее время с начала."""
        return time.time() - self.start_time


def simulate_random_walk(N, M, platform='local', seed=None, batch_size=None, progress_log=None):
    """
    Моделирует M траекторий одномерного случайного блуждания по N шагов.
    Использует пакетную обработку для экономии памяти на локальных машинах.
    
    Параметры:
    ----------
    N : int
        Число шагов в каждой траектории
    M : int
        Число траекторий для моделирования
    platform : str
        Платформа выполнения ('local' или 'supercomputer')
    seed : int, optional
        Seed для генератора случайных чисел (для воспроизводимости)
    batch_size : int, optional
        Размер пакета для обработки (None = автоматический выбор)
    progress_log : list, optional
        Список для сохранения временных меток прогресса
    
    Возвращает:
    -----------
    dict : Словарь с результатами (средняя позиция, среднее квадратичное смещение, время выполнения)
    """
    print(f"Начало моделирования: N={N}, M={M}")
    print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Устанавливаем seed для воспроизводимости
    if seed is not None:
        np.random.seed(seed)
        print(f"Используется seed: {seed}")
    
    # Автоматический выбор размера пакета
    if batch_size is None:
        if platform == 'local':
            # Для локальной машины используем очень маленькие пакеты
            # Обрабатываем по 1000 траекторий за раз (требует ~80 МБ)
            # Это медленнее, но требует минимум памяти
            batch_size = min(1000, M)
        else:
            # На суперкомпьютере можно обрабатывать все сразу
            batch_size = M
    
    print(f"Размер пакета: {batch_size:,} траекторий")
    print(f"Ожидаемое использование памяти: ~{batch_size * N * 8 / 1024**3:.3f} ГБ на пакет")
    
    start_time = time.time()
    
    # Инициализируем накопители для статистики
    sum_positions = 0.0
    sum_squared_positions = 0.0
    sum_positions_for_std = []
    min_position = float('inf')
    max_position = float('-inf')
    
    # Обрабатываем траектории пакетами
    num_batches = (M + batch_size - 1) // batch_size  # Округление вверх
    
    print(f"Обработка {num_batches:,} пакетов...")
    print("Это может занять значительное время на локальной машине.")
    
    for batch_idx in range(num_batches):
        # Определяем границы текущего пакета
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, M)
        current_batch_size = end_idx - start_idx
        
        # Показываем прогресс каждые 1% или каждые 100 пакетов
        progress_interval = max(1, min(num_batches // 100, 100))
        if (batch_idx + 1) % progress_interval == 0 or batch_idx == num_batches - 1:
            progress = (batch_idx + 1) / num_batches * 100
            elapsed = time.time() - start_time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Сохраняем временную метку прогресса
            if progress_log is not None:
                progress_log.append({
                    'batch': batch_idx + 1,
                    'total_batches': num_batches,
                    'progress_percent': progress,
                    'elapsed_seconds': elapsed,
                    'elapsed_minutes': elapsed / 60,
                    'timestamp': current_time,
                    'trajectories_processed': end_idx
                })
            
            if batch_idx > 0:
                avg_time_per_batch = elapsed / (batch_idx + 1)
                remaining_batches = num_batches - (batch_idx + 1)
                eta_seconds = avg_time_per_batch * remaining_batches
                eta_minutes = eta_seconds / 60
                trajectories_per_sec = end_idx / elapsed if elapsed > 0 else 0
                print(f"  Прогресс: {progress:.1f}% ({batch_idx + 1:,}/{num_batches:,} пакетов) | "
                      f"Время: {elapsed/60:.1f} мин | ETA: {eta_minutes:.1f} мин | "
                      f"Скорость: {trajectories_per_sec:.0f} траекторий/сек | "
                      f"Время: {current_time}")
            else:
                print(f"  Прогресс: {progress:.1f}% ({batch_idx + 1:,}/{num_batches:,} пакетов) | "
                      f"Время: {current_time}")
        
        # Генерируем случайные шаги для текущего пакета
        # Форма массива: (current_batch_size, N)
        random_values = np.random.random((current_batch_size, N))
        # Преобразуем в шаги: >0.5 -> +1, <=0.5 -> -1
        steps = np.where(random_values > 0.5, 1, -1)
        
        # Вычисляем финальные позиции для траекторий в пакете (сумма шагов)
        batch_positions = np.sum(steps, axis=1)
        
        # Обновляем статистику
        sum_positions += np.sum(batch_positions)
        sum_squared_positions += np.sum(batch_positions ** 2)
        
        # Сохраняем выборку для вычисления стандартного отклонения
        # (сохраняем только часть для экономии памяти)
        if len(sum_positions_for_std) < 100000:  # Максимум 100k значений
            sample_size = min(100, len(batch_positions))  # Берем меньше из каждого пакета
            if sample_size > 0:
                indices = np.random.choice(len(batch_positions), sample_size, replace=False)
                sum_positions_for_std.extend(batch_positions[indices])
        
        # Обновляем min/max
        batch_min = np.min(batch_positions)
        batch_max = np.max(batch_positions)
        if batch_min < min_position:
            min_position = batch_min
        if batch_max > max_position:
            max_position = batch_max
        
        # Освобождаем память
        del steps, random_values, batch_positions
    
    # Вычисляем итоговую статистику
    mean_position = sum_positions / M
    mean_squared_displacement = sum_squared_positions / M
    
    # Вычисляем стандартное отклонение по выборке
    if sum_positions_for_std:
        std_position = np.std(sum_positions_for_std)
    else:
        std_position = np.sqrt(N)  # Теоретическое значение
    
    elapsed_time = time.time() - start_time
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Расчет производительности
    trajectories_per_second = M / elapsed_time if elapsed_time > 0 else 0
    
    # Создаем массив финальных позиций только для выборки (для визуализации)
    # Это будет использовано позже при сохранении выборки
    final_positions_sample = np.array(sum_positions_for_std[:10000]) if sum_positions_for_std else None
    
    results = {
        'N': N,
        'M': M,
        'platform': platform,
        'seed': seed,
        'batch_size': batch_size,
        'start_time': start_time,
        'end_time': time.time(),
        'start_time_str': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
        'end_time_str': end_time_str,
        'mean_position': float(mean_position),
        'mean_squared_displacement': float(mean_squared_displacement),
        'std_position': float(std_position),
        'min_position': int(min_position),
        'max_position': int(max_position),
        'elapsed_time_seconds': elapsed_time,
        'elapsed_time_minutes': elapsed_time / 60,
        'elapsed_time_hours': elapsed_time / 3600,
        'trajectories_per_second': trajectories_per_second,
        'theoretical_mean_position': 0.0,
        'theoretical_mean_squared_displacement': float(N),
        'error_mean_position': abs(mean_position),
        'relative_error_msd': abs(mean_squared_displacement - N) / N * 100
    }
    
    print(f"\nРезультаты моделирования:")
    print(f"  Время окончания: {end_time_str}")
    print(f"  Средняя позиция: {mean_position:.6f} (теоретическая: 0.0)")
    print(f"  Среднее квадратичное смещение: {mean_squared_displacement:.2f} (теоретическое: {N})")
    print(f"  Стандартное отклонение: {std_position:.2f}")
    print(f"  Минимальная позиция: {min_position}")
    print(f"  Максимальная позиция: {max_position}")
    print(f"  Время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут, {elapsed_time/3600:.2f} часов)")
    print(f"  Производительность: {trajectories_per_second:.0f} траекторий/сек")
    print(f"  Относительная ошибка MSD: {results['relative_error_msd']:.4f}%")
    
    return results, final_positions_sample


def save_results(results, output_dir='results', progress_log=None):
    """
    Сохраняет результаты в JSON файл.
    
    Параметры:
    ----------
    results : dict
        Словарь с результатами
    output_dir : str
        Директория для сохранения результатов
    progress_log : list, optional
        Лог прогресса для сохранения
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    output_file = Path(output_dir) / f"results_N{results['N']}_M{results['M']}.json"
    
    # Добавляем лог прогресса в результаты
    if progress_log is not None:
        results['progress_log'] = progress_log
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в: {output_file}")
    
    # Сохраняем отдельный файл с детализацией времени выполнения
    if progress_log:
        timing_file = Path(output_dir) / f"timing_N{results['N']}_M{results['M']}.json"
        timing_data = {
            'N': results['N'],
            'M': results['M'],
            'platform': results['platform'],
            'total_time_seconds': results['elapsed_time_seconds'],
            'total_time_minutes': results['elapsed_time_minutes'],
            'total_time_hours': results['elapsed_time_hours'],
            'start_time': results.get('start_time_str', ''),
            'end_time': results.get('end_time_str', ''),
            'trajectories_per_second': results.get('trajectories_per_second', 0),
            'progress_log': progress_log
        }
        with open(timing_file, 'w', encoding='utf-8') as f:
            json.dump(timing_data, f, indent=4, ensure_ascii=False)
        print(f"Детализация времени сохранена в: {timing_file}")
    
    return output_file


def save_sample_positions(final_positions, N, M, output_dir='results', sample_size=10000):
    """
    Сохраняет выборку финальных позиций для визуализации.
    
    Параметры:
    ----------
    final_positions : ndarray
        Массив финальных позиций
    N : int
        Число шагов
    M : int
        Число траекторий
    output_dir : str
        Директория для сохранения
    sample_size : int
        Размер выборки для сохранения (слишком много данных для визуализации)
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    if len(final_positions) > sample_size:
        # Берем случайную выборку
        indices = np.random.choice(len(final_positions), sample_size, replace=False)
        sample = final_positions[indices]
    else:
        sample = final_positions
    
    output_file = Path(output_dir) / f"positions_sample_N{N}_M{M}.npy"
    np.save(output_file, sample)
    
    print(f"Выборка позиций сохранена в: {output_file} (размер выборки: {len(sample)})")
    return output_file


def main():
    """Основная функция."""
    # Параметры из задания
    N = 10**4  # Число шагов
    M = 10**8  # Число траекторий
    platform = 'local'  # Платформа по умолчанию
    seed = None  # Seed для воспроизводимости (None = случайный)
    batch_size = None  # Размер пакета (None = автоматический)
    
    # Позволяем переопределить параметры через аргументы командной строки
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        M = int(sys.argv[2])
    if len(sys.argv) > 3:
        platform = sys.argv[3]  # 'local' или 'supercomputer'
    if len(sys.argv) > 4:
        seed = int(sys.argv[4])  # Seed для воспроизводимости
    if len(sys.argv) > 5:
        batch_size = int(sys.argv[5])  # Размер пакета
    
    print("=" * 60)
    print("МОДЕЛИРОВАНИЕ СЛУЧАЙНОГО БЛУЖДАНИЯ")
    print("=" * 60)
    print(f"Параметры:")
    print(f"  N (шагов): {N:,}")
    print(f"  M (траекторий): {M:,}")
    print(f"  Платформа: {platform}")
    if seed is not None:
        print(f"  Seed: {seed}")
    print("=" * 60)
    
    # Создаем файл логирования
    Path('results').mkdir(exist_ok=True)
    log_filename = f"results/log_N{N}_M{M}_{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = TeeOutput(log_filename)
    sys.stdout = tee
    
    try:
        print(f"Логирование вывода в файл: {log_filename}")
        print("=" * 60)
        
        # Проверка доступной памяти (примерная)
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / 1024**3
            actual_batch_size = batch_size if batch_size else (min(1000, M) if platform == 'local' else M)
            required_memory = actual_batch_size * N * 8 / 1024**3  # Примерная оценка
            print(f"Доступная память: {available_memory:.2f} ГБ")
            print(f"Ожидаемое использование на пакет: ~{required_memory:.3f} ГБ")
            if required_memory > available_memory * 0.8:
                print("ПРЕДУПРЕЖДЕНИЕ: Возможно недостаточно памяти!")
                print("Скрипт автоматически использует меньший размер пакета.")
                # Автоматически уменьшаем размер пакета, если нужно
                if batch_size is None and platform == 'local':
                    # Вычисляем максимальный размер пакета на основе доступной памяти
                    max_batch_size = int(available_memory * 0.6 * 1024**3 / (N * 8))
                    batch_size = max(100, min(1000, max_batch_size))  # От 100 до 1000
                    print(f"Автоматически установлен размер пакета: {batch_size:,} траекторий")
        except ImportError:
            pass
        
        # Инициализируем лог прогресса
        progress_log = []
        
        # Выполняем моделирование
        results, final_positions_sample = simulate_random_walk(N, M, platform, seed, batch_size, progress_log)
        
        # Сохраняем результаты
        save_results(results, progress_log=progress_log)
        
        # Сохраняем выборку позиций для визуализации (если есть)
        if final_positions_sample is not None:
            save_sample_positions(final_positions_sample, N, M)
        else:
            print("Выборка позиций недоступна (слишком мало данных)")
        
        print(f"\nПлатформа: {platform}")
        print("\nМоделирование завершено успешно!")
        print(f"Полный лог сохранен в: {log_filename}")
        
    finally:
        # Восстанавливаем стандартный вывод
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nЛогирование завершено. Лог сохранен в: {log_filename}")
    
    return results


if __name__ == "__main__":
    results = main()

