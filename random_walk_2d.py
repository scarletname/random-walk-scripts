#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Моделирование двумерного случайного блуждания (4 и 8 направлений).
"""

import numpy as np
import time
import json
import sys
import gc
from pathlib import Path
from datetime import datetime


def simulate_2d_random_walk_4_directions(N, M, seed=None, batch_size=None):
    """
    Моделирует M траекторий 2D случайного блуждания с 4 направлениями (N, S, E, W).
    
    Параметры:
    ----------
    N : int
        Количество шагов в каждой траектории
    M : int
        Количество траекторий для моделирования
    seed : int, optional
        Seed для генератора случайных чисел
    batch_size : int, optional
        Размер батча для обработки
    
    Возвращает:
    -----------
    dict : Результаты моделирования
    ndarray : Выборка финальных позиций (x, y) для визуализации
    ndarray : Выборка радиальных расстояний
    ndarray : Выборка финальных углов
    """
    if seed is not None:
        np.random.seed(seed)
    
    if batch_size is None:
        batch_size = min(10000, M)
    
    start_time = time.time()
    
    # Направления: 0=N, 1=E, 2=S, 3=W
    directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])  # N, E, S, W
    
    sum_x = 0.0
    sum_y = 0.0
    sum_r_squared = 0.0  # MSD = <r²> = <x² + y²>
    sum_r = 0.0  # Среднее радиальное расстояние
    
    max_sample_size = 10000
    positions_sample = np.zeros((max_sample_size, 2), dtype=np.int32)
    radial_distances_sample = np.zeros(max_sample_size, dtype=np.float32)
    angles_sample = np.zeros(max_sample_size, dtype=np.float32)
    sample_count = 0
    
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    num_batches = (M + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, M)
        current_batch_size = end_idx - start_idx
        
        # Вывод прогресса каждые 10% или каждые 50 батчей
        if batch_idx == 0 or (batch_idx + 1) % max(1, num_batches // 20) == 0 or (batch_idx + 1) % 50 == 0:
            progress = 100.0 * (batch_idx + 1) / num_batches
            print(f"Прогресс 4 направлений: {progress:.1f}% ({batch_idx + 1}/{num_batches} батчей)", flush=True)
        
        # Оптимизированный алгоритм: подсчитываем количество каждого направления
        # вместо создания массива смещений (batch_size, N, 2) - это экономит ~16GB памяти
        random_directions = np.random.randint(0, 4, size=(current_batch_size, N), dtype=np.int32)
        
        # Подсчет количества каждого направления для каждой траектории
        # directions: 0=N(0,1), 1=E(1,0), 2=S(0,-1), 3=W(-1,0)
        counts_N = np.sum(random_directions == 0, axis=1, dtype=np.int32)  # количество шагов N
        counts_E = np.sum(random_directions == 1, axis=1, dtype=np.int32)  # количество шагов E
        counts_S = np.sum(random_directions == 2, axis=1, dtype=np.int32)  # количество шагов S
        counts_W = np.sum(random_directions == 3, axis=1, dtype=np.int32)  # количество шагов W
        
        # Вычисление финальных позиций напрямую без создания массива смещений
        batch_x = counts_E - counts_W  # E дает +1, W дает -1
        batch_y = counts_N - counts_S  # N дает +1, S дает -1
        batch_positions = np.stack([batch_x, batch_y], axis=1).astype(np.int32)
        
        del random_directions, counts_N, counts_E, counts_S, counts_W
        
        # Радиальные расстояния (batch_x и batch_y уже вычислены выше)
        batch_r_squared = (batch_x.astype(np.int64) ** 2 + batch_y.astype(np.int64) ** 2).astype(np.float64)
        batch_r = np.sqrt(batch_r_squared)
        
        # Углы (в радианах, от -π до π)
        batch_angles = np.arctan2(batch_y, batch_x)
        
        # Агрегация статистики
        sum_x += np.sum(batch_x, dtype=np.float64)
        sum_y += np.sum(batch_y, dtype=np.float64)
        sum_r_squared += np.sum(batch_r_squared, dtype=np.float64)
        sum_r += np.sum(batch_r, dtype=np.float64)
        
        # Обновление min/max
        batch_min_x, batch_max_x = int(np.min(batch_x)), int(np.max(batch_x))
        batch_min_y, batch_max_y = int(np.min(batch_y)), int(np.max(batch_y))
        min_x = min(min_x, batch_min_x)
        max_x = max(max_x, batch_max_x)
        min_y = min(min_y, batch_min_y)
        max_y = max(max_y, batch_max_y)
        
        # Сохранение выборки для визуализации
        if sample_count < max_sample_size:
            remaining_slots = max_sample_size - sample_count
            sample_size = min(50, current_batch_size, remaining_slots)
            if sample_size > 0:
                step = max(1, current_batch_size // sample_size)
                indices = np.arange(0, current_batch_size, step)[:sample_size]
                end_sample = sample_count + sample_size
                positions_sample[sample_count:end_sample] = batch_positions[indices]
                radial_distances_sample[sample_count:end_sample] = batch_r[indices]
                angles_sample[sample_count:end_sample] = batch_angles[indices]
                sample_count = end_sample
        
        del batch_positions, batch_x, batch_y, batch_r_squared, batch_r, batch_angles
        
        # Сборка мусора реже для ускорения (каждые 50 батчей вместо 10)
        if (batch_idx + 1) % 50 == 0:
            gc.collect()
    
    # Вычисление итоговых статистик
    mean_x = sum_x / M
    mean_y = sum_y / M
    mean_r_squared = sum_r_squared / M  # MSD
    mean_r = sum_r / M
    
    # Теоретические значения
    theoretical_msd = float(N)  # MSD = <r²> = N для 2D
    
    elapsed_time = time.time() - start_time
    
    # Подготовка выборок для визуализации
    final_positions_sample = positions_sample[:sample_count].copy() if sample_count > 0 else None
    final_radial_distances = radial_distances_sample[:sample_count].copy() if sample_count > 0 else None
    final_angles = angles_sample[:sample_count].copy() if sample_count > 0 else None
    
    del positions_sample, radial_distances_sample, angles_sample
    gc.collect()
    
    results = {
        'N': N,
        'M': M,
        'seed': seed,
        'batch_size': batch_size,
        'directions': 4,
        'mean_x': float(mean_x),
        'mean_y': float(mean_y),
        'mean_radial_distance': float(mean_r),
        'mean_squared_displacement': float(mean_r_squared),
        'theoretical_msd': theoretical_msd,
        'relative_error_msd': abs(mean_r_squared - theoretical_msd) / theoretical_msd * 100,
        'min_x': int(min_x),
        'max_x': int(max_x),
        'min_y': int(min_y),
        'max_y': int(max_y),
        'elapsed_time_seconds': elapsed_time,
        'sample_size': sample_count
    }
    
    return results, final_positions_sample, final_radial_distances, final_angles


def simulate_2d_random_walk_8_directions(N, M, seed=None, batch_size=None):
    """
    Моделирует M траекторий 2D случайного блуждания с 8 направлениями (N, NE, E, SE, S, SW, W, NW).
    
    Параметры:
    ----------
    N : int
        Количество шагов в каждой траектории
    M : int
        Количество траекторий для моделирования
    seed : int, optional
        Seed для генератора случайных чисел
    batch_size : int, optional
        Размер батча для обработки
    
    Возвращает:
    -----------
    dict : Результаты моделирования
    ndarray : Выборка финальных позиций (x, y) для визуализации
    ndarray : Выборка радиальных расстояний
    ndarray : Выборка финальных углов
    """
    if seed is not None:
        np.random.seed(seed)
    
    if batch_size is None:
        batch_size = min(10000, M)
    
    start_time = time.time()
    
    # Направления: 8 направлений (включая диагонали)
    # 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
    sqrt2_inv = 1.0 / np.sqrt(2)
    directions = np.array([
        [0, 1],           # N
        [sqrt2_inv, sqrt2_inv],   # NE
        [1, 0],           # E
        [sqrt2_inv, -sqrt2_inv],  # SE
        [0, -1],          # S
        [-sqrt2_inv, -sqrt2_inv], # SW
        [-1, 0],          # W
        [-sqrt2_inv, sqrt2_inv]   # NW
    ], dtype=np.float32)
    
    sum_x = 0.0
    sum_y = 0.0
    sum_r_squared = 0.0
    sum_r = 0.0
    
    max_sample_size = 10000
    positions_sample = np.zeros((max_sample_size, 2), dtype=np.float32)
    radial_distances_sample = np.zeros(max_sample_size, dtype=np.float32)
    angles_sample = np.zeros(max_sample_size, dtype=np.float32)
    sample_count = 0
    
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    num_batches = (M + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, M)
        current_batch_size = end_idx - start_idx
        
        # Вывод прогресса каждые 10% или каждые 50 батчей
        if batch_idx == 0 or (batch_idx + 1) % max(1, num_batches // 20) == 0 or (batch_idx + 1) % 50 == 0:
            progress = 100.0 * (batch_idx + 1) / num_batches
            print(f"Прогресс 8 направлений: {progress:.1f}% ({batch_idx + 1}/{num_batches} батчей)", flush=True)
        
        # Оптимизированный алгоритм для 8 направлений
        # Для 8 направлений сложнее, но можно оптимизировать суммирование
        random_directions = np.random.randint(0, 8, size=(current_batch_size, N), dtype=np.int32)
        
        # Быстрое вычисление позиций через прямое суммирование смещений
        # Используем то, что каждое направление имеет фиксированное смещение
        sqrt2_inv = 1.0 / np.sqrt(2)
        
        # Подсчет количества каждого направления (полностью векторизованная версия)
        # Используем broadcasting для подсчета всех направлений одновременно
        counts = np.zeros((current_batch_size, 8), dtype=np.int32)
        for dir_idx in range(8):
            counts[:, dir_idx] = np.sum(random_directions == dir_idx, axis=1, dtype=np.int32)
        
        # Вычисление позиций: x = Σ(count * dx), y = Σ(count * dy)
        # directions: [N, NE, E, SE, S, SW, W, NW]
        # dx для каждого: [0, sqrt2_inv, 1, sqrt2_inv, 0, -sqrt2_inv, -1, -sqrt2_inv]
        # dy для каждого: [1, sqrt2_inv, 0, -sqrt2_inv, -1, -sqrt2_inv, 0, sqrt2_inv]
        dx = np.array([0, sqrt2_inv, 1, sqrt2_inv, 0, -sqrt2_inv, -1, -sqrt2_inv], dtype=np.float32)
        dy = np.array([1, sqrt2_inv, 0, -sqrt2_inv, -1, -sqrt2_inv, 0, sqrt2_inv], dtype=np.float32)
        
        batch_x = np.dot(counts.astype(np.float32), dx)
        batch_y = np.dot(counts.astype(np.float32), dy)
        batch_positions = np.stack([batch_x, batch_y], axis=1).astype(np.float32)
        
        del random_directions, counts
        
        batch_r_squared = batch_x ** 2 + batch_y ** 2
        batch_r = np.sqrt(batch_r_squared)
        batch_angles = np.arctan2(batch_y, batch_x)
        
        # Агрегация статистики
        sum_x += np.sum(batch_x, dtype=np.float64)
        sum_y += np.sum(batch_y, dtype=np.float64)
        sum_r_squared += np.sum(batch_r_squared, dtype=np.float64)
        sum_r += np.sum(batch_r, dtype=np.float64)
        
        # Обновление min/max
        batch_min_x, batch_max_x = float(np.min(batch_x)), float(np.max(batch_x))
        batch_min_y, batch_max_y = float(np.min(batch_y)), float(np.max(batch_y))
        min_x = min(min_x, batch_min_x)
        max_x = max(max_x, batch_max_x)
        min_y = min(min_y, batch_min_y)
        max_y = max(max_y, batch_max_y)
        
        # Сохранение выборки
        if sample_count < max_sample_size:
            remaining_slots = max_sample_size - sample_count
            sample_size = min(50, current_batch_size, remaining_slots)
            if sample_size > 0:
                step = max(1, current_batch_size // sample_size)
                indices = np.arange(0, current_batch_size, step)[:sample_size]
                end_sample = sample_count + sample_size
                positions_sample[sample_count:end_sample] = batch_positions[indices]
                radial_distances_sample[sample_count:end_sample] = batch_r[indices]
                angles_sample[sample_count:end_sample] = batch_angles[indices]
                sample_count = end_sample
        
        del batch_positions, batch_x, batch_y, batch_r_squared, batch_r, batch_angles
        
        # Сборка мусора реже для ускорения (каждые 50 батчей вместо 10)
        if (batch_idx + 1) % 50 == 0:
            gc.collect()
    
    # Вычисление итоговых статистик
    mean_x = sum_x / M
    mean_y = sum_y / M
    mean_r_squared = sum_r_squared / M  # MSD
    mean_r = sum_r / M
    
    # Теоретические значения
    # Для 8 направлений длина шага не всегда 1 (диагонали = sqrt(2)/2)
    # Средняя длина шага = (4*1 + 4*sqrt(2)/2) / 8 = (4 + 2*sqrt(2)) / 8
    avg_step_length_sq = (4 * 1.0 + 4 * 0.5) / 8.0  # (4*1² + 4*(1/√2)²) / 8 = 0.75
    theoretical_msd = float(N * avg_step_length_sq)
    
    elapsed_time = time.time() - start_time
    
    # Подготовка выборок
    final_positions_sample = positions_sample[:sample_count].copy() if sample_count > 0 else None
    final_radial_distances = radial_distances_sample[:sample_count].copy() if sample_count > 0 else None
    final_angles = angles_sample[:sample_count].copy() if sample_count > 0 else None
    
    del positions_sample, radial_distances_sample, angles_sample
    gc.collect()
    
    results = {
        'N': N,
        'M': M,
        'seed': seed,
        'batch_size': batch_size,
        'directions': 8,
        'mean_x': float(mean_x),
        'mean_y': float(mean_y),
        'mean_radial_distance': float(mean_r),
        'mean_squared_displacement': float(mean_r_squared),
        'theoretical_msd': theoretical_msd,
        'relative_error_msd': abs(mean_r_squared - theoretical_msd) / theoretical_msd * 100,
        'min_x': float(min_x),
        'max_x': float(max_x),
        'min_y': float(min_y),
        'max_y': float(max_y),
        'elapsed_time_seconds': elapsed_time,
        'sample_size': sample_count
    }
    
    return results, final_positions_sample, final_radial_distances, final_angles


def save_results(results, positions, radial_distances, angles, output_dir='results'):
    """Сохраняет результаты в файлы."""
    Path(output_dir).mkdir(exist_ok=True)
    
    directions = results['directions']
    N = results['N']
    M = results['M']
    
    # Сохранение результатов в JSON
    json_file = Path(output_dir) / f"results_2d_{directions}dir_N{N}_M{M}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Сохранение данных для визуализации
    positions_file = None
    radial_file = None
    angles_file = None
    
    if positions is not None:
        positions_file = Path(output_dir) / f"positions_2d_{directions}dir_N{N}_M{M}.npy"
        np.save(positions_file, positions)
    
    if radial_distances is not None:
        radial_file = Path(output_dir) / f"radial_distances_{directions}dir_N{N}_M{M}.npy"
        np.save(radial_file, radial_distances)
    
    if angles is not None:
        angles_file = Path(output_dir) / f"angles_{directions}dir_N{N}_M{M}.npy"
        np.save(angles_file, angles)
    
    print(f"Результаты сохранены для {directions} направлений:")
    print(f"  JSON: {json_file}")
    if positions_file is not None:
        print(f"  Позиции: {positions_file}")
    if radial_file is not None:
        print(f"  Радиальные расстояния: {radial_file}")
    if angles_file is not None:
        print(f"  Углы: {angles_file}")


def main():
    """Главная функция."""
    N = 10**4
    M = 10**8
    seed = 42
    batch_size = None
    
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        M = int(sys.argv[2])
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
    if len(sys.argv) > 4:
        batch_size_arg = int(sys.argv[4])
        # Если batch_size = 0, используем автоматический расчет
        batch_size = None if batch_size_arg == 0 else batch_size_arg
    
    # Автоматический выбор размера батча
    # Расчет основывается на пиковом использовании памяти в процессе обработки батча
    # Основные массивы в памяти одновременно:
    # 1. random_directions: (batch_size, N) * int32 = batch_size * N * 4 байт
    # 2. batch_displacements: (batch_size, N, 2) * int32/float32 = batch_size * N * 2 * 4 байт
    # 3. batch_positions: (batch_size, 2) * int32/float32 = batch_size * 2 * 4 байт
    # 4. batch_x, batch_y: views (batch_size,) * int32/float32
    # 5. batch_r_squared: (batch_size,) * float64 = batch_size * 8 байт
    # 6. batch_r: (batch_size,) * float64 = batch_size * 8 байт
    # 7. batch_angles: (batch_size,) * float32 = batch_size * 4 байт
    # 
    # Пиковое использование: random_directions + batch_displacements + остальные
    # ≈ batch_size * N * 4 + batch_size * N * 8 + batch_size * 20 байт
    # ≈ batch_size * (N * 12 + 20) байт
    # 
    # Для 8 направлений все float32, так что похоже
    # Используем консервативную оценку: batch_size * N * 16 байт (с запасом)
    if batch_size is None:
        try:
            import psutil
            # Получаем доступную память в байтах
            available_memory_bytes = psutil.virtual_memory().available
            
            # Используем только 75% доступной памяти (оставляем запас для системы и Python)
            usable_memory = available_memory_bytes * 0.75
            
            # Расчет максимального batch_size на основе пикового использования памяти
            # Для 4 направлений: int32, для 8 направлений: float32
            # Учитываем самый большой массив: batch_displacements (batch_size, N, 2)
            # + random_directions (batch_size, N) + промежуточные массивы
            # Консервативная оценка: ~16 байт на элемент на шаг (N * batch_size * 16 байт)
            bytes_per_element_per_step = 16  # Запас с учетом всех массивов
            
            max_batch_size = int(usable_memory / (N * bytes_per_element_per_step))
            
            # Ограничения: минимум 1000, максимум M
            batch_size = max(1000, min(max_batch_size, M))
            
            # Выводим информацию о расчете
            print(f"Расчет batch_size:")
            print(f"  Доступная память: {available_memory_bytes / 1024**3:.2f} GB")
            print(f"  Используемая память (75%): {usable_memory / 1024**3:.2f} GB")
            print(f"  Расчетный max_batch_size: {max_batch_size:,}")
            print(f"  Выбранный batch_size: {batch_size:,}")
            print(f"  Ожидаемое использование памяти на батч: ~{batch_size * N * bytes_per_element_per_step / 1024**3:.3f} GB")
            print()
        except ImportError:
            # Если psutil недоступен, используем консервативное значение
            batch_size = min(10000, M)
            print(f"psutil недоступен, используется batch_size по умолчанию: {batch_size:,}")
            print()
    
    print("=" * 60)
    print("МОДЕЛИРОВАНИЕ 2D СЛУЧАЙНОГО БЛУЖДАНИЯ")
    print("=" * 60)
    print(f"Параметры: N={N}, M={M}, seed={seed}, batch_size={batch_size}")
    print()
    sys.stdout.flush()  # Принудительная очистка буфера
    
    # Моделирование с 4 направлениями
    print("Моделирование с 4 направлениями (N, E, S, W)...")
    print("-" * 60)
    sys.stdout.flush()  # Принудительная очистка буфера
    results_4dir, pos_4dir, rad_4dir, ang_4dir = simulate_2d_random_walk_4_directions(
        N, M, seed, batch_size
    )
    save_results(results_4dir, pos_4dir, rad_4dir, ang_4dir)
    
    print(f"Время выполнения (4 направления): {results_4dir['elapsed_time_seconds']:.2f} секунд")
    print(f"MSD: {results_4dir['mean_squared_displacement']:.2f} (теоретическое: {results_4dir['theoretical_msd']:.2f})")
    print(f"Относительная ошибка MSD: {results_4dir['relative_error_msd']:.4f}%")
    print()
    
    # Моделирование с 8 направлениями
    print("Моделирование с 8 направлениями (включая диагонали)...")
    print("-" * 60)
    sys.stdout.flush()  # Принудительная очистка буфера
    results_8dir, pos_8dir, rad_8dir, ang_8dir = simulate_2d_random_walk_8_directions(
        N, M, seed, batch_size
    )
    save_results(results_8dir, pos_8dir, rad_8dir, ang_8dir)
    
    print(f"Время выполнения (8 направлений): {results_8dir['elapsed_time_seconds']:.2f} секунд")
    print(f"MSD: {results_8dir['mean_squared_displacement']:.2f} (теоретическое: {results_8dir['theoretical_msd']:.2f})")
    print(f"Относительная ошибка MSD: {results_8dir['relative_error_msd']:.4f}%")
    print()
    
    # Итоговое сравнение
    print("=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 60)
    print(f"Время выполнения (4 направления):  {results_4dir['elapsed_time_seconds']:.2f} с")
    print(f"Время выполнения (8 направлений):  {results_8dir['elapsed_time_seconds']:.2f} с")
    print(f"Отношение времени (8/4):           {results_8dir['elapsed_time_seconds'] / results_4dir['elapsed_time_seconds']:.3f}")
    print()
    print(f"MSD (4 направления):               {results_4dir['mean_squared_displacement']:.2f}")
    print(f"MSD (8 направлений):               {results_8dir['mean_squared_displacement']:.2f}")
    
    return results_4dir, results_8dir


if __name__ == "__main__":
    results_4dir, results_8dir = main()

