#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для визуализации распределения финальных позиций случайного блуждания.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re


def plot_position_distribution(positions_file, output_file='position_distribution.png', N=10000):
    """
    Создает гистограмму распределения финальных позиций.
    
    Параметры:
    ----------
    positions_file : str
        Путь к файлу с выборкой позиций (.npy)
    output_file : str
        Имя выходного файла
    N : int
        Количество шагов (для вычисления теоретического распределения)
    """
    positions = np.load(positions_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Распределение финальных позиций случайного блуждания', 
                 fontsize=16, fontweight='bold')
    
    n_bins = min(100, max(10, int(np.sqrt(len(positions)))))
    sigma_theoretical = np.sqrt(N)
    
    # 1. Гистограмма частот
    ax1 = axes[0]
    counts, bins, patches = ax1.hist(positions, bins=n_bins, edgecolor='black', 
                                     alpha=0.7, color='skyblue')
    ax1.set_xlabel('Финальная позиция')
    ax1.set_ylabel('Частота')
    ax1.set_title('Гистограмма распределения позиций')
    ax1.grid(True, alpha=0.3)
    
    mean_pos = np.mean(positions)
    ax1.axvline(mean_pos, color='red', linestyle='--', linewidth=2, 
                label=f'Среднее: {mean_pos:.2f}')
    ax1.axvline(0, color='green', linestyle='--', linewidth=2, 
                label='Теоретическое среднее: 0')
    ax1.legend()
    
    # 2. Нормализованная гистограмма с теоретическим распределением
    ax2 = axes[1]
    counts_norm, bins_norm, patches_norm = ax2.hist(positions, bins=n_bins, 
                                                     density=True, 
                                                     edgecolor='black', 
                                                     alpha=0.7, 
                                                     color='lightcoral',
                                                     label='Экспериментальное')
    
    # Теоретическая кривая (нормальное распределение N(0, N))
    x_theoretical = np.linspace(bins_norm[0], bins_norm[-1], 1000)
    y_theoretical = (1 / (sigma_theoretical * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * (x_theoretical / sigma_theoretical) ** 2)
    ax2.plot(x_theoretical, y_theoretical, 'b-', linewidth=2, 
             label='Теоретическое (N(0, N))')
    
    ax2.set_xlabel('Финальная позиция')
    ax2.set_ylabel('Плотность вероятности')
    ax2.set_title('Сравнение с теоретическим распределением')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранен: {output_file}")


def main():
    """Основная функция."""
    import sys
    
    if len(sys.argv) < 2:
        # Ищем файлы позиций в текущей директории
        position_files = list(Path('.').glob('**/positions_sample_*.npy'))
        if not position_files:
            print("Не найдено файлов positions_sample_*.npy")
            print("Использование: python visualize.py <путь_к_файлу.npy> [N]")
            return
        positions_file = position_files[0]
        print(f"Найден файл: {positions_file}")
    else:
        positions_file = sys.argv[1]
    
    # Извлекаем N из имени файла или используем значение по умолчанию
    N = 10000
    if len(sys.argv) > 2:
        N = int(sys.argv[2])
    else:
        # Пытаемся извлечь N из имени файла
        match = re.search(r'N(\d+)_M', str(positions_file))
        if match:
            N = int(match.group(1))
    
    output_file = 'position_distribution.png'
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    plot_position_distribution(positions_file, output_file, N)


if __name__ == "__main__":
    main()
