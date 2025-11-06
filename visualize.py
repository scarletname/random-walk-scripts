#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для визуализации результатов моделирования случайного блуждания.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Для работы на суперкомпьютере без дисплея
import matplotlib.pyplot as plt
import json
from pathlib import Path
import glob


def load_results(results_dir='results'):
    """
    Загружает результаты из JSON файлов.
    
    Параметры:
    ----------
    results_dir : str
        Директория с результатами
    
    Возвращает:
    -----------
    dict : Словарь с результатами, разделенными по платформам
    """
    results_files = list(Path(results_dir).glob('results_*.json'))
    
    if not results_files:
        print(f"Не найдено файлов результатов в директории {results_dir}")
        return {'local': None, 'supercomputer': None, 'all': []}
    
    results = {'all': []}
    for file in results_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results['all'].append(data)
            platform = data.get('platform', 'unknown')
            if platform in ['local', 'supercomputer']:
                results[platform] = data
    
    return results


def load_sample_positions(results_dir='results'):
    """
    Загружает выборки позиций из .npy файлов.
    
    Параметры:
    ----------
    results_dir : str
        Директория с результатами
    
    Возвращает:
    -----------
    dict : Словарь {имя_файла: массив_позиций}
    """
    position_files = list(Path(results_dir).glob('positions_sample_*.npy'))
    
    positions = {}
    for file in position_files:
        pos = np.load(file)
        positions[str(file.name)] = pos
    
    return positions


def plot_statistics_comparison(results, output_dir='results'):
    """
    Создает графики сравнения результатов с теоретическими значениями.
    
    Параметры:
    ----------
    results : list
        Список словарей с результатами
    output_dir : str
        Директория для сохранения графиков
    """
    if not results:
        print("Нет данных для визуализации")
        return
    
    # Берем первый результат (или можно создать графики для всех)
    result = results[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Статистические характеристики случайного блуждания', fontsize=16, fontweight='bold')
    
    # 1. Сравнение средних значений
    ax1 = axes[0]
    categories = ['Средняя позиция\n(теоретическая)', 'Средняя позиция\n(вычисленная)', 
                  'MSD\n(теоретическое)', 'MSD\n(вычисленное)']
    values = [0, result['mean_position'], 
              result['theoretical_mean_squared_displacement'], 
              result['mean_squared_displacement']]
    colors = ['blue', 'red', 'blue', 'red']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Значение')
    ax1.set_title('Сравнение с теоретическими значениями')
    ax1.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Относительные ошибки
    ax2 = axes[1]
    error_types = ['Ошибка\nсредней позиции', 'Относительная\nошибка MSD (%)']
    error_values = [result['error_mean_position'], result['relative_error_msd']]
    bars2 = ax2.bar(error_types, error_values, color=['orange', 'green'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Значение ошибки')
    ax2.set_title('Точность вычислений')
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, error_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Информация о параметрах и результатах
    ax3 = axes[2]
    ax3.axis('off')
    info_text = f"""
Параметры моделирования:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N (число шагов): {result['N']:,}
M (число траекторий): {result['M']:,}

Результаты:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Средняя позиция: {result['mean_position']:.6f}
Теоретическая: {result['theoretical_mean_position']:.2f}

Среднее квадратичное смещение: {result['mean_squared_displacement']:.2f}
Теоретическое: {result['theoretical_mean_squared_displacement']:.2f}

Статистика:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Стандартное отклонение: {result['std_position']:.2f}
Минимальная позиция: {result['min_position']}
Максимальная позиция: {result['max_position']}

Производительность:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Время выполнения: {result['elapsed_time_seconds']:.2f} сек
({result['elapsed_time_seconds']/60:.2f} минут)
"""
    ax3.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'statistics_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График статистики сохранен: {output_file}")
    plt.close()


def plot_position_distribution(positions_dict, output_dir='results'):
    """
    Создает гистограмму распределения финальных позиций.
    
    Параметры:
    ----------
    positions_dict : dict
        Словарь {имя_файла: массив_позиций}
    output_dir : str
        Директория для сохранения графиков
    """
    if not positions_dict:
        print("Нет данных о позициях для визуализации")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Распределение финальных позиций случайного блуждания', fontsize=16, fontweight='bold')
    
    # Берем первую выборку
    file_name, positions = next(iter(positions_dict.items()))
    
    # Извлекаем N из имени файла или используем значение по умолчанию
    try:
        # Пытаемся извлечь N из имени файла: positions_sample_N10000_M100000000.npy
        import re
        match = re.search(r'N(\d+)_M', file_name)
        N = int(match.group(1)) if match else 10**4
    except:
        N = 10**4
    
    # 1. Гистограмма
    ax1 = axes[0]
    n_bins = min(100, int(np.sqrt(len(positions))))
    counts, bins, patches = ax1.hist(positions, bins=n_bins, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Финальная позиция')
    ax1.set_ylabel('Частота')
    ax1.set_title('Гистограмма распределения позиций')
    ax1.grid(True, alpha=0.3)
    
    # Добавляем вертикальную линию на среднем значении
    mean_pos = np.mean(positions)
    ax1.axvline(mean_pos, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_pos:.2f}')
    ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='Теоретическое среднее: 0')
    ax1.legend()
    
    # 2. Нормализованная гистограмма с теоретическим распределением
    ax2 = axes[1]
    # Теоретическое распределение - нормальное с μ=0, σ²=N
    # Для N=10^4: σ = sqrt(N) = 100
    sigma_theoretical = np.sqrt(N)
    
    # Нормализуем гистограмму
    counts_norm, bins_norm, patches_norm = ax2.hist(positions, bins=n_bins, density=True, 
                                                     edgecolor='black', alpha=0.7, color='lightcoral',
                                                     label='Экспериментальное')
    
    # Теоретическая кривая (нормальное распределение)
    x_theoretical = np.linspace(bins_norm[0], bins_norm[-1], 1000)
    y_theoretical = (1 / (sigma_theoretical * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * (x_theoretical / sigma_theoretical) ** 2)
    ax2.plot(x_theoretical, y_theoretical, 'b-', linewidth=2, label='Теоретическое (N(0, N))')
    
    ax2.set_xlabel('Финальная позиция')
    ax2.set_ylabel('Плотность вероятности')
    ax2.set_title('Сравнение с теоретическим распределением')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'position_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График распределения сохранен: {output_file}")
    plt.close()


def create_comparison_report(results_local=None, results_supercomputer=None, output_dir='results'):
    """
    Создает отчет сравнения результатов локального и суперкомпьютерного выполнения.
    
    Параметры:
    ----------
    results_local : dict or None
        Результаты локального выполнения
    results_supercomputer : dict or None
        Результаты выполнения на суперкомпьютере
    output_dir : str
        Директория для сохранения
    """
    if not results_local and not results_supercomputer:
        print("Нет данных для сравнения")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнение результатов локального и суперкомпьютерного выполнения', 
                 fontsize=16, fontweight='bold')
    
    if results_local and results_supercomputer:
        # Сравнение времени выполнения
        ax1 = axes[0, 0]
        platforms = ['Локальная\nмашина', 'Суперкомпьютер']
        times = [results_local['elapsed_time_seconds'], results_supercomputer['elapsed_time_seconds']]
        bars = ax1.bar(platforms, times, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Время (секунды)')
        ax1.set_title('Время выполнения')
        ax1.grid(True, alpha=0.3)
        for bar, val in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f} сек', ha='center', va='bottom', fontweight='bold')
        
        # Ускорение
        speedup = results_local['elapsed_time_seconds'] / results_supercomputer['elapsed_time_seconds']
        ax1.text(0.5, max(times) * 0.9, f'Ускорение: {speedup:.2f}x', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Сравнение средних позиций
        ax2 = axes[0, 1]
        mean_positions = [results_local['mean_position'], results_supercomputer['mean_position']]
        bars2 = ax2.bar(platforms, mean_positions, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        ax2.axhline(0, color='green', linestyle='--', linewidth=2, label='Теоретическое значение: 0')
        ax2.set_ylabel('Средняя позиция')
        ax2.set_title('Средняя позиция')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Сравнение MSD
        ax3 = axes[1, 0]
        msd_values = [results_local['mean_squared_displacement'], 
                      results_supercomputer['mean_squared_displacement']]
        theoretical_msd = results_local['theoretical_mean_squared_displacement']
        bars3 = ax3.bar(platforms, msd_values, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        ax3.axhline(theoretical_msd, color='green', linestyle='--', linewidth=2, 
                   label=f'Теоретическое: {theoretical_msd:.0f}')
        ax3.set_ylabel('Среднее квадратичное смещение')
        ax3.set_title('Среднее квадратичное смещение')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Относительные ошибки
        ax4 = axes[1, 1]
        error_types = ['Ошибка\nсредней позиции\n(локально)', 
                      'Ошибка\nсредней позиции\n(суперкомпьютер)',
                      'Отн. ошибка MSD\n(локально, %)',
                      'Отн. ошибка MSD\n(суперкомпьютер, %)']
        error_values = [results_local['error_mean_position'],
                       results_supercomputer['error_mean_position'],
                       results_local['relative_error_msd'],
                       results_supercomputer['relative_error_msd']]
        bars4 = ax4.bar(error_types, error_values, color=['lightblue', 'lightcoral', 
                                                          'lightblue', 'lightcoral'], 
                       alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Значение ошибки')
        ax4.set_title('Точность вычислений')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    elif results_local:
        # Только локальные результаты
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.5, 'Данные только с локальной машины.\n\nЗагрузите результаты\nс суперкомпьютера для сравнения.',
                ha='center', va='center', fontsize=14, transform=ax1.transAxes)
        ax1.axis('off')
        
        # Показываем локальные результаты
        ax2 = axes[0, 1]
        info_text = f"""
Локальные результаты:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N: {results_local['N']:,}
M: {results_local['M']:,}
Время: {results_local['elapsed_time_seconds']:.2f} сек
Средняя позиция: {results_local['mean_position']:.6f}
MSD: {results_local['mean_squared_displacement']:.2f}
"""
        ax2.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax2.transAxes)
        ax2.axis('off')
        
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'comparison_report.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Отчет сравнения сохранен: {output_file}")
    plt.close()


def main():
    """Основная функция визуализации."""
    print("Загрузка результатов...")
    results_dict = load_results()
    
    if not results_dict['all']:
        print("Не найдено результатов для визуализации.")
        print("Сначала запустите random_walk.py для генерации данных.")
        return
    
    print(f"Загружено результатов: {len(results_dict['all'])}")
    
    # Создаем графики статистики для всех результатов
    print("\nСоздание графиков статистики...")
    plot_statistics_comparison(results_dict['all'])
    
    # Загружаем и визуализируем распределение позиций
    print("\nЗагрузка данных о позициях...")
    positions = load_sample_positions()
    
    if positions:
        print("Создание графиков распределения...")
        plot_position_distribution(positions)
    
    # Создаем отчет сравнения
    print("\nСоздание отчета сравнения...")
    create_comparison_report(results_dict.get('local'), results_dict.get('supercomputer'))
    
    print("\nВизуализация завершена!")


if __name__ == "__main__":
    main()

