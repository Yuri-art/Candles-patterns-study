import pandas as pd
import numpy as np
import os


def calculate_pattern_stats(pattern_data, df, candle=None):
    """
    Рассчитывает статистику для бычьих и медвежьих сигналов паттерна
    с учетом последовательных свечей одного цвета

    Args:
        pattern_data (DataFrame): DataFrame с данными о паттерне
        df (DataFrame): Исходный DataFrame с данными
        candle (str, optional): Имя паттерна

    Returns:
        dict: Словарь со статистикой
    """
    bull_high_changes = []
    bull_low_changes = []
    bear_high_changes = []
    bear_low_changes = []

    for index, row in pattern_data.iterrows():
        # Проверяем, не последняя ли это свеча
        if index < len(df) - 1:
            pattern_close = row['close']  # Цена закрытия свечи с сигналом

            # Если candle не указан, предполагаем, что pattern_data содержит столбец с именем паттерна
            pattern_value = row[candle] if candle else row['pattern_value']

            if pattern_value > 0:  # Bull сигнал
                max_high = -float('inf')
                min_low = float('inf')

                # Проверяем последующие свечи, начиная со следующей
                next_index = index + 1

                # Если первая свеча красная, берем только её
                if not df.iloc[next_index]['is_green']:
                    next_candle = df.iloc[next_index]
                    max_high = next_candle['high']
                    min_low = next_candle['low']
                else:
                    # Иначе берем все зеленые свечи подряд
                    while next_index < len(df) and df.iloc[next_index]['is_green']:
                        next_candle = df.iloc[next_index]
                        max_high = max(max_high, next_candle['high'])
                        min_low = min(min_low, next_candle['low'])
                        next_index += 1

                # Если нашли хотя бы одну подходящую свечу
                if max_high != -float('inf') and min_low != float('inf'):
                    high_pct_change = (max_high - pattern_close) / pattern_close * 100
                    low_pct_change = (min_low - pattern_close) / pattern_close * 100

                    bull_high_changes.append(high_pct_change)
                    bull_low_changes.append(low_pct_change)

            else:  # Bear сигнал
                max_high = -float('inf')
                min_low = float('inf')

                # Проверяем последующие свечи, начиная со следующей
                next_index = index + 1

                # Если первая свеча зеленая, берем только её
                if df.iloc[next_index]['is_green']:
                    next_candle = df.iloc[next_index]
                    max_high = next_candle['high']
                    min_low = next_candle['low']
                else:
                    # Иначе берем все красные свечи подряд
                    while next_index < len(df) and not df.iloc[next_index]['is_green']:
                        next_candle = df.iloc[next_index]
                        max_high = max(max_high, next_candle['high'])
                        min_low = min(min_low, next_candle['low'])
                        next_index += 1

                # Если нашли хотя бы одну подходящую свечу
                if max_high != -float('inf') and min_low != float('inf'):
                    high_pct_change = (max_high - pattern_close) / pattern_close * 100
                    low_pct_change = (min_low - pattern_close) / pattern_close * 100

                    bear_high_changes.append(high_pct_change)
                    bear_low_changes.append(low_pct_change)

    # Рассчитываем средние значения
    stats = {}

    if bull_high_changes:
        stats['bull_high_avg'] = np.mean(bull_high_changes)
        stats['bull_low_avg'] = np.mean(bull_low_changes)
        stats['bull_count'] = len(bull_high_changes)
    else:
        stats['bull_high_avg'] = None
        stats['bull_low_avg'] = None
        stats['bull_count'] = 0

    if bear_high_changes:
        stats['bear_high_avg'] = np.mean(bear_high_changes)
        stats['bear_low_avg'] = np.mean(bear_low_changes)
        stats['bear_count'] = len(bear_high_changes)
    else:
        stats['bear_high_avg'] = None
        stats['bear_low_avg'] = None
        stats['bear_count'] = 0

    return stats


def calculate_risk_reward(summary_df):
    """
    Рассчитывает соотношение риск/прибыль для каждого паттерна

    Args:
        summary_df (DataFrame): DataFrame со статистикой по паттернам

    Returns:
        DataFrame: Обновленный DataFrame с соотношением риск/прибыль
    """
    # Создаем копию для модификации
    result_df = summary_df.copy()

    # Инициализируем столбцы для соотношения риск/прибыль
    result_df['bull_risk_reward'] = np.nan
    result_df['bear_risk_reward'] = np.nan

    for idx, row in result_df.iterrows():
        # Для BULL сигналов
        if not np.isnan(row['bull_high_avg']) and not np.isnan(row['bull_low_avg']):
            # Если MIN отрицательный (риск), то считаем соотношение риск/прибыль
            if row['bull_low_avg'] < 0:
                result_df.at[idx, 'bull_risk_reward'] = row['bull_high_avg'] / abs(row['bull_low_avg'])

        # Для BEAR сигналов
        if not np.isnan(row['bear_high_avg']) and not np.isnan(row['bear_low_avg']):
            # Если MAX положительный (риск), то считаем соотношение риск/прибыль
            if row['bear_high_avg'] > 0:
                result_df.at[idx, 'bear_risk_reward'] = abs(row['bear_low_avg']) / row['bear_high_avg']

    return result_df


def compile_pattern_statistics(df, candle_names):
    """
    Собирает статистику по всем паттернам и возвращает сводный DataFrame

    Args:
        df (DataFrame): Исходный DataFrame с данными
        candle_names (list): Список имен свечных паттернов

    Returns:
        DataFrame: Сводный DataFrame со статистикой
    """
    summary_stats = {
        'pattern': [],
        'bull_high_avg': [],
        'bull_low_avg': [],
        'bull_count': [],
        'bear_high_avg': [],
        'bear_low_avg': [],
        'bear_count': []
    }

    # Собираем статистику для всех паттернов
    for candle in candle_names:
        pattern_data = df[df[candle] != 0]

        if not pattern_data.empty:
            stats = calculate_pattern_stats(pattern_data, df, candle)

            # Добавляем данные в сводную таблицу
            summary_stats['pattern'].append(candle)
            summary_stats['bull_high_avg'].append(
                stats['bull_high_avg'] if stats['bull_high_avg'] is not None else np.nan)
            summary_stats['bull_low_avg'].append(stats['bull_low_avg'] if stats['bull_low_avg'] is not None else np.nan)
            summary_stats['bull_count'].append(stats['bull_count'])
            summary_stats['bear_high_avg'].append(
                stats['bear_high_avg'] if stats['bear_high_avg'] is not None else np.nan)
            summary_stats['bear_low_avg'].append(stats['bear_low_avg'] if stats['bear_low_avg'] is not None else np.nan)
            summary_stats['bear_count'].append(stats['bear_count'])

    # Создаем DataFrame из собранных данных
    summary_df = pd.DataFrame(summary_stats)

    # Добавляем соотношение риск/прибыль
    summary_df = calculate_risk_reward(summary_df)

    return summary_df


def save_pattern_statistics(df, candle_names, output_folder, reliable_threshold=5):
    """
    Сохраняет статистику по паттернам в CSV файлы

    Args:
        df (DataFrame): Исходный DataFrame с данными
        candle_names (list): Список имен свечных паттернов
        output_folder (str): Путь к папке для сохранения
        reliable_threshold (int): Порог для определения надежных паттернов

    Returns:
        tuple: (полная статистика, надежные паттерны)
    """
    # Компилируем статистику
    summary_df = compile_pattern_statistics(df, candle_names)

    # Создаем отдельные отсортированные DataFrames для BULL и BEAR паттернов
    # Сортировка BULL паттернов по среднему % MAX (по убыванию)
    summary_bull = summary_df.dropna(subset=['bull_high_avg']).sort_values(by='bull_high_avg', ascending=False)

    # Сортировка BEAR паттернов по среднему % MIN (по возрастанию - самые отрицательные первые)
    summary_bear = summary_df.dropna(subset=['bear_low_avg']).sort_values(by='bear_low_avg', ascending=True)

    # Сохраняем общую статистику
    summary_df.to_csv(f"{output_folder}/all_patterns_stats.csv", index=False)

    # Сохраняем ранжированные списки
    summary_bull.to_csv(f"{output_folder}/bull_patterns_ranking.csv", index=False)
    summary_bear.to_csv(f"{output_folder}/bear_patterns_ranking.csv", index=False)

    # Сохраняем только надежные паттерны (с достаточным количеством сигналов)
    reliable_patterns = summary_df[
        ((summary_df['bull_count'] >= reliable_threshold) & ~summary_df['bull_high_avg'].isna()) |
        ((summary_df['bear_count'] >= reliable_threshold) & ~summary_df['bear_low_avg'].isna())
        ]
    reliable_patterns.to_csv(f"{output_folder}/reliable_patterns.csv", index=False)

    return summary_df, reliable_patterns


def evaluate_strategy(strategy, df):
    """
    Оценивает эффективность торговой стратегии

    Args:
        strategy: Торговая стратегия для оценки
        df (DataFrame): DataFrame с ценовыми данными

    Returns:
        dict: Результаты оценки
    """
    # Генерируем сигналы
    df_with_signals = strategy.generate_signals(df)

    # Выполняем бэктестинг
    results = strategy.backtest(df_with_signals)

    return results


def compare_strategies(strategies_list, df):
    """
    Сравнивает эффективность нескольких торговых стратегий

    Args:
        strategies_list (list): Список торговых стратегий для сравнения
        df (DataFrame): DataFrame с ценовыми данными

    Returns:
        DataFrame: Сравнительная таблица эффективности стратегий
    """
    comparison = {
        'strategy_name': [],
        'total_trades': [],
        'win_rate': [],
        'avg_profit': [],
        'total_profit': [],
        'max_drawdown': [],
        'sharpe_ratio': []
    }

    for strategy in strategies_list:
        results = evaluate_strategy(strategy, df)

        comparison['strategy_name'].append(strategy.name)
        comparison['total_trades'].append(results['total_trades'])
        comparison['win_rate'].append(results['win_rate'])
        comparison['avg_profit'].append(results['avg_profit'])
        comparison['total_profit'].append(results['total_profit'])
        comparison['max_drawdown'].append(results['max_drawdown'])
        comparison['sharpe_ratio'].append(results['sharpe_ratio'])

    return pd.DataFrame(comparison)


def optimize_strategy_parameters(strategy_class, df, param_grid):
    """
    Оптимизирует параметры стратегии методом перебора

    Args:
        strategy_class: Класс торговой стратегии
        df (DataFrame): DataFrame с ценовыми данными
        param_grid (dict): Словарь параметров для оптимизации

    Returns:
        tuple: (лучшие параметры, результаты по всем комбинациям)
    """
    from itertools import product

    # Получаем все возможные комбинации параметров
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    results = []

    # Перебираем все комбинации параметров
    for values in product(*param_values):
        params = dict(zip(param_names, values))

        # Создаем экземпляр стратегии с текущими параметрами
        strategy = strategy_class(**params)

        # Оцениваем стратегию
        evaluation = evaluate_strategy(strategy, df)

        # Добавляем параметры к результатам
        evaluation.update(params)
        results.append(evaluation)

    # Преобразуем результаты в DataFrame
    results_df = pd.DataFrame(results)

    # Находим лучшие параметры (по общей прибыли)
    best_idx = results_df['total_profit'].idxmax()
    best_params = {param: results_df.loc[best_idx, param] for param in param_names}

    return best_params, results_df