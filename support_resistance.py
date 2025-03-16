import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


def find_extreme_levels(data, order=100, min_touches=5):
    """
    Находит уровни поддержки и сопротивления на основе экстремумов цены

    Args:
        data (DataFrame): DataFrame с ценовыми данными
        order (int): Размер окна для поиска локальных экстремумов
        min_touches (int): Минимальное количество касаний для подтверждения уровня

    Returns:
        tuple: (DataFrame с уровнями поддержки, DataFrame с уровнями сопротивления)
    """
    prices = data['close'].values
    max_idx = argrelextrema(prices, np.greater, order=order)[0]
    min_idx = argrelextrema(prices, np.less, order=order)[0]

    resistance_levels = [(data.iloc[i]['close'], data.iloc[i]['time'], data.iloc[i]['time_num']) for i in max_idx]
    support_levels = [(data.iloc[i]['close'], data.iloc[i]['time'], data.iloc[i]['time_num']) for i in min_idx]

    # Фильтруем уровни по количеству касаний
    support_levels = [(level, time, time_num) for level, time, time_num in support_levels
                      if (data['close'] - level).abs().lt(level * 0.01).sum() >= min_touches]
    resistance_levels = [(level, time, time_num) for level, time, time_num in resistance_levels
                         if (data['close'] - level).abs().lt(level * 0.01).sum() >= min_touches]

    # Преобразуем в DataFrame для удобства дальнейшей работы
    support_df = pd.DataFrame(support_levels, columns=['level', 'formation_time', 'formation_time_num'])
    resistance_df = pd.DataFrame(resistance_levels, columns=['level', 'formation_time', 'formation_time_num'])

    return support_df, resistance_df


def cluster_levels(levels_df, tolerance=0.01):
    """
    Группирует близкие уровни в кластеры
    Кластеризация необходима, потому что часто на графике могут быть очень близкие уровни
    поддержки или сопротивления, которые, по сути, представляют одну и ту же зону интереса
    для трейдеров.
    Объединение таких близких уровней делает анализ более четким и снижает "шум" на графике.
    Args:
        levels_df (DataFrame): DataFrame с уровнями
        tolerance (float): Процент допустимого отклонения для объединения уровней

    Returns:
        DataFrame: DataFrame с кластеризованными уровнями
    """
    if levels_df.empty:
        return levels_df

    # Сортируем уровни по значению
    sorted_levels = levels_df.sort_values('level').reset_index(drop=True)

    # Инициализируем список для хранения кластеров
    clusters = []
    current_cluster = [0]  # Начинаем с первого уровня

    # Группируем близкие уровни
    for i in range(1, len(sorted_levels)):
        current_level = sorted_levels.iloc[i]['level']
        prev_level = sorted_levels.iloc[current_cluster[0]]['level']

        # Если уровень близок к предыдущему, добавляем его в текущий кластер
        if abs(current_level - prev_level) / prev_level <= tolerance:
            current_cluster.append(i)
        else:
            # Иначе завершаем текущий кластер и начинаем новый
            clusters.append(current_cluster)
            current_cluster = [i]

    # Добавляем последний кластер
    if current_cluster:
        clusters.append(current_cluster)

    # Для каждого кластера выбираем "представителя" (уровень с наибольшим количеством касаний)
    clustered_levels = []

    for cluster in clusters:
        cluster_df = sorted_levels.iloc[cluster]

        # Если в кластере только один уровень
        if len(cluster) == 1:
            clustered_levels.append(cluster_df.iloc[0].to_dict())
        else:
            # Найдем средний уровень в кластере и используем самую раннюю дату формирования
            avg_level = cluster_df['level'].mean()
            earliest_idx = cluster_df['formation_time'].argmin() if 'formation_time' in cluster_df.columns else 0

            clustered_level = cluster_df.iloc[earliest_idx].to_dict()
            clustered_level['level'] = avg_level
            clustered_levels.append(clustered_level)

    # Создаем новый DataFrame с кластеризованными уровнями
    return pd.DataFrame(clustered_levels)


def identify_broken_levels(df, levels_df, level_type='support', threshold_pct=0.03, consecutive_bars=2):
    """
    Определяет пробитые уровни поддержки/сопротивления

    Args:
        df (DataFrame): DataFrame с ценовыми данными
        levels_df (DataFrame): DataFrame с уровнями
        level_type (str): Тип уровня ('support' или 'resistance')
        threshold_pct (float): Процент пробития уровня
        consecutive_bars (int): Количество последовательных свечей для подтверждения пробития

    Returns:
        DataFrame: DataFrame с пробитыми уровнями и датами пробития

       Основная логика функции:

        Для каждого уровня она рассматривает только данные после формирования этого уровня
        Проверяет, произошло ли пробитие уровня:

        Для уровня поддержки: цена закрытия должна быть ниже уровня на threshold_pct процентов
        Для уровня сопротивления: цена закрытия должна быть выше уровня на threshold_pct процентов


        Для подтверждения пробоя требуется consecutive_bars последовательных свечей
        Для каждого пробитого уровня собирается информация о:

        значении уровня
        времени формирования уровня
        времени пробития
        цене пробития
        количестве баров от формирования до пробития

    """
    if levels_df.empty:
        return pd.DataFrame()

    broken_levels = []

    for _, level_row in levels_df.iterrows():
        level = level_row['level']
        formation_time = level_row['formation_time']

        # Рассматриваем только данные после формирования уровня
        if isinstance(formation_time, str):
            try:
                formation_time = pd.to_datetime(formation_time)
            except:
                continue

        filtered_df = df[df['time'] > formation_time]

        if filtered_df.empty:
            continue

        # Проверяем пробитие уровня
        if level_type == 'support':
            # Для уровня поддержки: проверяем, закрылась ли цена ниже уровня на threshold_pct процентов
            broken = (filtered_df['close'] < level * (1 - threshold_pct))
        else:  # resistance
            # Для уровня сопротивления: проверяем, закрылась ли цена выше уровня на threshold_pct процентов
            broken = (filtered_df['close'] > level * (1 + threshold_pct))

        # Ищем consecutive_bars последовательных свечей с пробитием
        if broken.any():
            for i in range(len(filtered_df) - consecutive_bars + 1):
                if broken.iloc[i:i + consecutive_bars].all():
                    # Нашли пробитие
                    break_time = filtered_df.iloc[i]['time']
                    break_price = filtered_df.iloc[i]['close']

                    broken_levels.append({
                        'level': level,
                        'formation_time': formation_time,
                        'break_time': break_time,
                        'break_price': break_price,
                        'bars_to_break': i
                    })
                    break

    return pd.DataFrame(broken_levels)


def calculate_level_strength(levels_df, df, window=20):
    """
    Рассчитывает силу уровней на основе объема и количества касаний

    Args:
        levels_df (DataFrame): DataFrame с уровнями
        df (DataFrame): DataFrame с ценовыми данными
        window (int): Размер окна для анализа объема

    Returns:
        DataFrame: DataFrame с уровнями и показателями их силы
    """
    if levels_df.empty:
        return levels_df

    result_df = levels_df.copy()
    result_df['strength'] = 0
    result_df['volume_ratio'] = 0
    result_df['num_touches'] = 0

    for idx, row in result_df.iterrows():
        level = row['level']

        # Считаем количество касаний уровня
        touches = (df['close'] - level).abs().lt(level * 0.01).sum()
        result_df.at[idx, 'num_touches'] = touches

        # Анализируем объем вокруг уровня
        level_rows = df[(df['close'] - level).abs() < level * 0.01]
        if not level_rows.empty:
            avg_volume_at_level = level_rows['volume'].mean()
            avg_volume_overall = df['volume'].mean()
            volume_ratio = avg_volume_at_level / avg_volume_overall if avg_volume_overall > 0 else 1
            result_df.at[idx, 'volume_ratio'] = volume_ratio

            # Рассчитываем общую силу уровня
            # Формула может быть настроена в зависимости от требований
            strength = touches * volume_ratio
            result_df.at[idx, 'strength'] = strength

    return result_df


def find_consolidation_zones(df, threshold_pct=0.02, min_bars=5):
    """
    Находит зоны консолидации (боковое движение цены)

    Args:
        df (DataFrame): DataFrame с ценовыми данными
        threshold_pct (float): Максимальный процент колебания цены в зоне консолидации
        min_bars (int): Минимальное количество свечей для зоны консолидации

    Returns:
        list: Список зон консолидации (кортежи с индексами начала и конца)
    """
    consolidation_zones = []
    i = 0

    while i < len(df) - min_bars:
        # Проверяем следующие min_bars свечей
        window = df.iloc[i:i + min_bars]
        price_range = (window['high'].max() - window['low'].min()) / window['low'].min()

        if price_range <= threshold_pct:
            # Нашли начало зоны консолидации
            start_idx = i

            # Расширяем зону, пока она соответствует критериям
            end_idx = i + min_bars
            while end_idx < len(df):
                extended_window = df.iloc[start_idx:end_idx + 1]
                extended_range = (extended_window['high'].max() - extended_window['low'].min()) / extended_window[
                    'low'].min()

                if extended_range <= threshold_pct:
                    end_idx += 1
                else:
                    break

            # Добавляем зону в список
            consolidation_zones.append((start_idx, end_idx - 1))

            # Переходим к позиции после найденной зоны
            i = end_idx
        else:
            i += 1

    return consolidation_zones


def get_level_touches(df, level, tolerance_pct=0.01):
    """
    Возвращает все касания указанного уровня

    Args:
        df (DataFrame): DataFrame с ценовыми данными
        level (float): Значение уровня
        tolerance_pct (float): Допустимое отклонение в процентах

    Returns:
        DataFrame: DataFrame с данными о касаниях уровня
    """
    # Рассчитываем допустимое отклонение
    tolerance = level * tolerance_pct

    # Находим свечи, которые касаются уровня
    touches = df[(df['high'] >= level - tolerance) & (df['low'] <= level + tolerance)].copy()

    if not touches.empty:
        # Добавляем информацию о том, как произошло касание
        touches['touch_type'] = 'unknown'
        touches.loc[touches['high'] >= level - tolerance, 'touch_type'] = 'resistance'
        touches.loc[touches['low'] <= level + tolerance, 'touch_type'] = 'support'
        touches.loc[
            (touches['high'] >= level - tolerance) & (touches['low'] <= level + tolerance), 'touch_type'] = 'both'

        # Добавляем информацию о расстоянии до уровня
        touches['distance_pct'] = np.minimum(
            abs(touches['high'] - level) / level,
            abs(touches['low'] - level) / level
        ) * 100

    return touches