import pandas as pd
import numpy as np
import talib


def add_candlestick_patterns(df):
    """
    Добавляет свечные паттерны в DataFrame

    Args:
        df (DataFrame): DataFrame с данными OHLCV

    Returns:
        DataFrame: DataFrame с добавленными свечными паттернами
    """
    # Получаем все функции для распознавания паттернов из TA-Lib
    candle_names = talib.get_function_groups()['Pattern Recognition']

    # Исключаем некоторые паттерны, если необходимо
    removed = ['CDLCOUNTERATTACK', 'CDLLONGLINE', 'CDLSHORTLINE', 'CDLSTALLEDPATTERN', 'CDLKICKINGBYLENGTH']
    candle_names = [name for name in candle_names if name not in removed]

    # Добавляем каждый паттерн в DataFrame
    op, hi, lo, cl = df['open'], df['high'], df['low'], df['close']

    for candle in candle_names:
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)

    return df, candle_names


def add_technical_indicators(df, include_all=False):
    """
    Добавляет технические индикаторы в DataFrame

    Args:
        df (DataFrame): DataFrame с данными OHLCV
        include_all (bool): Если True, добавляет все доступные индикаторы

    Returns:
        DataFrame: DataFrame с добавленными индикаторами
    """
    # Индикаторы, основанные на цене закрытия
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)

    # Скользящие средние
    df['sma_9'] = talib.SMA(df['close'], timeperiod=9)
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)

    # Полосы Боллинджера
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    # ATR (Average True Range) для оценки волатильности
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    if include_all:
        # Стохастический осциллятор
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                   fastk_period=5, slowk_period=3, slowk_matype=0,
                                   slowd_period=3, slowd_matype=0)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd

        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # Другие индикаторы можно добавить по мере необходимости

    return df


def detect_support_resistance_zones(df, levels_df, buffer_pct=0.01):
    """
    Добавляет маркеры зон поддержки и сопротивления

    Args:
        df (DataFrame): DataFrame с данными OHLCV
        levels_df (DataFrame): DataFrame с уровнями поддержки/сопротивления
        buffer_pct (float): Процент буфера вокруг уровня для определения зоны

    Returns:
        DataFrame: Исходный DataFrame с добавленными маркерами зон
    """
    result_df = df.copy()

    # Инициализируем столбцы для маркировки зон
    result_df['in_support_zone'] = False
    result_df['in_resistance_zone'] = False

    # Если dataframe с уровнями пуст, возвращаем исходный dataframe
    if levels_df.empty:
        return result_df

    # Проверяем для каждой свечи, находится ли она в зоне поддержки или сопротивления
    for i, row in result_df.iterrows():
        current_price = row['close']

        # Проверяем зоны поддержки
        for _, level_row in levels_df.iterrows():
            level = level_row['level']
            buffer = level * buffer_pct

            # Если цена близка к уровню, отмечаем, что она в зоне
            if abs(current_price - level) <= buffer:
                result_df.at[i, 'in_support_zone' if current_price >= level else 'in_resistance_zone'] = True
                break

    return result_df