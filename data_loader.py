import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import matplotlib.dates as mdates
import talib

def get_binance_klines(symbol='ETHUSDT', interval='1h', start_date='2025-01-01'):
    """
    Загружает исторические данные с Binance API

    Args:
        symbol (str): Торговая пара (например, 'ETHUSDT')
        interval (str): Интервал свечей (например, '1h', '4h', '1d')
        start_date (str): Дата начала в формате 'YYYY-MM-DD'

    Returns:
        list: Список данных Kline/Candlestick
    """
    base_url = "https://api.binance.com/api/v3/klines"
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_time = int(datetime.now().timestamp() * 1000)
    limit = 1000

    all_data = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        response = requests.get(base_url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        start_time = data[-1][0] + 1
        time.sleep(0.5)  # Добавляем задержку, чтобы не превысить лимиты API

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        print(f"Получено {len(all_data)} свечей")  # Этот лог показывает 0 свечей

        if not all_data:
            print("API вернул пустой список данных")
    except Exception as e:
        print(f"Ошибка при запросе к API: {e}")
        return []


    return all_data

# Загрузка данных для двух таймфреймов
def load_multi_timeframe_data(symbol, start_date, main_interval='1h', exec_interval='15m'):
    """
    Загружает и подготавливает данные для двух таймфреймов

    Args:
        symbol (str): Торговая пара
        start_date (str): Дата начала в формате YYYY-MM-DD
        main_interval (str): Интервал для анализа сигналов
        exec_interval (str): Интервал для исполнения сделок

    Returns:
        tuple: (DataFrame для main_interval, DataFrame для exec_interval)
    """
    # Загружаем 1h данные
    h1_raw_data = get_binance_klines(symbol, main_interval, start_date)
    h1_df = prepare_dataframe(h1_raw_data)
    h1_df = add_time_features(h1_df)
    h1_df = add_candle_features(h1_df)

    # Загружаем 15m данные
    m15_raw_data = get_binance_klines(symbol, exec_interval, start_date)
    m15_df = prepare_dataframe(m15_raw_data)
    m15_df = add_time_features(m15_df)
    m15_df = add_candle_features(m15_df)

    # Добавляем технические индикаторы
    h1_df = add_technical_indicators(h1_df)

    # Важно: добавляем ATR и на 15-минутный график для расчетов трейлинг-стопа
    m15_df['atr'] = talib.ATR(m15_df['high'], m15_df['low'], m15_df['close'], timeperiod=14)

    # Добавляем свечные паттерны только к 1h данным
    h1_df, _ = add_candlestick_patterns(h1_df)

    return h1_df, m15_df

def prepare_dataframe(raw_data, timezone='UTC'):
    """
    Преобразует сырые данные из Binance API в pandas DataFrame

    Args:
        raw_data (list): Список данных Kline/Candlestick
        timezone (str): Часовой пояс для преобразования времени

    Returns:
        DataFrame: Обработанный DataFrame с ценовыми данными
       
            Индекс	Поле	Описание
        0	open_time	Время открытия свечи (timestamp в мс)
        1	open	Цена открытия
        2	high	Максимальная цена
        3	low	Минимальная цена
        4	close	Цена закрытия
        5	volume	Объем (количество базового актива)
        6	close_time	Время закрытия свечи (timestamp в мс)
        7	quote_asset_volume	Объем торгов в котируемой валюте
        8	number_of_trades	Количество сделок
        9	taker_buy_base_asset_volume	Объем базового актива, купленного маркет-ордерами
        10	taker_buy_quote_asset_volume	Объем котируемого актива, купленного маркет-ордерами
        11	ignore	Игнорируемый параметр (всегда "0")   
        
    """
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
               'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']

    # Создаем DataFrame
    df = pd.DataFrame(raw_data, columns=columns)

    # Преобразование типов данных
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Оставляем только нужные столбцы
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Преобразуем числовые столбцы
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Устанавливаем временную метку как индекс и преобразуем часовой пояс
    df.set_index('timestamp', inplace=True)

    if timezone != 'UTC':
        df.index = df.index.tz_localize('UTC').tz_convert(timezone)

    # Сбрасываем индекс и переименовываем столбцы
    df.reset_index(inplace=True)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    return df


def add_time_features(df):
    """
    Добавляет признаки, связанные со временем

    Args:
        df (DataFrame): DataFrame с ценовыми данными

    Returns:
        DataFrame: DataFrame с добавленными временными признаками
    """
    df = df.copy()

    # Добавляем признаки для визуализации
    df['time_num'] = mdates.date2num(df['time'])

    # Добавляем признаки для анализа
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_of_month'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    return df


def add_candle_features(df):
    """
    Добавляет признаки, связанные со свечами

    Args:
        df (DataFrame): DataFrame с ценовыми данными

    Returns:
        DataFrame: DataFrame с добавленными признаками свечей
    """
    df = df.copy()

    # Определяем цвет свечи
    df['is_green'] = df['close'] >= df['open']

    # Рассчитываем размеры свечей
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']

    # Рассчитываем размеры в процентах
    df['body_pct'] = df['body_size'] / df['close'] * 100
    df['upper_shadow_pct'] = df['upper_shadow'] / df['close'] * 100
    df['lower_shadow_pct'] = df['lower_shadow'] / df['close'] * 100
    df['total_range_pct'] = df['total_range'] / df['close'] * 100

    return df


def add_price_features(df, windows=[3, 5, 10, 20]):
    """
    Добавляет признаки, связанные с ценой

    Args:
        df (DataFrame): DataFrame с ценовыми данными
        windows (list): Список размеров окон для расчета признаков

    Returns:
        DataFrame: DataFrame с добавленными признаками цены
    """
    df = df.copy()

    # Рассчитываем изменения цен
    df['price_change'] = df['close'].diff()
    df['price_change_pct'] = df['price_change'] / df['close'].shift(1) * 100

    # Добавляем скользящие статистики для разных окон
    for window in windows:
        # Скользящие средние
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

        # Скользящие min и max
        df[f'min_{window}'] = df['low'].rolling(window).min()
        df[f'max_{window}'] = df['high'].rolling(window).max()

        # Волатильность
        df[f'std_{window}'] = df['close'].rolling(window).std()
        df[f'atr_{window}'] = talib_atr(df['high'], df['low'], df['close'], window)

        # Отношение текущей цены к скользящим статистикам
        df[f'close_to_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1
        df[f'close_to_min_{window}'] = (df['close'] - df[f'min_{window}']) / df[f'min_{window}'] * 100
        df[f'close_to_max_{window}'] = (df['close'] - df[f'max_{window}']) / df[f'max_{window}'] * 100

    return df


def talib_atr(high, low, close, timeperiod):
    """
    Рассчитывает Average True Range (ATR) без зависимости от TA-Lib

    Args:
        high (Series): Максимальные цены
        low (Series): Минимальные цены
        close (Series): Цены закрытия
        timeperiod (int): Период расчета

    Returns:
        Series: ATR
    """
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(timeperiod).mean()

    return atr


def load_and_prepare_data(symbol='ETHUSDT', interval='1h', start_date='2025-01-01', timezone='UTC'):
    """
    Загружает и подготавливает данные для анализа

    Args:
        symbol (str): Торговая пара
        interval (str): Интервал свечей
        start_date (str): Дата начала
        timezone (str): Часовой пояс

    Returns:
        DataFrame: Подготовленный DataFrame с ценовыми данными и признаками
    """
    # Загружаем данные
    raw_data = get_binance_klines(symbol, interval, start_date)

    # Преобразуем в DataFrame
    df = prepare_dataframe(raw_data, timezone)

    # Добавляем временные признаки
    df = add_time_features(df)

    # Добавляем признаки свечей
    df = add_candle_features(df)

    return df