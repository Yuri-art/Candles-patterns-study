import os
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime


def ensure_directory(directory):
    """
    Убеждается, что указанный каталог существует, создает его, если необходимо

    Args:
        directory (str): Путь к каталогу
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_dataframe_to_csv(df, filepath):
    """
    Сохраняет DataFrame в CSV файл, создавая необходимый каталог

    Args:
        df (DataFrame): DataFrame для сохранения
        filepath (str): Путь для сохранения
    """
    directory = os.path.dirname(filepath)
    ensure_directory(directory)
    df.to_csv(filepath, index=False)


def load_dataframe_from_csv(filepath, parse_dates=None):
    """
    Загружает DataFrame из CSV файла

    Args:
        filepath (str): Путь к файлу
        parse_dates (list, optional): Список столбцов для преобразования в даты

    Returns:
        DataFrame: Загруженный DataFrame или None в случае ошибки
    """
    try:
        if parse_dates:
            return pd.read_csv(filepath, parse_dates=parse_dates)
        else:
            return pd.read_csv(filepath)
    except Exception as e:
        print(f"Ошибка при загрузке {filepath}: {e}")
        return None


def save_dict_to_json(data, filepath):
    """
    Сохраняет словарь в JSON файл

    Args:
        data (dict): Словарь для сохранения
        filepath (str): Путь для сохранения
    """
    directory = os.path.dirname(filepath)
    ensure_directory(directory)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=json_serializer)


def load_dict_from_json(filepath):
    """
    Загружает словарь из JSON файла

    Args:
        filepath (str): Путь к файлу

    Returns:
        dict: Загруженный словарь или None в случае ошибки
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке {filepath}: {e}")
        return None


def json_serializer(obj):
    """
    Вспомогательная функция для сериализации объектов в JSON

    Args:
        obj: Объект для сериализации

    Returns:
        Сериализуемое представление объекта
    """
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        raise TypeError(f"Объект типа {type(obj)} не сериализуется")


def save_model(model, filepath):
    """
    Сохраняет модель в файл

    Args:
        model: Модель для сохранения
        filepath (str): Путь для сохранения
    """
    directory = os.path.dirname(filepath)
    ensure_directory(directory)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """
    Загружает модель из файла

    Args:
        filepath (str): Путь к файлу

    Returns:
        Загруженная модель или None в случае ошибки
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке модели {filepath}: {e}")
        return None


def calculate_drawdown(returns):
    """
    Рассчитывает просадку на основе ряда доходностей

    Args:
        returns (Series): Ряд доходностей

    Returns:
        tuple: (максимальная просадка, текущая просадка)
    """
    # Преобразуем проценты в коэффициенты
    wealth_index = (1 + returns / 100).cumprod()

    # Рассчитываем текущий максимум
    previous_peaks = wealth_index.cummax()

    # Рассчитываем просадку
    drawdown = (wealth_index / previous_peaks - 1) * 100

    return drawdown.min(), drawdown.iloc[-1]


def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=252):
    """
    Рассчитывает коэффициент Шарпа

    Args:
        returns (Series): Ряд доходностей в процентах
        risk_free_rate (float): Безрисковая ставка в процентах годовых
        periods_per_year (int): Количество периодов в году

    Returns:
        float: Коэффициент Шарпа
    """
    # Преобразуем проценты в десятичные дроби
    returns_decimal = returns / 100
    risk_free_rate_decimal = risk_free_rate / 100

    # Рассчитываем дневную безрисковую ставку
    daily_risk_free = ((1 + risk_free_rate_decimal) ** (1 / periods_per_year)) - 1

    # Рассчитываем избыточную доходность
    excess_returns = returns_decimal - daily_risk_free

    # Рассчитываем коэффициент Шарпа
    if excess_returns.std() == 0:
        return 0

    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def format_pct(value):
    """
    Форматирует число как процент

    Args:
        value (float): Число для форматирования

    Returns:
        str: Отформатированная строка
    """
    return f"{value:.2f}%" if value is not None else "N/A"


def format_number(value, decimals=2):
    """
    Форматирует число с указанным количеством десятичных знаков

    Args:
        value (float): Число для форматирования
        decimals (int): Количество десятичных знаков

    Returns:
        str: Отформатированная строка
    """
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def get_timestamp_str():
    """
    Возвращает текущую метку времени в формате строки для использования в именах файлов

    Returns:
        str: Текущая метка времени в формате YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_table(data, headers=None, padding=2):
    """
    Печатает данные в виде таблицы

    Args:
        data (list): Список строк данных
        headers (list, optional): Заголовки столбцов
        padding (int): Отступ между столбцами
    """
    if not data:
        return

    # Определяем заголовки, если не указаны
    if headers is None and isinstance(data[0], dict):
        headers = list(data[0].keys())
        # Преобразуем список словарей в список списков
        data = [[row.get(header, '') for header in headers] for row in data]

    # Добавляем заголовки к данным, если указаны
    if headers:
        all_data = [headers] + data
    else:
        all_data = data

    # Определяем ширину каждого столбца
    col_widths = []
    for col_idx in range(len(all_data[0])):
        max_width = max(len(str(row[col_idx])) for row in all_data)
        col_widths.append(max_width)

    # Печатаем заголовки, если указаны
    if headers:
        header_row = ' ' * padding
        for col_idx, header in enumerate(headers):
            header_row += str(header).ljust(col_widths[col_idx] + padding)
        print(header_row)

        # Печатаем разделитель
        separator = ' ' * padding
        for width in col_widths:
            separator += '-' * width + ' ' * padding
        print(separator)

    # Печатаем данные
    for row in data:
        data_row = ' ' * padding
        for col_idx, cell in enumerate(row):
            data_row += str(cell).ljust(col_widths[col_idx] + padding)
        print(data_row)


def log_progress(message, verbose=True):
    """
    Выводит сообщение о прогрессе выполнения с меткой времени

    Args:
        message (str): Сообщение для вывода
        verbose (bool): Флаг вывода сообщения
    """
    if verbose:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")


def calculate_win_rate(trades):
    """
    Рассчитывает процент выигрышных сделок

    Args:
        trades (list): Список сделок

    Returns:
        float: Процент выигрышных сделок
    """
    if not trades:
        return 0.0

    winning_trades = sum(1 for trade in trades if trade['profit_pct'] > 0)
    return (winning_trades / len(trades)) * 100