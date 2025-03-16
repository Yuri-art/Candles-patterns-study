import pandas as pd
import numpy as np
import talib
import warnings
from datetime import datetime
import os
import matplotlib.pyplot as plt
import argparse

# Импортируем созданные модули
import data_loader
import indicators
import support_resistance
import trading_strategies
import backtester
import visualization
import utils
import config

# Добавляем ATR бэктестер
# Импортируем модуль с ATR бэктестером
from atr_backtester import ATRBacktester


def parse_arguments():
    """
    Разбирает аргументы командной строки

    Returns:
        argparse.Namespace: Аргументы командной строки
    """
    parser = argparse.ArgumentParser(description='Анализ свечных паттернов и уровней')

    # Аргументы для загрузки данных
    parser.add_argument('--symbol', type=str, default=config.SYMBOL,
                        help=f'Торговая пара (по умолчанию: {config.SYMBOL})')
    parser.add_argument('--interval', type=str, default=config.INTERVAL,
                        help=f'Интервал свечей (по умолчанию: {config.INTERVAL})')
    parser.add_argument('--start_date', type=str, default=config.START_DATE,
                        help=f'Дата начала в формате YYYY-MM-DD (по умолчанию: {config.START_DATE})')

    # Аргументы для анализа уровней
    parser.add_argument('--sr_order', type=int, default=config.SR_ORDER,
                        help=f'Размер окна для поиска экстремумов (по умолчанию: {config.SR_ORDER})')
    parser.add_argument('--sr_min_touches', type=int, default=config.SR_MIN_TOUCHES,
                        help=f'Минимальное количество касаний для уровня (по умолчанию: {config.SR_MIN_TOUCHES})')

    # Аргументы для отображения индикаторов
    parser.add_argument('--show_rsi', action='store_true', default=config.INDICATORS_CONFIG.get('show_rsi', True),
                        help='Показывать RSI на графиках')
    parser.add_argument('--show_macd', action='store_true', default=config.INDICATORS_CONFIG.get('show_macd', False),
                        help='Показывать MACD на графиках')
    parser.add_argument('--show_bollinger', action='store_true',
                        default=config.INDICATORS_CONFIG.get('show_bollinger', False),
                        help='Показывать полосы Боллинджера на графиках')
    parser.add_argument('--show_sma', action='store_true', default=config.INDICATORS_CONFIG.get('show_sma', False),
                        help='Показывать скользящие средние на графиках')
    parser.add_argument('--show_atr', action='store_true', default=config.INDICATORS_CONFIG.get('show_atr', False),
                        help='Показывать ATR на графиках')

    # Аргументы для отображения пробитых уровней
    parser.add_argument('--show_broken_levels', action='store_true', default=config.SHOW_BROKEN_LEVELS,
                        help='Показывать пробитые уровни на графиках')
    parser.add_argument('--show_broken_support', action='store_true', default=config.SHOW_BROKEN_SUPPORT,
                        help='Показывать пробитые уровни поддержки на графиках')
    parser.add_argument('--show_broken_resistance', action='store_true', default=config.SHOW_BROKEN_RESISTANCE,
                        help='Показывать пробитые уровни сопротивления на графиках')

    # Аргументы для вывода
    parser.add_argument('--show_plots', action='store_true', default=config.SHOW_PLOTS,
                        help='Показывать графики во время выполнения')
    parser.add_argument('--save_plots', action='store_true', default=config.SAVE_PLOTS,
                        help='Сохранять графики в файлы')
    parser.add_argument('--verbose', action='store_true', default=config.VERBOSE,
                        help='Подробный вывод информации')

    # Добавляем аргумент для выбора режима фильтрации по тренду
    parser.add_argument('--trend_filter_mode', type=str,
                        choices=['with_trend', 'against_trend', 'ignore_trend'],
                        default=config.TREND_FILTER_MODE,
                        help=f'Режим фильтрации по тренду (по умолчанию: {config.TREND_FILTER_MODE})')

    # Аргументы для выбора типа бэктестинга
    parser.add_argument('--use_atr_backtest', action='store_true', default=config.USE_ATR_BACKTEST,
                        help='Использовать ATR бэктестер вместо стандартного')

    # Параметры для ATR бэктестера
    parser.add_argument('--atr_period', type=int, default=config.ATR_PERIOD,
                        help=f'Период для расчета ATR (по умолчанию: {config.ATR_PERIOD})')
    parser.add_argument('--atr_confirmation_multiple', type=float, default=config.ATR_CONFIRMATION_MULTIPLE,
                        help=f'Множитель ATR для подтверждения сигнала (по умолчанию: {config.ATR_CONFIRMATION_MULTIPLE})')
    parser.add_argument('--atr_stop_multiple', type=float, default=config.ATR_STOP_MULTIPLE,
                        help=f'Множитель ATR для стоп-лосса и трейлинг-стопа (по умолчанию: {config.ATR_STOP_MULTIPLE})')
    parser.add_argument('--confirmation_period', type=int, default=config.CONFIRMATION_PERIOD,
                        help=f'Количество свечей для ожидания подтверждения сигнала (по умолчанию: {config.CONFIRMATION_PERIOD})')
    parser.add_argument('--optimize_atr', action='store_true', default=config.OPTIMIZE_ATR,
                        help='Оптимизировать параметры ATR для лучших паттернов')

    parser.add_argument('--trend_lookback', type=int, default=config.TREND_LOOKBACK,
                        help=f'Период для определения тренда (по умолчанию: {config.TREND_LOOKBACK})')
    parser.add_argument('--trend_r_squared', type=float, default=config.TREND_R_SQUARED,
                        help=f'Минимальный R² для определения тренда (по умолчанию: {config.TREND_R_SQUARED})')

    return parser.parse_args()

def prepare_data_for_backtest(df):
    """
    Подготавливает данные для ATR бэктестинга

    Args:
        df (DataFrame): Исходный DataFrame с данными свечей

    Returns:
        DataFrame: Подготовленный DataFrame
    """
    # Убедимся, что у нас есть необходимые столбцы
    required_columns = ['open', 'high', 'low', 'close', 'time']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"В DataFrame отсутствует столбец: {col}")

    # Создаем копию для изменений
    result_df = df.copy()

    # Рассчитываем ATR, если его нет
    if 'atr' not in result_df.columns:
        result_df['atr'] = talib.ATR(
            result_df['high'].values,
            result_df['low'].values,
            result_df['close'].values,
            timeperiod=14
        )

    return result_df

def plot_atr_trades_with_trends_ver4(df, trades, trend_data, pattern_name=None, save_path=None, figsize=(15, 10)):
    """
    Строит график с отмеченными сделками для ATR стратегии и линиями трендов

    Args:
        df (DataFrame): DataFrame с данными свечей
        trades (list): Список сделок
        trend_data (list): Данные о трендах для визуализации
        pattern_name (str, optional): Название паттерна для заголовка
        save_path (str, optional): Путь для сохранения графика
        figsize (tuple): Размер графика

    Returns:
        Figure: Matplotlib Figure с построенным графиком
    """
    print(f"Вызов plot_atr_trades_with_trends:")
    print(f"  - df: {df.shape if df is not None else 'None'}")
    print(f"  - trades: {len(trades) if trades else 'None'}")
    print(f"  - trend_data: {len(trend_data) if trend_data else 'None'}")
    print(f"  - pattern_name: {pattern_name}")
    print(f"  - save_path: {save_path}")

    if not trades:
        print("Нет сделок для отображения")
        # Создаем простой график только с ценой
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df['time'], df['close'], color='blue', linewidth=1, label='Цена закрытия')
        ax.set_title(f"Нет сделок по паттерну {pattern_name}" if pattern_name else "Нет сделок")
        ax.set_xlabel("Время")
        ax.set_ylabel("Цена")
        ax.grid(True)
        if save_path:
            plt.savefig(save_path)
        return fig

    # Создаем DataFrame из списка сделок
    try:
        trades_df = pd.DataFrame(trades)
        print(f"Создан DataFrame сделок: {trades_df.shape}")
    except Exception as e:
        print(f"Ошибка при создании DataFrame сделок: {e}")
        trades_df = pd.DataFrame()

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=figsize)

    # Отображаем ценовой график
    try:
        ax.plot(df['time'], df['close'], color='blue', linewidth=1, label='Цена закрытия')
    except Exception as e:
        print(f"Ошибка при отображении ценового графика: {e}")

    # Отмечаем точки входа и выхода для длинных позиций
    if not trades_df.empty and 'type' in trades_df.columns:
        try:
            long_entries = trades_df[trades_df['type'] == 'long']
            if not long_entries.empty:
                print(f"Отображение {len(long_entries)} длинных позиций")
                for _, trade in long_entries.iterrows():
                    entry_idx = trade['entry_index']
                    exit_idx = trade['exit_index']

                    # Проверяем, что индексы в допустимом диапазоне
                    if 0 <= entry_idx < len(df) and 0 <= exit_idx < len(df):
                        # Точка входа (зеленый треугольник вверх)
                        ax.scatter(df.iloc[entry_idx]['time'], df.iloc[entry_idx]['close'],
                                   color='green', marker='^', s=100, label='_nolegend_')

                        # Точка выхода (зеленый круг, если прибыль, красный круг, если убыток)
                        marker_color = 'green' if trade['profit_pct'] > 0 else 'red'
                        ax.scatter(df.iloc[exit_idx]['time'], df.iloc[exit_idx]['close'],
                                   color=marker_color, marker='o', s=100, label='_nolegend_')

                        # Рисуем линию между входом и выходом
                        ax.plot([df.iloc[entry_idx]['time'], df.iloc[exit_idx]['time']],
                                [df.iloc[entry_idx]['close'], df.iloc[exit_idx]['close']],
                                color=marker_color, alpha=0.5, linestyle='--', label='_nolegend_')
        except Exception as e:
            print(f"Ошибка при отображении длинных позиций: {e}")

        # Отмечаем точки входа и выхода для коротких позиций
        try:
            short_entries = trades_df[trades_df['type'] == 'short']
            if not short_entries.empty:
                print(f"Отображение {len(short_entries)} коротких позиций")
                for _, trade in short_entries.iterrows():
                    entry_idx = trade['entry_index']
                    exit_idx = trade['exit_index']

                    # Проверяем, что индексы в допустимом диапазоне
                    if 0 <= entry_idx < len(df) and 0 <= exit_idx < len(df):
                        # Точка входа (красный треугольник вниз)
                        ax.scatter(df.iloc[entry_idx]['time'], df.iloc[entry_idx]['close'],
                                   color='red', marker='v', s=100, label='_nolegend_')

                        # Точка выхода (зеленый круг, если прибыль, красный круг, если убыток)
                        marker_color = 'green' if trade['profit_pct'] > 0 else 'red'
                        ax.scatter(df.iloc[exit_idx]['time'], df.iloc[exit_idx]['close'],
                                   color=marker_color, marker='o', s=100, label='_nolegend_')

                        # Рисуем линию между входом и выходом
                        ax.plot([df.iloc[entry_idx]['time'], df.iloc[exit_idx]['time']],
                                [df.iloc[entry_idx]['close'], df.iloc[exit_idx]['close']],
                                color=marker_color, alpha=0.5, linestyle='--', label='_nolegend_')
        except Exception as e:
            print(f"Ошибка при отображении коротких позиций: {e}")

    # Визуализация данных о трендах
    if trend_data:
        try:
            print(f"Начало отображения трендов, всего {len(trend_data)} трендов")
            trend_count = 0
            for trend in trend_data:
                # Отображаем только каждый N-й тренд, чтобы избежать перегруженности
                if trend_count % 20 != 0:
                    trend_count += 1
                    continue

                if 'data' in trend and trend['data'] is not None:
                    slope, intercept, x0, x1, price0, price1 = trend['data']

                    # Преобразуем индексы в значения времени
                    idx0 = trend['index'] - (x1 - x0)
                    idx1 = trend['index']

                    if idx0 >= 0 and idx1 < len(df):
                        time0 = df.iloc[idx0]['time']
                        time1 = df.iloc[idx1]['time']

                        # Цвет линии в зависимости от направления тренда
                        if trend['direction'] == 'up':
                            color = 'green'
                            label = f"↗ ({trend['strength']:.2f})"
                        elif trend['direction'] == 'down':
                            color = 'red'
                            label = f"↘ ({trend['strength']:.2f})"
                        else:
                            color = 'gray'
                            label = f"→ ({trend['strength']:.2f})"

                        # Рисуем линию тренда
                        ax.plot([time0, time1], [price0, price1],
                                color=color, linestyle='-', linewidth=2, alpha=0.6)

                        # Добавляем метку тренда с некоторым шагом
                        if trend_count % 100 == 0:  # Показываем не все метки
                            ax.text(time1, price1, label, color=color,
                                    fontsize=8, verticalalignment='bottom')

                trend_count += 1

            print(f"Отображено {trend_count} трендов")
        except Exception as e:
            print(f"Ошибка при отображении трендов: {e}")

    # Настройки графика
    try:
        ax.set_title(f"Сделки по паттерну {pattern_name}" if pattern_name else "Результаты торговли")
        ax.set_xlabel("Время")
        ax.set_ylabel("Цена")
        ax.grid(True)
        ax.legend()

        # Форматируем ось времени
        plt.gcf().autofmt_xdate()
    except Exception as e:
        print(f"Ошибка при настройке графика: {e}")

    # Сохраняем график, если указан путь
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"График сохранен в {save_path}")
        except Exception as e:
            print(f"Ошибка при сохранении графика: {e}")

    return fig


def plot_atr_trades_with_trends(df, trades, trend_data, pattern_signals=None, pattern_name=None, save_path=None,
                                figsize=(15, 10)):
    """
    Строит график с отмеченными сделками для ATR стратегии, линиями трендов и сигналами паттернов

    Args:
        df (DataFrame): DataFrame с данными свечей
        trades (list): Список сделок
        trend_data (list): Данные о трендах для визуализации
        pattern_signals (list, optional): Список сигналов паттернов
        pattern_name (str, optional): Название паттерна для заголовка
        save_path (str, optional): Путь для сохранения графика
        figsize (tuple): Размер графика

    Returns:
        Figure: Matplotlib Figure с построенным графиком
    """
    print(f"Вызов plot_atr_trades_with_trends:")
    print(f"  - df: {df.shape if df is not None else 'None'}")
    print(f"  - trades: {len(trades) if trades else 'None'}")
    print(f"  - trend_data: {len(trend_data) if trend_data else 'None'}")
    print(f"  - pattern_signals: {len(pattern_signals) if pattern_signals else 'None'}")
    print(f"  - pattern_name: {pattern_name}")
    print(f"  - save_path: {save_path}")

    if not trades:
        print("Нет сделок для отображения")
        # Создаем простой график только с ценой
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df['time'], df['close'], color='blue', linewidth=1, label='Цена закрытия')
        ax.set_title(f"Нет сделок по паттерну {pattern_name}" if pattern_name else "Нет сделок")
        ax.set_xlabel("Время")
        ax.set_ylabel("Цена")
        ax.grid(True)
        if save_path:
            plt.savefig(save_path)
        return fig

    # Создаем DataFrame из списка сделок
    try:
        trades_df = pd.DataFrame(trades)
        print(f"Создан DataFrame сделок: {trades_df.shape}")
    except Exception as e:
        print(f"Ошибка при создании DataFrame сделок: {e}")
        trades_df = pd.DataFrame()

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=figsize)

    # Отображаем ценовой график
    try:
        ax.plot(df['time'], df['close'], color='blue', linewidth=1, label='Цена закрытия')
    except Exception as e:
        print(f"Ошибка при отображении ценового графика: {e}")

    # Отображаем сигналы паттернов, если они предоставлены
    if pattern_signals:
        try:
            print(f"Отображение сигналов паттернов: {len(pattern_signals)}")

            # Создаем словарь для связи сигналов и сделок
            signals_to_trades = {}

            # Если есть сделки и сигналы, связываем их
            if not trades_df.empty and 'entry_index' in trades_df.columns:
                for idx, trade in trades_df.iterrows():
                    entry_idx = trade['entry_index']

                    # Ищем сигнал, который мог привести к этой сделке
                    for signal_idx, signal in enumerate(pattern_signals):
                        if signal.get('confirmation_index') == entry_idx:
                            signal_original_idx = signal.get('index')
                            if signal_original_idx is not None:
                                signals_to_trades[signal_idx] = idx
                                break

            # Отмечаем все сигналы паттернов
            for i, signal in enumerate(pattern_signals):
                signal_idx = signal.get('index')

                # Проверяем, что индекс в допустимом диапазоне
                if signal_idx is not None and 0 <= signal_idx < len(df):
                    # Определяем тип сигнала
                    signal_type = signal.get('type', 'unknown')

                    # Цвет и стиль зависят от типа сигнала
                    if signal_type == 'bull':
                        color = 'magenta'  # Фиолетовый для бычьих
                        marker = 'D'  # Ромб
                        label = 'Бычий сигнал' if 'Бычий сигнал' not in ax.get_legend_handles_labels()[
                            1] else '_nolegend_'
                    elif signal_type == 'bear':
                        color = 'darkorange'  # Оранжевый для медвежьих
                        marker = 'D'  # Ромб
                        label = 'Медвежий сигнал' if 'Медвежий сигнал' not in ax.get_legend_handles_labels()[
                            1] else '_nolegend_'
                    else:
                        color = 'gray'
                        marker = 'D'
                        label = 'Неизвестный сигнал' if 'Неизвестный сигнал' not in ax.get_legend_handles_labels()[
                            1] else '_nolegend_'

                    # Отмечаем сигнал
                    ax.scatter(df.iloc[signal_idx]['time'], df.iloc[signal_idx]['close'],
                               color=color, marker=marker, s=80, alpha=0.7, label=label)

                    # Вертикальная линия в месте сигнала
                    ax.axvline(x=df.iloc[signal_idx]['time'], color=color, linestyle=':', alpha=0.3, label='_nolegend_')

                    # Если сигнал связан с торговой сделкой, рисуем стрелку к точке входа
                    if i in signals_to_trades:
                        trade_idx = signals_to_trades[i]
                        trade = trades_df.iloc[trade_idx]
                        entry_idx = trade['entry_index']

                        if 0 <= entry_idx < len(df):
                            # Стрелка от сигнала к точке входа
                            ax.annotate('',
                                        xy=(df.iloc[entry_idx]['time'], df.iloc[entry_idx]['close']),
                                        xytext=(df.iloc[signal_idx]['time'], df.iloc[signal_idx]['close']),
                                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.6),
                                        annotation_clip=True)

                            # Линия между сигналом и входом
                            ax.plot([df.iloc[signal_idx]['time'], df.iloc[entry_idx]['time']],
                                    [df.iloc[signal_idx]['close'], df.iloc[entry_idx]['close']],
                                    color=color, alpha=0.3, linestyle=':', label='_nolegend_')
        except Exception as e:
            print(f"Ошибка при отображении сигналов паттернов: {e}")
            import traceback
            traceback.print_exc()

    # Отмечаем точки входа и выхода для длинных позиций
    if not trades_df.empty and 'type' in trades_df.columns:
        try:
            long_entries = trades_df[trades_df['type'] == 'long']
            if not long_entries.empty:
                print(f"Отображение {len(long_entries)} длинных позиций")
                for _, trade in long_entries.iterrows():
                    entry_idx = trade['entry_index']
                    exit_idx = trade['exit_index']

                    # Проверяем, что индексы в допустимом диапазоне
                    if 0 <= entry_idx < len(df) and 0 <= exit_idx < len(df):
                        # Точка входа (зеленый треугольник вверх)
                        ax.scatter(df.iloc[entry_idx]['time'], df.iloc[entry_idx]['close'],
                                   color='green', marker='^', s=100,
                                   label='Вход Long' if 'Вход Long' not in ax.get_legend_handles_labels()[
                                       1] else '_nolegend_')

                        # Точка выхода (зеленый круг, если прибыль, красный круг, если убыток)
                        marker_color = 'green' if trade['profit_pct'] > 0 else 'red'
                        ax.scatter(df.iloc[exit_idx]['time'], df.iloc[exit_idx]['close'],
                                   color=marker_color, marker='o', s=100, label='_nolegend_')

                        # Рисуем линию между входом и выходом
                        ax.plot([df.iloc[entry_idx]['time'], df.iloc[exit_idx]['time']],
                                [df.iloc[entry_idx]['close'], df.iloc[exit_idx]['close']],
                                color=marker_color, alpha=0.5, linestyle='--', label='_nolegend_')
        except Exception as e:
            print(f"Ошибка при отображении длинных позиций: {e}")

        # Отмечаем точки входа и выхода для коротких позиций
        try:
            short_entries = trades_df[trades_df['type'] == 'short']
            if not short_entries.empty:
                print(f"Отображение {len(short_entries)} коротких позиций")
                for _, trade in short_entries.iterrows():
                    entry_idx = trade['entry_index']
                    exit_idx = trade['exit_index']

                    # Проверяем, что индексы в допустимом диапазоне
                    if 0 <= entry_idx < len(df) and 0 <= exit_idx < len(df):
                        # Точка входа (красный треугольник вниз)
                        ax.scatter(df.iloc[entry_idx]['time'], df.iloc[entry_idx]['close'],
                                   color='red', marker='v', s=100,
                                   label='Вход Short' if 'Вход Short' not in ax.get_legend_handles_labels()[
                                       1] else '_nolegend_')

                        # Точка выхода (зеленый круг, если прибыль, красный круг, если убыток)
                        marker_color = 'green' if trade['profit_pct'] > 0 else 'red'
                        ax.scatter(df.iloc[exit_idx]['time'], df.iloc[exit_idx]['close'],
                                   color=marker_color, marker='o', s=100, label='_nolegend_')

                        # Рисуем линию между входом и выходом
                        ax.plot([df.iloc[entry_idx]['time'], df.iloc[exit_idx]['time']],
                                [df.iloc[entry_idx]['close'], df.iloc[exit_idx]['close']],
                                color=marker_color, alpha=0.5, linestyle='--', label='_nolegend_')
        except Exception as e:
            print(f"Ошибка при отображении коротких позиций: {e}")

    # Визуализация данных о трендах
    if trend_data:
        try:
            print(f"Начало отображения трендов, всего {len(trend_data)} трендов")
            trend_count = 0
            for trend in trend_data:
                # Отображаем только каждый N-й тренд, чтобы избежать перегруженности
                if trend_count % 20 != 0:
                    trend_count += 1
                    continue

                if 'data' in trend and trend['data'] is not None:
                    slope, intercept, x0, x1, price0, price1 = trend['data']

                    # Преобразуем индексы в значения времени
                    idx0 = trend['index'] - (x1 - x0)
                    idx1 = trend['index']

                    if idx0 >= 0 and idx1 < len(df):
                        time0 = df.iloc[idx0]['time']
                        time1 = df.iloc[idx1]['time']

                        # Цвет линии в зависимости от направления тренда
                        if trend['direction'] == 'up':
                            color = 'green'
                            label = f"↗ ({trend['strength']:.2f})"
                        elif trend['direction'] == 'down':
                            color = 'red'
                            label = f"↘ ({trend['strength']:.2f})"
                        else:
                            color = 'gray'
                            label = f"→ ({trend['strength']:.2f})"

                        # Рисуем линию тренда
                        ax.plot([time0, time1], [price0, price1],
                                color=color, linestyle='-', linewidth=2, alpha=0.6)

                        # Добавляем метку тренда с некоторым шагом
                        if trend_count % 100 == 0:  # Показываем не все метки
                            ax.text(time1, price1, label, color=color,
                                    fontsize=8, verticalalignment='bottom')

                trend_count += 1

            print(f"Отображено {trend_count} трендов")
        except Exception as e:
            print(f"Ошибка при отображении трендов: {e}")

    # Настройки графика
    try:
        ax.set_title(f"Сделки по паттерну {pattern_name}" if pattern_name else "Результаты торговли")
        ax.set_xlabel("Время")
        ax.set_ylabel("Цена")
        ax.grid(True)

        # Создаем легенду с объяснением обозначений
        handles, labels = ax.get_legend_handles_labels()

        # Добавляем объяснения для выходов с прибылью и убытком, если их нет
        if 'Выход с прибылью' not in labels:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10))
            labels.append('Выход с прибылью')
        if 'Выход с убытком' not in labels:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10))
            labels.append('Выход с убытком')

        ax.legend(handles, labels, loc='best')

        # Форматируем ось времени
        plt.gcf().autofmt_xdate()
    except Exception as e:
        print(f"Ошибка при настройке графика: {e}")

    # Сохраняем график, если указан путь
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"График сохранен в {save_path}")
        except Exception as e:
            print(f"Ошибка при сохранении графика: {e}")

    return fig

def plot_atr_equity_curve(trades, title=None, save_path=None, figsize=(12, 8)):
    """
    Строит кривую доходности на основе истории сделок

    Args:
        trades (list): Список сделок
        title (str, optional): Заголовок графика
        save_path (str, optional): Путь для сохранения графика
        figsize (tuple): Размер графика

    Returns:
        Figure: Matplotlib Figure с построенным графиком
    """
    if not trades:
        return None

    # Создаем DataFrame из списка сделок
    trades_df = pd.DataFrame(trades)

    # Сортируем по времени выхода
    trades_df = trades_df.sort_values('exit_time')

    # Создаем фигуру с двумя графиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})

    # Рассчитываем кумулятивную доходность
    trades_df['equity'] = (1 + trades_df['profit_pct'] / 100).cumprod()

    # Строим кривую доходности на верхнем графике
    ax1.plot(range(len(trades_df)), trades_df['equity'], color='blue', linewidth=2)
    ax1.axhline(y=1, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(title or "Кривая доходности")
    ax1.set_ylabel("Доходность (мультипликатор)")
    ax1.grid(True)

    # Рассчитываем просадку
    trades_df['drawdown'] = trades_df['equity'].div(trades_df['equity'].cummax()).sub(1).mul(100)

    # Добавляем информацию о статистике
    total_trades = len(trades_df)
    win_rate = (trades_df['profit_pct'] > 0).mean() * 100
    final_return = (trades_df['equity'].iloc[-1] - 1) * 100
    max_drawdown = trades_df['drawdown'].min()

    stats_text = (f"Всего сделок: {total_trades}\n"
                  f"Выигрышных: {win_rate:.2f}%\n"
                  f"Итоговая доходность: {final_return:.2f}%\n"
                  f"Макс. просадка: {max_drawdown:.2f}%")

    ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Строим график просадок на нижнем графике
    ax2.plot(range(len(trades_df)), trades_df['drawdown'], color='red', linewidth=1.5)
    ax2.fill_between(range(len(trades_df)), 0, trades_df['drawdown'], color='red', alpha=0.3)
    ax2.set_ylabel("Просадка (%)")
    ax2.set_ylim(min(trades_df['drawdown'].min() * 1.5, -5), 5)  # Оставляем немного места вверху
    ax2.grid(True)

    # Закрываем пространство между графиками
    plt.tight_layout()

    # Сохраняем график, если указан путь
    if save_path:
        plt.savefig(save_path)

    return fig

def run_atr_backtest_VER4(df, pattern_names, args, output_folders):
    """
    Запускает бэктестинг с использованием ATR бэктестера

    Args:
        df (DataFrame): DataFrame с данными
        pattern_names (list): Список имен паттернов
        args (Namespace): Аргументы командной строки
        output_folders (dict): Словарь с путями к папкам вывода

    Returns:
        dict: Результаты бэктестинга
    """
    #utils.log_progress("Запуск ATR бэктестинга...", args.verbose)
    utils.log_progress(f"Запуск ATR бэктестинга с режимом фильтрации тренда: {args.trend_filter_mode}", args.verbose)

    # Подготавливаем данные для ATR бэктестинга
    prepared_df = prepare_data_for_backtest(df)

    # Параметры для ATR бэктестера, используем значения из конфигурации
   # atr_params = {
   #     'atr_period': args.atr_period,
   #     'atr_confirmation_multiple': args.atr_confirmation_multiple,
   #     'atr_stop_multiple': args.atr_stop_multiple,
   #     'confirmation_period': args.confirmation_period
   # }
    atr_params = {
        'atr_period': args.atr_period,
        'atr_confirmation_multiple': args.atr_confirmation_multiple,
        'atr_stop_multiple': args.atr_stop_multiple,
        'confirmation_period': args.confirmation_period,
        'trend_lookback': getattr(args, 'trend_lookback', config.TREND_LOOKBACK),
        'trend_r_squared': getattr(args, 'trend_r_squared', config.TREND_R_SQUARED),
        'trend_filter_mode': args.trend_filter_mode
    }

    # Добавляем параметры тренда, если они существуют
    if hasattr(args, 'trend_lookback'):
        atr_params['trend_lookback'] = args.trend_lookback
    else:
        atr_params['trend_lookback'] = 20  # Значение по умолчанию

    if hasattr(args, 'trend_r_squared'):
        atr_params['trend_r_squared'] = args.trend_r_squared
    else:
        atr_params['trend_r_squared'] = 0.6  # Значение по умолчанию

    # Создаем экземпляр ATR бэктестера
    backtester = ATRBacktester(**atr_params)

    # Тестирование расчета трендов
    utils.log_progress("Тестирование расчета трендов...", args.verbose)
    backtester.test_trend_calculation(prepared_df)

    # Создаем папку для результатов ATR бэктестинга, используя переменную из конфигурации
    atr_output_folder = f"{output_folders['backtest']}/{os.path.basename(config.ATR_OUTPUT_FOLDER)}"
    os.makedirs(atr_output_folder, exist_ok=True)

    # Запускаем бэктестинг для всех паттернов
    all_results = []

    # Тестирование расчета трендов
    utils.log_progress("Тестирование расчета трендов...", args.verbose)
    backtester.test_trend_calculation(prepared_df)

    for pattern in pattern_names:
        pattern_data = df[df[pattern] != 0]

        if not pattern_data.empty:
            utils.log_progress(f"ATR бэктестинг для паттерна {pattern}...", args.verbose)

            # Запускаем бэктестинг
            stats = backtester.test_pattern(prepared_df, pattern)

            # Сохраняем информацию о всех сигналах
            if 'all_signals' in stats:
                all_signals_df = pd.DataFrame(stats['all_signals'])
                all_signals_df.to_csv(f"{atr_output_folder}/{pattern}_all_signals.csv", index=False)


            # Проверка наличия трендов в результате
            print(f"Ключи в stats: {list(stats.keys())}")
            print(f"Тренды в результате: {'trend_data' in stats}")
            if 'trend_data' in stats:
                print(f"Размер trend_data в результате: {len(stats['trend_data'])}")
            else:
                print("ПРЕДУПРЕЖДЕНИЕ: trend_data отсутствует в результатах!")

            # Добавляем информацию о паттерне
            stats_with_pattern = stats.copy()
            stats_with_pattern['pattern'] = pattern
            all_results.append(stats_with_pattern)

            # Если есть сделки, строим графики и сохраняем результаты
            if stats['trades']:
                # Сохраняем список сделок
                trades_df = pd.DataFrame(stats['trades'])
                trades_df.to_csv(f"{atr_output_folder}/{pattern}_trades.csv", index=False)

                # Строим график сделок
                if args.save_plots:
                    try:
                        # Явное извлечение trend_data с проверкой
                        if 'trend_data' not in stats:
                            print("ВНИМАНИЕ: trend_data отсутствует в результатах!")
                            trend_data = []
                        else:
                            trend_data = stats['trend_data']
                            print(f"Получены данные о трендах: {len(trend_data)} записей")

                        # Получаем сигналы для паттерна
                        pattern_signals = []
                        if 'all_signals' in stats:
                            pattern_signals = [s for s in stats['all_signals'] if s.get('pattern') == pattern_name]
                            print(
                                f"Получены данные о сигналах: {len(pattern_signals)} сигналов паттерна {pattern_name}")

                        fig = plot_atr_trades_with_trends(
                            prepared_df,
                            stats['trades'],
                            trend_data,
                            pattern_signals=pattern_signals,  # Передаем сигналы
                            pattern_name=pattern_name,
                            save_path=f"{atr_output_folder}/{pattern_name}_trades.png"
                        )
                        if args.show_plots:
                            plt.show()
                        else:
                            plt.close(fig)
                    except Exception as e:
                        print(f"Ошибка при построении графика сделок: {e}")
                        # Создаем упрощенный график в случае ошибки
                        try:
                            fig, ax = plt.subplots(figsize=(15, 10))
                            ax.plot(prepared_df['time'], prepared_df['close'], color='blue')
                            plt.title(f"Сделки по паттерну {pattern} (упрощенный график)")
                            plt.savefig(f"{atr_output_folder}/{pattern}_trades_simple.png")
                            plt.close(fig)
                        except Exception as e2:
                            print(f"Ошибка при построении упрощенного графика: {e2}")

                # Строим кривую доходности
                if args.save_plots:
                    try:
                        equity_fig = plot_atr_equity_curve(
                            stats['trades'],
                            title=f"Кривая доходности для паттерна {pattern}",
                            save_path=f"{atr_output_folder}/{pattern}_equity.png"
                        )
                        if args.show_plots:
                            plt.show()
                        else:
                            plt.close(equity_fig)
                    except Exception as e:
                        print(f"Ошибка при построении кривой доходности: {e}")

    # Создаем сводную таблицу результатов
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.sort_values('total_profit', ascending=False)

        # Перемещаем столбец 'pattern' в начало DataFrame
        # Получаем список всех столбцов
        cols = summary_df.columns.tolist()

        # Удаляем 'pattern' из списка
        cols.remove('pattern')

        # Добавляем 'pattern' в начало списка
        cols = ['pattern'] + cols

        # Переупорядочиваем столбцы в DataFrame
        summary_df = summary_df[cols]

        # Сохраняем сводную таблицу
        summary_df.to_csv(f"{atr_output_folder}/pattern_stats_summary.csv", index=False)

        # Определяем топ паттерны
        top_patterns = summary_df[summary_df['win_rate'] > 35].sort_values('total_profit', ascending=False)

        if not top_patterns.empty:
            top_patterns.to_csv(f"{atr_output_folder}/top_patterns.csv", index=False)

            # Оптимизация параметров для лучших паттернов, если включена
            if args.optimize_atr and len(top_patterns) > 0:
                utils.log_progress("Оптимизация параметров ATR...", args.verbose)

                # Определяем сетку параметров для оптимизации
                param_grid = {
                    'atr_period': [10, 14, 20],
                    'atr_confirmation_multiple': [1.0, 1.5, 2.0],
                    'atr_stop_multiple': [1.0, 1.5, 2.0],
                    'confirmation_period': [3, 5, 7]
                }

                # Оптимизируем параметры для топ-3 паттернов (или меньше, если их меньше)
                for idx, row in top_patterns.head(min(3, len(top_patterns))).iterrows():
                    pattern = row['pattern']
                    utils.log_progress(f"Оптимизация для паттерна {pattern}...", args.verbose)

                    # Запускаем оптимизацию
                    opt_results = backtester.optimize_parameters(prepared_df, pattern, param_grid)

                    # Сохраняем результаты оптимизации
                    opt_results['all_results'].to_csv(f"{atr_output_folder}/{pattern}_optimization.csv", index=False)

                    # Выводим лучшие параметры
                    utils.log_progress(f"Лучшие параметры для паттерна {pattern}:", args.verbose)
                    for param, value in opt_results['best_params'].items():
                        print(f"  {param}: {value}")

                    print(f"  Прибыль: {utils.format_pct(opt_results['best_stats']['total_profit'])}")
                    print(f"  Винрейт: {utils.format_pct(opt_results['best_stats']['win_rate'])}")
                    print(f"  Сделок: {opt_results['best_stats']['total_trades']}")

                    # Запускаем бэктестинг с оптимальными параметрами
                    optimal_backtester = ATRBacktester(**opt_results['best_params'])
                    optimal_stats = optimal_backtester.test_pattern(prepared_df, pattern)

                    # Сохраняем результаты с оптимальными параметрами
                    if optimal_stats['trades']:
                        # Сохраняем список сделок
                        opt_trades_df = pd.DataFrame(optimal_stats['trades'])
                        opt_trades_df.to_csv(f"{atr_output_folder}/{pattern}_optimal_trades.csv", index=False)

                        # Строим график сделок с оптимальными параметрами
                        if args.save_plots:
                            try:
                                # Проверяем наличие данных о трендах
                                trend_data = optimal_stats.get('trend_data', [])

                                fig = plot_atr_trades_with_trends(
                                    prepared_df,
                                    optimal_stats['trades'],
                                    trend_data,
                                    pattern_name=f"{pattern} (optimal)",
                                    save_path=f"{atr_output_folder}/{pattern}_optimal_trades.png"
                                )
                                if args.show_plots:
                                    plt.show()
                                else:
                                    plt.close(fig)
                            except Exception as e:
                                print(f"Ошибка при построении графика оптимальных сделок: {e}")

                        # Строим кривую доходности с оптимальными параметрами
                        if args.save_plots:
                            try:
                                equity_fig = plot_atr_equity_curve(
                                    optimal_stats['trades'],
                                    title=f"Кривая доходности для паттерна {pattern} (optimal)",
                                    save_path=f"{atr_output_folder}/{pattern}_optimal_equity.png"
                                )
                                if args.show_plots:
                                    plt.show()
                                else:
                                    plt.close(equity_fig)
                            except Exception as e:
                                print(f"Ошибка при построении кривой доходности: {e}")

        # Выводим топ-5 паттернов по прибыли
        utils.log_progress("Топ-5 паттернов по общей прибыли:", args.verbose)
        top5 = summary_df.head(5)
        for idx, row in top5.iterrows():
            print(
                f"{row['pattern']}: {utils.format_pct(row['total_profit'])} (n={row['total_trades']}, win={utils.format_pct(row['win_rate'])})")

        return summary_df

    return None


def run_atr_backtest(df, m15_df, pattern_names, args, output_folders):
    """
    Запускает бэктестинг с использованием ATR бэктестера

    Args:
        df (DataFrame): DataFrame с данными
        pattern_names (list): Список имен паттернов
        args (Namespace): Аргументы командной строки
        output_folders (dict): Словарь с путями к папкам вывода

    Returns:
        dict: Результаты бэктестинга
    """
    #utils.log_progress("Запуск ATR бэктестинга...", args.verbose)
    utils.log_progress(f"Запуск ATR бэктестинга с режимом фильтрации тренда: {args.trend_filter_mode}", args.verbose)
    utils.log_progress(f"Запуск ATR бэктестинга с использованием 15m для исполнения...", args.verbose)

    # Подготавливаем данные для ATR бэктестинга
    prepared_df = prepare_data_for_backtest(df)

    # Параметры для ATR бэктестера, используем значения из конфигурации
   # atr_params = {
   #     'atr_period': args.atr_period,
   #     'atr_confirmation_multiple': args.atr_confirmation_multiple,
   #     'atr_stop_multiple': args.atr_stop_multiple,
   #     'confirmation_period': args.confirmation_period
   # }
    atr_params = {
        'atr_period': args.atr_period,
        'atr_confirmation_multiple': args.atr_confirmation_multiple,
        'atr_stop_multiple': args.atr_stop_multiple,
        'confirmation_period': args.confirmation_period,
        'trend_lookback': args.trend_lookback,
        'trend_r_squared': args.trend_r_squared,
        'trend_filter_mode': args.trend_filter_mode  # Используем значение из аргументов
    }

    # Добавляем параметры тренда, если они существуют
    if hasattr(args, 'trend_lookback'):
        atr_params['trend_lookback'] = args.trend_lookback
    else:
        atr_params['trend_lookback'] = config.TREND_LOOKBACK

    if hasattr(args, 'trend_r_squared'):
        atr_params['trend_r_squared'] = args.trend_r_squared
    else:
        atr_params['trend_r_squared'] = config.TREND_R_SQUARED

    # Создаем экземпляр ATR бэктестера
    backtester = ATRBacktester(**atr_params)
    backtester.main_df = prepared_df  # Добавляем также основной DataFrame для удобства

    # Передаем дополнительно m15_df
    backtester.m15_df = m15_df

    # Тестирование расчета трендов
    utils.log_progress("Тестирование расчета трендов...", args.verbose)
    backtester.test_trend_calculation(prepared_df)

    # Создаем папку для результатов ATR бэктестинга, используя переменную из конфигурации
    atr_output_folder = f"{output_folders['backtest']}/{os.path.basename(config.ATR_OUTPUT_FOLDER)}"
    os.makedirs(atr_output_folder, exist_ok=True)

    # Запускаем бэктестинг для всех паттернов
    all_results = []

    # Тестирование расчета трендов
    utils.log_progress("Тестирование расчета трендов...", args.verbose)
    backtester.test_trend_calculation(prepared_df)

    for pattern in pattern_names:
        pattern_data = df[df[pattern] != 0]

        if not pattern_data.empty:
            utils.log_progress(f"ATR бэктестинг для паттерна {pattern}...", args.verbose)

            # Запускаем бэктестинг
            stats = backtester.test_pattern(prepared_df, pattern)

            # Сохраняем информацию о всех сигналах
            if 'all_signals' in stats:
                all_signals_df = pd.DataFrame(stats['all_signals'])
                all_signals_df.to_csv(f"{atr_output_folder}/{pattern}_all_signals.csv", index=False)


            # Проверка наличия трендов в результате
            print(f"Ключи в stats: {list(stats.keys())}")
            print(f"Тренды в результате: {'trend_data' in stats}")
            if 'trend_data' in stats:
                print(f"Размер trend_data в результате: {len(stats['trend_data'])}")
            else:
                print("ПРЕДУПРЕЖДЕНИЕ: trend_data отсутствует в результатах!")

            # Добавляем информацию о паттерне
            stats_with_pattern = stats.copy()
            stats_with_pattern['pattern'] = pattern
            all_results.append(stats_with_pattern)

            # Если есть сделки, строим графики и сохраняем результаты
            if stats['trades']:
                # Сохраняем список сделок
                trades_df = pd.DataFrame(stats['trades'])
                trades_df.to_csv(f"{atr_output_folder}/{pattern}_trades.csv", index=False)

                # Строим график сделок
                if args.save_plots:
                    try:
                        # Явное извлечение trend_data с проверкой
                        if 'trend_data' not in stats:
                            print("ВНИМАНИЕ: trend_data отсутствует в результатах!")
                            trend_data = []
                        else:
                            trend_data = stats['trend_data']
                            print(f"Получены данные о трендах: {len(trend_data)} записей")

                        # Получаем данные о сигналах паттернов
                        pattern_signals = None
                        if 'all_signals' in stats:
                            pattern_signals = [s for s in stats['all_signals'] if s.get('pattern') == pattern]
                            print(f"Найдено {len(pattern_signals)} сигналов паттерна {pattern}")

                        fig = plot_atr_trades_with_trends(
                            prepared_df,
                            stats['trades'],
                            trend_data,  # Передаем данные о трендах
                            pattern_signals=pattern_signals,  # Добавляем этот параметр
                            pattern_name=pattern,
                            save_path=f"{atr_output_folder}/{pattern}_trades.png"
                        )
                        if args.show_plots:
                            plt.show()
                        else:
                            plt.close(fig)
                    except Exception as e:
                        print(f"Ошибка при построении графика сделок: {e}")
                        # Создаем упрощенный график в случае ошибки
                        try:
                            fig, ax = plt.subplots(figsize=(15, 10))
                            ax.plot(prepared_df['time'], prepared_df['close'], color='blue')
                            plt.title(f"Сделки по паттерну {pattern} (упрощенный график)")
                            plt.savefig(f"{atr_output_folder}/{pattern}_trades_simple.png")
                            plt.close(fig)
                        except Exception as e2:
                            print(f"Ошибка при построении упрощенного графика: {e2}")

                # Строим кривую доходности
                if args.save_plots:
                    try:
                        equity_fig = plot_atr_equity_curve(
                            stats['trades'],
                            title=f"Кривая доходности для паттерна {pattern}",
                            save_path=f"{atr_output_folder}/{pattern}_equity.png"
                        )
                        if args.show_plots:
                            plt.show()
                        else:
                            plt.close(equity_fig)
                    except Exception as e:
                        print(f"Ошибка при построении кривой доходности: {e}")

    # Создаем сводную таблицу результатов
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.sort_values('total_profit', ascending=False)

        # Перемещаем столбец 'pattern' в начало DataFrame
        # Получаем список всех столбцов
        cols = summary_df.columns.tolist()

        # Удаляем 'pattern' из списка
        cols.remove('pattern')

        # Добавляем 'pattern' в начало списка
        cols = ['pattern'] + cols

        # Переупорядочиваем столбцы в DataFrame
        summary_df = summary_df[cols]

        # Сохраняем сводную таблицу
        summary_df.to_csv(f"{atr_output_folder}/pattern_stats_summary.csv", index=False)

        # Определяем топ паттерны
        top_patterns = summary_df[summary_df['win_rate'] > 35].sort_values('total_profit', ascending=False)

        if not top_patterns.empty:
            top_patterns.to_csv(f"{atr_output_folder}/top_patterns.csv", index=False)

            # Оптимизация параметров для лучших паттернов, если включена
            if args.optimize_atr and len(top_patterns) > 0:
                utils.log_progress("Оптимизация параметров ATR...", args.verbose)

                # Определяем сетку параметров для оптимизации
                param_grid = {
                    'atr_period': [10, 14, 20],
                    'atr_confirmation_multiple': [1.0, 1.5, 2.0],
                    'atr_stop_multiple': [1.0, 1.5, 2.0],
                    'confirmation_period': [3, 5, 7]
                }

                # Оптимизируем параметры для топ-3 паттернов (или меньше, если их меньше)
                for idx, row in top_patterns.head(min(3, len(top_patterns))).iterrows():
                    pattern = row['pattern']
                    utils.log_progress(f"Оптимизация для паттерна {pattern}...", args.verbose)

                    # Запускаем оптимизацию
                    opt_results = backtester.optimize_parameters(prepared_df, pattern, param_grid)

                    # Сохраняем результаты оптимизации
                    opt_results['all_results'].to_csv(f"{atr_output_folder}/{pattern}_optimization.csv", index=False)

                    # Выводим лучшие параметры
                    utils.log_progress(f"Лучшие параметры для паттерна {pattern}:", args.verbose)
                    for param, value in opt_results['best_params'].items():
                        print(f"  {param}: {value}")

                    print(f"  Прибыль: {utils.format_pct(opt_results['best_stats']['total_profit'])}")
                    print(f"  Винрейт: {utils.format_pct(opt_results['best_stats']['win_rate'])}")
                    print(f"  Сделок: {opt_results['best_stats']['total_trades']}")

                    # Запускаем бэктестинг с оптимальными параметрами
                    optimal_backtester = ATRBacktester(**opt_results['best_params'])
                    optimal_stats = optimal_backtester.test_pattern(prepared_df, pattern)

                    # Сохраняем результаты с оптимальными параметрами
                    if optimal_stats['trades']:
                        # Сохраняем список сделок
                        opt_trades_df = pd.DataFrame(optimal_stats['trades'])
                        opt_trades_df.to_csv(f"{atr_output_folder}/{pattern}_optimal_trades.csv", index=False)

                        # Строим график сделок с оптимальными параметрами
                        if args.save_plots:
                            try:
                                # Проверяем наличие данных о трендах
                                trend_data = optimal_stats.get('trend_data', [])

                                fig = plot_atr_trades_with_trends(
                                    prepared_df,
                                    optimal_stats['trades'],
                                    trend_data,
                                    pattern_name=f"{pattern} (optimal)",
                                    save_path=f"{atr_output_folder}/{pattern}_optimal_trades.png"
                                )
                                if args.show_plots:
                                    plt.show()
                                else:
                                    plt.close(fig)
                            except Exception as e:
                                print(f"Ошибка при построении графика оптимальных сделок: {e}")

                        # Строим кривую доходности с оптимальными параметрами
                        if args.save_plots:
                            try:
                                equity_fig = plot_atr_equity_curve(
                                    optimal_stats['trades'],
                                    title=f"Кривая доходности для паттерна {pattern} (optimal)",
                                    save_path=f"{atr_output_folder}/{pattern}_optimal_equity.png"
                                )
                                if args.show_plots:
                                    plt.show()
                                else:
                                    plt.close(equity_fig)
                            except Exception as e:
                                print(f"Ошибка при построении кривой доходности: {e}")

        # Выводим топ-5 паттернов по прибыли
        utils.log_progress("Топ-5 паттернов по общей прибыли:", args.verbose)
        top5 = summary_df.head(5)
        for idx, row in top5.iterrows():
            print(
                f"{row['pattern']}: {utils.format_pct(row['total_profit'])} (n={row['total_trades']}, win={utils.format_pct(row['win_rate'])})")

        return summary_df

    return None



def main():
    # Разбираем аргументы командной строки
    args = parse_arguments()

    # Отключаем предупреждения
    warnings.filterwarnings("ignore")

    # Логируем начало выполнения
    utils.log_progress(f"Запуск анализа для {args.symbol} на интервале {args.interval}", args.verbose)
    #utils.log_progress(f"Запуск анализа для {args.symbol} на интервалах 1h и 15m", args.verbose)

    # Загружаем данные
    utils.log_progress("Загрузка данных...", args.verbose)
    raw_data = data_loader.get_binance_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date
    )


    if not raw_data:
        print("Не удалось получить данные с API. Завершение работы.")
        return

    # Подготавливаем DataFrame
    utils.log_progress("Подготовка данных...", args.verbose)
    crypto = data_loader.prepare_dataframe(raw_data, timezone=config.TIMEZONE)
    crypto = data_loader.add_time_features(crypto)
    crypto = data_loader.add_candle_features(crypto)


    # Дополнительно загружаем данные для 15-минутного таймфрейма
    m15_raw_data = data_loader.get_binance_klines(
        symbol=args.symbol,
        interval='15m',
        start_date=args.start_date
    )

    # Подготавливаем DataFrame для 15m
    m15_df = data_loader.prepare_dataframe(m15_raw_data, timezone=config.TIMEZONE)
    m15_df = data_loader.add_time_features(m15_df)
    m15_df = data_loader.add_candle_features(m15_df)

    # Добавляем ATR к 15m для расчетов трейлинг-стопа
    m15_df['atr'] = talib.ATR(m15_df['high'], m15_df['low'], m15_df['close'], timeperiod=14)

    # Добавляем свечные паттерны
    utils.log_progress("Добавление свечных паттернов...", args.verbose)
    crypto, candle_names = indicators.add_candlestick_patterns(crypto)

    # Добавляем технические индикаторы
    utils.log_progress("Добавление технических индикаторов...", args.verbose)
    crypto = indicators.add_technical_indicators(crypto)

    # Находим уровни поддержки и сопротивления
    utils.log_progress("Поиск уровней поддержки и сопротивления...", args.verbose)
    support_levels_df, resistance_levels_df = support_resistance.find_extreme_levels(
        crypto,
        order=args.sr_order,
        min_touches=args.sr_min_touches
    )

    # Кластеризуем близкие уровни
    if not support_levels_df.empty:
        support_levels_df = support_resistance.cluster_levels(support_levels_df, tolerance=config.SR_TOLERANCE)
    if not resistance_levels_df.empty:
        resistance_levels_df = support_resistance.cluster_levels(resistance_levels_df, tolerance=config.SR_TOLERANCE)

    # Создаем папки для вывода
    output_folders = visualization.setup_output_folders()

    # Сохраняем информацию об уровнях
    if not support_levels_df.empty:
        utils.save_dataframe_to_csv(support_levels_df, f"{output_folders['levels']}/support_levels.csv")
    if not resistance_levels_df.empty:
        utils.save_dataframe_to_csv(resistance_levels_df, f"{output_folders['levels']}/resistance_levels.csv")

    # Находим пробитые уровни поддержки и сопротивления
    broken_support = support_resistance.identify_broken_levels(
        crypto,
        support_levels_df,
        level_type='support',
        threshold_pct=config.SR_BREAKOUT_THRESHOLD,
        consecutive_bars=config.SR_CONSECUTIVE_BARS
    )

    broken_resistance = support_resistance.identify_broken_levels(
        crypto,
        resistance_levels_df,
        level_type='resistance',
        threshold_pct=config.SR_BREAKOUT_THRESHOLD,
        consecutive_bars=config.SR_CONSECUTIVE_BARS
    )

    # Сохраняем информацию о пробитых уровнях
    if not broken_support.empty:
        utils.save_dataframe_to_csv(broken_support, f"{output_folders['levels']}/broken_support_levels.csv")
    if not broken_resistance.empty:
        utils.save_dataframe_to_csv(broken_resistance, f"{output_folders['levels']}/broken_resistance_levels.csv")


    # Формируем настройки индикаторов для визуализации на основе аргументов командной строки
    indicators_config = {
        'show_rsi': args.show_rsi,
        'show_macd': args.show_macd,
        'show_bollinger': args.show_bollinger,
        'show_sma': args.show_sma,
        'show_atr': args.show_atr
    }

    # Если выбран ATR бэктестинг, запускаем его
    if args.use_atr_backtest:
        #atr_results = run_atr_backtest(crypto, candle_names, args, output_folders)
        atr_results = run_atr_backtest(crypto, m15_df, candle_names, args, output_folders)
    else:
        # Анализируем каждый паттерн
        utils.log_progress("Анализ свечных паттернов...", args.verbose)
        for candle in candle_names:
            pattern_data = crypto[crypto[candle] != 0]

            if not pattern_data.empty:
                # Рассчитываем статистику
                stats = backtester.calculate_pattern_stats(pattern_data, crypto, candle)

                # Строим и сохраняем график
                utils.log_progress(f"Обработка паттерна {candle}...", args.verbose)

                fig = visualization.plot_candlestick_with_patterns(
                    crypto,
                    pattern_data,
                    candle,
                    support_levels=support_levels_df,
                    resistance_levels=resistance_levels_df,
                    broken_support=broken_support,
                    broken_resistance=broken_resistance,
                    show_broken_levels=args.show_broken_levels,
                    show_broken_support=args.show_broken_support,
                    show_broken_resistance=args.show_broken_resistance,
                    stats=stats,
                    save_path=f"{output_folders['charts']}/{candle}.png" if args.save_plots else None,
                    figsize=config.PLOT_FIGSIZE,
                    indicators_config=indicators_config
                )

                if args.show_plots:
                    plt.show()
                else:
                    plt.close(fig)

        # Сохраняем статистику по паттернам
        utils.log_progress("Сохранение статистики по паттернам...", args.verbose)
        summary_df, reliable_patterns = backtester.save_pattern_statistics(
            crypto,
            candle_names,
            output_folders['stats'],
            reliable_threshold=config.RELIABLE_PATTERN_THRESHOLD
        )

        # Выводим топ-5 бычьих и медвежьих паттернов
        if not summary_df.empty:
            utils.log_progress("Топ-5 бычьих паттернов по потенциальной прибыли:", args.verbose)
            bull_top = summary_df.dropna(subset=['bull_high_avg']).sort_values('bull_high_avg', ascending=False).head(5)
            for idx, row in bull_top.iterrows():
                print(f"{row['pattern']}: {utils.format_pct(row['bull_high_avg'])} (n={row['bull_count']})")

            utils.log_progress("Топ-5 медвежьих паттернов по потенциальному снижению:", args.verbose)
            bear_top = summary_df.dropna(subset=['bear_low_avg']).sort_values('bear_low_avg', ascending=True).head(5)
            for idx, row in bear_top.iterrows():
                print(f"{row['pattern']}: {utils.format_pct(row['bear_low_avg'])} (n={row['bear_count']})")

        # Тестирование стратегии на основе паттернов
        if reliable_patterns is not None and not reliable_patterns.empty:
            utils.log_progress("Тестирование торговой стратегии...", args.verbose)

            # Создаем стратегию на основе надежных паттернов
            bull_patterns = reliable_patterns[~reliable_patterns['bull_high_avg'].isna()]['pattern'].tolist()
            bear_patterns = reliable_patterns[~reliable_patterns['bear_low_avg'].isna()]['pattern'].tolist()

            if bull_patterns or bear_patterns:
                strategy = trading_strategies.CandlePatternStrategy(
                    pattern_names=bull_patterns + bear_patterns,
                    exit_strategy=config.EXIT_STRATEGY,
                    stop_loss_pct=config.STOP_LOSS,
                    take_profit_pct=config.TAKE_PROFIT
                )

                # Выполняем бэктестинг
                results = backtester.evaluate_strategy(strategy, crypto)

                # Выводим результаты
                utils.log_progress("Результаты тестирования стратегии:", args.verbose)
                print(f"Всего сделок: {results['total_trades']}")
                print(f"Процент выигрышных сделок: {utils.format_pct(results['win_rate'])}")
                print(f"Средняя прибыль на сделку: {utils.format_pct(results['avg_profit'])}")
                print(f"Общая прибыль: {utils.format_pct(results['total_profit'])}")
                print(f"Максимальная просадка: {utils.format_pct(results['max_drawdown'])}")
                print(f"Коэффициент Шарпа: {utils.format_number(results['sharpe_ratio'])}")

                # Сохраняем результаты бэктестинга
                if results['trades']:
                    trades_df = pd.DataFrame(results['trades'])
                    utils.save_dataframe_to_csv(trades_df, f"{output_folders['backtest']}/trades.csv")

                    # Строим график доходности
                    fig = visualization.plot_equity_curve(
                        trades_df,
                        strategy_name=strategy.name,
                        save_path=f"{output_folders['backtest']}/equity_curve.png" if args.save_plots else None
                    )

                    if args.show_plots:
                        plt.show()
                    else:
                        plt.close(fig)

    utils.log_progress("Анализ завершен!", args.verbose)


if __name__ == "__main__":
    main()

