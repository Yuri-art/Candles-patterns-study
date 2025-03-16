import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import os
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec


def setup_output_folders():
    """
    Создает папки для вывода результатов

    Returns:
        dict: Словарь с путями к созданным папкам
    """
    folders = {
        'charts': 'output/charts',
        'stats': 'output/stats',
        'levels': 'output/levels',
        'backtest': 'output/backtest',
    }

    for folder in folders.values():
        try:
            os.makedirs(folder, exist_ok=True)
            # Проверка прав на запись
            test_file = f"{folder}/test.txt"
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"Директория {folder} создана и доступна для записи")
        except Exception as e:
            print(f"ОШИБКА при создании директории {folder}: {e}")

    return folders


def plot_candlestick_with_patterns(df, pattern_data, candle_name, support_levels=None, resistance_levels=None,
                                   stats=None, save_path=None, figsize=(15, 12), indicators_config=None):
    """
    Создает график свечного паттерна с уровнями и индикаторами

    Args:
        df (DataFrame): DataFrame с ценовыми данными
        pattern_data (DataFrame): DataFrame с данными о паттерне
        candle_name (str): Название свечного паттерна
        support_levels (DataFrame, optional): DataFrame с уровнями поддержки
        resistance_levels (DataFrame, optional): DataFrame с уровнями сопротивления
        stats (dict, optional): Статистика эффективности паттерна
        save_path (str, optional): Путь для сохранения графика
        figsize (tuple): Размер графика
        indicators_config (dict): Конфигурация индикаторов для отображения

    Returns:
        Figure: Matplotlib Figure с построенным графиком
    """
    # Если конфигурация индикаторов не передана, используем пустой словарь
    if indicators_config is None:
        indicators_config = {
            'show_rsi': True,
            'show_macd': False,
            'show_bollinger': False,
            'show_sma': False,
            'show_atr': False
        }

    # Определяем количество подграфиков в зависимости от настроек
    num_indicator_plots = sum([
        indicators_config.get('show_rsi', False),
        indicators_config.get('show_macd', False),
        indicators_config.get('show_atr', False)
    ])
    
    # Всегда добавляем основной график и график статистики
    total_plots = 2 + num_indicator_plots
    
    # Создаем сетку подграфиков с помощью GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(total_plots, 1, height_ratios=[3] + [1] * (total_plots - 1))
    
    # Основной график свечей всегда первый
    ax1 = fig.add_subplot(gs[0])
    
    # Словарь для хранения осей индикаторов
    indicator_axes = {}
    current_subplot = 1
    
    # Добавляем оси для индикаторов
    if indicators_config.get('show_rsi', False):
        indicator_axes['rsi'] = fig.add_subplot(gs[current_subplot], sharex=ax1)
        current_subplot += 1
        
    if indicators_config.get('show_macd', False):
        indicator_axes['macd'] = fig.add_subplot(gs[current_subplot], sharex=ax1)
        current_subplot += 1
        
    if indicators_config.get('show_atr', False):
        indicator_axes['atr'] = fig.add_subplot(gs[current_subplot], sharex=ax1)
        current_subplot += 1
    
    # График статистики всегда последний
    ax_stats = fig.add_subplot(gs[-1])

    # Форматируем данные для candlestick_ohlc
    ohlc = df[['time_num', 'open', 'high', 'low', 'close']].values

    # Отображаем свечной график на основном графике
    candlestick_ohlc(ax1, ohlc, width=0.0008, colorup='g', colordown='r')

    # Добавляем Bollinger Bands на основной график, если включено
    if indicators_config.get('show_bollinger', False) and 'bb_upper' in df.columns:
        ax1.plot(df['time_num'], df['bb_upper'], 'b--', alpha=0.5, label='Upper BB')
        ax1.plot(df['time_num'], df['bb_middle'], 'b-', alpha=0.5, label='Middle BB')
        ax1.plot(df['time_num'], df['bb_lower'], 'b--', alpha=0.5, label='Lower BB')
        
    # Добавляем скользящие средние на основной график, если включено
    if indicators_config.get('show_sma', False):
        if 'sma_9' in df.columns:
            ax1.plot(df['time_num'], df['sma_9'], 'g-', alpha=0.7, label='SMA 9')
        if 'sma_20' in df.columns:
            ax1.plot(df['time_num'], df['sma_20'], 'r-', alpha=0.7, label='SMA 20')
        if 'sma_50' in df.columns:
            ax1.plot(df['time_num'], df['sma_50'], 'b-', alpha=0.7, label='SMA 50')
        ax1.legend(loc='upper left')

    # Добавляем вертикальные линии и текстовые метки для сигналов
    for index, row in pattern_data.iterrows():
        x = row['time_num']
        y_min, y_max = df['low'].min(), df['high'].max()  # Границы графика

        if row[candle_name] > 0:  # Bull (Зеленая линия)
            ax1.axvline(x, color='green', linestyle='--', alpha=0.7)
            ax1.text(x, y_max * 1.02, "Bull", color='green', fontsize=10, rotation=90, verticalalignment='bottom')
        else:  # Bear (Красная линия)
            ax1.axvline(x, color='red', linestyle='--', alpha=0.7)
            ax1.text(x, y_min * 0.98, "Bear", color='red', fontsize=10, rotation=90, verticalalignment='top')

    # Отображаем уровни поддержки и сопротивления, если предоставлены
    if support_levels is not None and not support_levels.empty:
        plot_support_levels(ax1, support_levels, df)

    if resistance_levels is not None and not resistance_levels.empty:
        plot_resistance_levels(ax1, resistance_levels, df)

    # Отображаем пробитие уровней поддержки и сопротивления, если предоставлены
    if config.SHOW_BROKEN_LEVELS and config.SHOW_BROKEN_SUPPORT and broken_support is not None and not broken_support.empty:
        plot_broken_support_levels(ax1, broken_support, df)

    if config.SHOW_BROKEN_LEVELS and config.SHOW_BROKEN_RESISTANCE and broken_resistance is not None and not broken_resistance.empty:
        plot_broken_resistance_levels(ax1, broken_resistance, df)
    
    # Настройки оси времени для основного графика
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=45)
    ax1.set_title(f"ETH/USDT - {candle_name}")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    
    
    # Отображаем RSI, если включен
    if indicators_config.get('show_rsi', False) and 'rsi' in indicator_axes:
        plot_rsi(indicator_axes['rsi'], df, pattern_data, candle_name)
        plt.setp(indicator_axes['rsi'].get_xticklabels(), visible=False)

    # Отображаем MACD, если включен
    if indicators_config.get('show_macd', False) and 'macd' in indicator_axes:
        plot_macd(indicator_axes['macd'], df, pattern_data, candle_name)
        plt.setp(indicator_axes['macd'].get_xticklabels(), visible=False)

    # Отображаем ATR, если включен
    if indicators_config.get('show_atr', False) and 'atr' in indicator_axes:
        plot_atr(indicator_axes['atr'], df, pattern_data, candle_name)
        plt.setp(indicator_axes['atr'].get_xticklabels(), visible=False)

    # Отображаем статистику на нижнем графике, если предоставлена
    if stats:
        plot_pattern_stats(ax_stats, stats)

    plt.tight_layout()

    # Сохраняем график, если указан путь
    if save_path:
        plt.savefig(save_path)

    return fig


def plot_support_levels(ax, support_levels, df):
    """
    Отображает уровни поддержки на графике

    Args:
        ax (Axes): Оси для отображения
        support_levels (DataFrame): DataFrame с уровнями поддержки
        df (DataFrame): DataFrame с ценовыми данными
    """
    date_range = [df['time_num'].min(), df['time_num'].max()]

    for _, level_data in support_levels.iterrows():
        level = level_data['level']
        formation_time_num = level_data['formation_time_num']
        formation_time_str = level_data['formation_time'].strftime('%Y-%m-%d') if hasattr(level_data['formation_time'],
                                                                                          'strftime') else level_data[
            'formation_time']

        # Рисуем горизонтальную линию уровня (зеленый цвет для поддержки)
        ax.axhline(y=level, color='green', linestyle='-', alpha=0.5)

        # Добавляем метку с временем формирования за пределами графика
        ax.annotate(f"S: {level:.2f} ({formation_time_str})",
                    xy=(date_range[1], level),
                    xytext=(date_range[1] + (date_range[1] - date_range[0]) * 0.02, level),
                    color='green', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.3'))

        # Отмечаем точку формирования уровня
        ax.plot(formation_time_num, level, 'go', markersize=4)


def plot_resistance_levels(ax, resistance_levels, df):
    """
    Отображает уровни сопротивления на графике

    Args:
        ax (Axes): Оси для отображения
        resistance_levels (DataFrame): DataFrame с уровнями сопротивления
        df (DataFrame): DataFrame с ценовыми данными
    """
    date_range = [df['time_num'].min(), df['time_num'].max()]

    for _, level_data in resistance_levels.iterrows():
        level = level_data['level']
        formation_time_num = level_data['formation_time_num']
        formation_time_str = level_data['formation_time'].strftime('%Y-%m-%d') if hasattr(level_data['formation_time'],
                                                                                          'strftime') else level_data[
            'formation_time']

        # Рисуем горизонтальную линию уровня
        ax.axhline(y=level, color='red', linestyle='-', alpha=0.5)

        # Добавляем метку с временем формирования за пределами графика
        ax.annotate(f"R: {level:.2f} ({formation_time_str})",
                    xy=(date_range[1], level),
                    xytext=(date_range[1] + (date_range[1] - date_range[0]) * 0.02, level),
                    color='red', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.3'))

        # Отмечаем точку формирования уровня
        ax.plot(formation_time_num, level, 'ro', markersize=4)


def plot_broken_resistance_levels(ax, broken_levels, df):
    """
    Отображает пробитые уровни сопротивления на графике

    Эта функция:

    1.Проходит по каждому пробитому уровню сопротивления
    2.Отображает сплошную горизонтальную линию от момента формирования уровня до момента его пробития
    3.Отображает пунктирную горизонтальную линию после пробития (что указывает, что уровень был пробит)
    4.Отмечает точку пробития зеленым крестиком (зеленый цвет используется для обозначения положительного движения цены вверх)
    5.Добавляет аннотацию с датой пробития

    Args:
        ax (Axes): Оси для отображения
        broken_levels (DataFrame): DataFrame с пробитыми уровнями сопротивления
        df (DataFrame): DataFrame с ценовыми данными
    """
    date_range = [df['time_num'].min(), df['time_num'].max()]

    for _, level_data in broken_levels.iterrows():
        level = level_data['level']
        formation_time_num = level_data['formation_time_num']
        break_time_num = mdates.date2num(level_data['break_time'])

        # Рисуем горизонтальную линию уровня до момента пробития (сплошная)
        ax.plot([formation_time_num, break_time_num], [level, level],
                color='red', linestyle='-', alpha=0.5)

        # Рисуем горизонтальную линию уровня после пробития (пунктирная)
        ax.plot([break_time_num, date_range[1]], [level, level],
                color='red', linestyle='--', alpha=0.5)

        # Отмечаем точку пробития
        ax.plot(break_time_num, level, 'gx', markersize=6)

        # Добавляем аннотацию с информацией о пробитии
        break_time_str = level_data['break_time'].strftime('%Y-%m-%d') if hasattr(level_data['break_time'],
                                                                                  'strftime') else level_data[
            'break_time']

        ax.annotate(f"Break: {break_time_str}",
                    xy=(break_time_num, level),
                    xytext=(break_time_num + 5, level + (df['high'].max() - df['low'].min()) * 0.02),
                    color='red', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.3'),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))


def plot_broken_support_levels(ax, broken_levels, df):
    """
    Отображает пробитые уровни поддержки на графике
    Эта функция:

    1.Проходит по каждому пробитому уровню поддержки
    2. Отображает сплошную горизонтальную линию от момента формирования уровня до момента его пробития
    3. Отображает пунктирную горизонтальную линию после пробития (что указывает, что уровень был пробит)
    4. Отмечает точку пробития красным крестиком (красный цвет используется для обозначения негативного движения цены вниз)
    5. Добавляет аннотацию с датой пробития

    Args:
        ax (Axes): Оси для отображения
        broken_levels (DataFrame): DataFrame с пробитыми уровнями поддержки
        df (DataFrame): DataFrame с ценовыми данными
    """
    date_range = [df['time_num'].min(), df['time_num'].max()]

    for _, level_data in broken_levels.iterrows():
        level = level_data['level']
        formation_time_num = level_data['formation_time_num']
        break_time_num = mdates.date2num(level_data['break_time'])

        # Рисуем горизонтальную линию уровня до момента пробития (сплошная)
        ax.plot([formation_time_num, break_time_num], [level, level],
                color='green', linestyle='-', alpha=0.5)

        # Рисуем горизонтальную линию уровня после пробития (пунктирная)
        ax.plot([break_time_num, date_range[1]], [level, level],
                color='green', linestyle='--', alpha=0.5)

        # Отмечаем точку пробития
        ax.plot(break_time_num, level, 'rx', markersize=6)

        # Добавляем аннотацию с информацией о пробитии
        break_time_str = level_data['break_time'].strftime('%Y-%m-%d') if hasattr(level_data['break_time'],
                                                                                  'strftime') else level_data[
            'break_time']

        ax.annotate(f"Break: {break_time_str}",
                    xy=(break_time_num, level),
                    xytext=(break_time_num + 5, level - (df['high'].max() - df['low'].min()) * 0.02),
                    color='green', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.3'),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))


def plot_rsi(ax, df, pattern_data, candle_name=None):
    """
    Отображает индикатор RSI на графике

    Args:
        ax (Axes): Оси для отображения
        df (DataFrame): DataFrame с ценовыми данными и RSI
        pattern_data (DataFrame): DataFrame с данными о паттерне
        candle_name (str, optional): Название свечного паттерна
    """
    rsi_data = df['rsi']
    ax.plot(df['time_num'], rsi_data, color='purple', linewidth=1)
    ax.axhline(y=30, color='green', linestyle='-', alpha=0.5)  # Уровень перепроданности (30%)
    ax.axhline(y=70, color='red', linestyle='-', alpha=0.5)  # Уровень перекупленности (70%)
    ax.fill_between(df['time_num'], rsi_data, 30, where=(rsi_data <= 30), color='green',
                    alpha=0.2)  # Подсветка перепроданной зоны
    ax.fill_between(df['time_num'], rsi_data, 70, where=(rsi_data >= 70), color='red',
                    alpha=0.2)  # Подсветка перекупленной зоны

    # Добавляем вертикальные линии для сигналов на график RSI
    for index, row in pattern_data.iterrows():
        x = row['time_num']
        if candle_name:
            signal_value = row[candle_name]
        else:
            signal_value = row['pattern_value'] if 'pattern_value' in row else None

        if signal_value is not None:
            if signal_value > 0:  # Bull сигнал
                ax.axvline(x, color='green', linestyle='--', alpha=0.7)
            else:  # Bear сигнал
                ax.axvline(x, color='red', linestyle='--', alpha=0.7)

    # Настройки графика RSI
    ax.set_ylabel("RSI (14)")
    ax.grid(True)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


def plot_macd(ax, df, pattern_data, candle_name=None):
    """
    Отображает индикатор MACD на графике

    Args:
        ax (Axes): Оси для отображения
        df (DataFrame): DataFrame с ценовыми данными и MACD
        pattern_data (DataFrame): DataFrame с данными о паттерне
        candle_name (str, optional): Название свечного паттерна
    """
    if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
        # Отображаем линии MACD и сигнальную линию
        ax.plot(df['time_num'], df['macd'], color='blue', linewidth=1, label='MACD')
        ax.plot(df['time_num'], df['macd_signal'], color='red', linewidth=1, label='Signal')
        
        # Отображаем гистограмму MACD
        for i in range(len(df) - 1):
            if df['macd_hist'].iloc[i] >= 0:
                ax.bar(df['time_num'].iloc[i], df['macd_hist'].iloc[i], color='green', width=0.0008, alpha=0.5)
            else:
                ax.bar(df['time_num'].iloc[i], df['macd_hist'].iloc[i], color='red', width=0.0008, alpha=0.5)
        
        # Добавляем уровень ноля
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Добавляем вертикальные линии для сигналов на график MACD
        for index, row in pattern_data.iterrows():
            x = row['time_num']
            if candle_name and candle_name in row:
                signal_value = row[candle_name]
                if signal_value > 0:  # Bull сигнал
                    ax.axvline(x, color='green', linestyle='--', alpha=0.7)
                else:  # Bear сигнал
                    ax.axvline(x, color='red', linestyle='--', alpha=0.7)
        
        # Настройки графика MACD
        ax.set_ylabel("MACD")
        ax.grid(True)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


def plot_atr(ax, df, pattern_data, candle_name=None):
    """
    Отображает индикатор ATR на графике

    Args:
        ax (Axes): Оси для отображения
        df (DataFrame): DataFrame с ценовыми данными и ATR
        pattern_data (DataFrame): DataFrame с данными о паттерне
        candle_name (str, optional): Название свечного паттерна
    """
    if 'atr' in df.columns:
        # Отображаем ATR
        ax.plot(df['time_num'], df['atr'], color='orange', linewidth=1, label='ATR (14)')
        
        # Добавляем вертикальные линии для сигналов на график ATR
        for index, row in pattern_data.iterrows():
            x = row['time_num']
            if candle_name and candle_name in row:
                signal_value = row[candle_name]
                if signal_value > 0:  # Bull сигнал
                    ax.axvline(x, color='green', linestyle='--', alpha=0.7)
                else:  # Bear сигнал
                    ax.axvline(x, color='red', linestyle='--', alpha=0.7)
        
        # Настройки графика ATR
        ax.set_ylabel("ATR (14)")
        ax.grid(True)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


def plot_pattern_stats(ax, stats):
    """
    Отображает статистику эффективности паттерна на графике

    Args:
        ax (Axes): Оси для отображения
        stats (dict): Статистика эффективности паттерна
    """
    x_positions = [0.25, 0.75]
    bar_width = 0.2

    # Бычья статистика
    if stats['bull_high_avg'] is not None:
        ax.bar(x_positions[0] - bar_width / 2, stats['bull_high_avg'], width=bar_width, color='green', alpha=0.7,
               label='Bull High')
        ax.bar(x_positions[0] + bar_width / 2, stats['bull_low_avg'], width=bar_width, color='lightgreen',
               alpha=0.7, label='Bull Low')

        # Добавляем текст со значениями
        ax.text(x_positions[0] - bar_width / 2,
                stats['bull_high_avg'] * (1.1 if stats['bull_high_avg'] > 0 else 0.9),
                f"{stats['bull_high_avg']:.2f}%", ha='center',
                va='bottom' if stats['bull_high_avg'] > 0 else 'top')
        ax.text(x_positions[0] + bar_width / 2,
                stats['bull_low_avg'] * (1.1 if stats['bull_low_avg'] > 0 else 0.9),
                f"{stats['bull_low_avg']:.2f}%", ha='center', va='bottom' if stats['bull_low_avg'] > 0 else 'top')

        # Добавляем количество сигналов
        ax.text(x_positions[0], 0, f"n={stats['bull_count']}", ha='center', va='bottom', fontsize=9)

    # Медвежья статистика
    if stats['bear_high_avg'] is not None:
        ax.bar(x_positions[1] - bar_width / 2, stats['bear_high_avg'], width=bar_width, color='darkred', alpha=0.7,
               label='Bear High')
        ax.bar(x_positions[1] + bar_width / 2, stats['bear_low_avg'], width=bar_width, color='red', alpha=0.7,
               label='Bear Low')

        # Добавляем текст со значениями
        ax.text(x_positions[1] - bar_width / 2,
                stats['bear_high_avg'] * (1.1 if stats['bear_high_avg'] > 0 else 0.9),
                f"{stats['bear_high_avg']:.2f}%", ha='center',
                va='bottom' if stats['bear_high_avg'] > 0 else 'top')
        ax.text(x_positions[1] + bar_width / 2,
                stats['bear_low_avg'] * (1.1 if stats['bear_low_avg'] > 0 else 0.9),
                f"{stats['bear_low_avg']:.2f}%", ha='center', va='bottom' if stats['bear_low_avg'] > 0 else 'top')

        # Добавляем количество сигналов
        ax.text(x_positions[1], 0, f"n={stats['bear_count']}", ha='center', va='bottom', fontsize=9)

    # Настройки графика статистики
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylabel("Изменение цены (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["BULL сигналы", "BEAR сигналы"])
    ax.set_title("Средний % изменения цены в серии свечей до разворота")
    ax.grid(True, axis='y')
    ax.legend()

    # Добавляем подписи для столбцов
    if stats['bull_high_avg'] is not None:
        ax.annotate("MAX", (x_positions[0] - bar_width / 2, 0), xytext=(0, -20),
                    textcoords='offset points', ha='center', va='top')
        ax.annotate("MIN", (x_positions[0] + bar_width / 2, 0), xytext=(0, -20),
                    textcoords='offset points', ha='center', va='top')

    if stats['bear_high_avg'] is not None:
        ax.annotate("MAX", (x_positions[1] - bar_width / 2, 0), xytext=(0, -20),
                    textcoords='offset points', ha='center', va='top')
        ax.annotate("MIN", (x_positions[1] + bar_width / 2, 0), xytext=(0, -20),
                    textcoords='offset points', ha='center', va='top')


def plot_equity_curve(trades_df, strategy_name=None, figsize=(10, 6), save_path=None):
    """
    Отображает кривую доходности на основе истории сделок

    Args:
        trades_df (DataFrame): DataFrame с историей сделок
        strategy_name (str, optional): Название стратегии
        figsize (tuple): Размер графика
        save_path (str, optional): Путь для сохранения графика

    Returns:
        Figure: Matplotlib Figure с построенным графиком
    """
    if trades_df.empty:
        return None

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=figsize)

    # Рассчитываем кумулятивную доходность
    trades_df = trades_df.sort_values('entry_time')

    # Преобразуем проценты в коэффициенты
    returns = trades_df['profit_pct'] / 100
    cumulative_returns = (1 + returns).cumprod() - 1

    # Строим график
    ax.plot(range(len(cumulative_returns)), cumulative_returns * 100, label=strategy_name or 'Strategy')

    # Добавляем горизонтальную линию на уровне 0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Рассчитываем подсветку для просадок
    dd = cumulative_returns.divide(cumulative_returns.cummax()) - 1
    ax.fill_between(range(len(dd)), 0, dd * 100, color='red', alpha=0.3, label='Drawdown')

    # Настройки графика
    ax.set_xlabel('Trades')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Equity Curve' + (f' - {strategy_name}' if strategy_name else ''))
    ax.grid(True)
    ax.legend()

    # Добавляем информацию о доходности
    final_return = cumulative_returns.iloc[-1] * 100
    max_drawdown = dd.min() * 100
    win_rate = len(trades_df[trades_df['profit_pct'] > 0]) / len(trades_df) * 100

    info_text = (f'Total Return: {final_return:.2f}%\n'
                 f'Max Drawdown: {max_drawdown:.2f}%\n'
                 f'Win Rate: {win_rate:.2f}%\n'
                 f'Total Trades: {len(trades_df)}')

    ax.text(0.02, 0.97, info_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Сохраняем график, если указан путь
    if save_path:
        plt.savefig(save_path)

    return fig


def compare_strategies_plot(comparison_df, metric='total_profit', figsize=(12, 8), save_path=None):
    """
    Строит сравнительный график эффективности стратегий

    Args:
        comparison_df (DataFrame): DataFrame со сравнительной статистикой стратегий
        metric (str): Метрика для сравнения
        figsize (tuple): Размер графика
        save_path (str, optional): Путь для сохранения графика

    Returns:
        Figure: Matplotlib Figure с построенным графиком
    """
    if comparison_df.empty:
        return None

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=figsize)

    # Сортируем DataFrame по выбранной метрике
    sorted_df = comparison_df.sort_values(metric, ascending=False)

    # Создаем бар-график
    bars = ax.bar(sorted_df['strategy_name'], sorted_df[metric], color='skyblue')

    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02 * max(sorted_df[metric]),
                f'{height:.2f}', ha='center', va='bottom')

    # Настройки графика
    ax.set_xlabel('Strategy')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Strategy Comparison by {metric.replace("_", " ").title()}')
    ax.grid(True, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Сохраняем график, если указан путь
    if save_path:
        plt.savefig(save_path)

    return fig


def plot_trends_on_chart(ax, trend_data, df):
    """
    Отображает тренды на графике

    Args:
        ax (Axes): Объект оси для рисования
        trend_data (list): Список данных о трендах
        df (DataFrame): DataFrame с данными свечей
    """
    for trend in trend_data:
        index = trend['index']
        direction = trend['direction']
        strength = trend['strength']

        if 'data' in trend and trend['data']:
            slope, intercept, x0, x1, price0, price1 = trend['data']

            # Преобразуем индексы в значения для оси X
            time0 = df.iloc[index - (x1 - x0)]['time_num']
            time1 = df.iloc[index]['time_num']

            # Цвет линии в зависимости от направления тренда
            if direction == 'up':
                color = 'green'
                label = f"Восходящий ({strength:.2f})"
            elif direction == 'down':
                color = 'red'
                label = f"Нисходящий ({strength:.2f})"
            else:
                color = 'gray'
                label = f"Боковой ({strength:.2f})"

            # Рисуем линию тренда
            ax.plot([time0, time1], [price0, price1],
                    color=color, linestyle='-', linewidth=2, alpha=0.7)

            # Добавляем текстовую метку с информацией о тренде
            if index % 10 == 0:  # Рисуем не все метки, чтобы избежать перегрузки
                ax.text(time1, price1, label, color=color,
                        fontsize=8, verticalalignment='bottom')