import pandas as pd
import numpy as np
from datetime import datetime
import config
import utils

class ATRBacktester:
    """
    Класс для бэктестинга стратегий, основанных на ATR (Average True Range)
    с использованием трейлинг-стопа и подтверждения сигнала
    """

    def __init__(self, *args, trend_lookback=20, trend_r_squared=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.trend_lookback = trend_lookback
        self.trend_r_squared = trend_r_squared
        self.trend_data = []  # для хранения истории трендов
        # Новая структура для хранения всех сигналов паттерна
        self.all_pattern_signals = []

    def __init__(self,
                 atr_period=config.ATR_PERIOD,
                 atr_confirmation_multiple=config.ATR_CONFIRMATION_MULTIPLE,
                 atr_stop_multiple=config.ATR_STOP_MULTIPLE,
                 confirmation_period=config.CONFIRMATION_PERIOD,
                 position_size=1.0,  # Этот параметр оставляем без изменений, если нет в config
                 trend_lookback=config.TREND_LOOKBACK,
                 trend_r_squared=config.TREND_R_SQUARED,
                 trend_filter_mode=config.TREND_FILTER_MODE):
        """
        Инициализирует бэктестер на основе ATR

        Args:
            atr_period (int): Период для расчета ATR
            atr_confirmation_multiple (float): Множитель ATR для подтверждения сигнала
            atr_stop_multiple (float): Множитель ATR для стоп-лосса и трейлинг-стопа
            confirmation_period (int): Количество свечей для ожидания подтверждения сигнала
            position_size (float): Размер позиции (доля от капитала)
            trend_lookback (int): Период для расчета тренда с помощью линейной регрессии
            trend_r_squared (float): Минимальный R² для определения значимого тренда
        """
        self.atr_period = atr_period
        self.atr_confirmation_multiple = atr_confirmation_multiple
        self.atr_stop_multiple = atr_stop_multiple
        self.confirmation_period = confirmation_period
        self.position_size = position_size
        self.trend_lookback = trend_lookback
        self.trend_r_squared = trend_r_squared
        self.trend_filter_mode = trend_filter_mode

        self.trades = []
        self.open_positions = []
        self.pending_signals = []
        self.trend_data = []  # для хранения истории трендов
        # Новая структура для хранения всех сигналов паттерна
        self.all_pattern_signals = []
        
    def test_trend_calculation(self, df):
        """
        Тестовый метод для проверки расчета трендов
        """
        print("====== ТЕСТИРОВАНИЕ РАСЧЕТА ТРЕНДОВ ======")
        for i in range(0, len(df), max(1, len(df) // 10)):  # Проверяем ~10 точек
            trend_direction, trend_strength, trend_info = self._calculate_trend(df, i)
            print(f"Индекс {i}: тренд={trend_direction}, сила={trend_strength:.4f}, данные={trend_info is not None}")
        print("==========================================")
        return True


    def test_pattern_ver4(self, df, pattern_name):
        """
        Проводит бэктестинг конкретного свечного паттерна с учетом тренда

        Args:
            df (DataFrame): DataFrame с данными свечей и ATR
            pattern_name (str): Название паттерна для тестирования

        Returns:
            dict: Статистика по результатам бэктестинга, включая данные о трендах
        """
        # Убедимся, что у нас есть необходимые данные
        required_columns = ['open', 'high', 'low', 'close', 'time', 'atr', pattern_name]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"В DataFrame отсутствует столбец: {col}")

        # Очищаем предыдущие результаты
        self.trades = []
        self.open_positions = []
        self.pending_signals = []
        self.trend_data = []  # Очищаем данные о трендах
        self.all_pattern_signals = [] # Очищаем данные о сигналах от паттернов

        trend_count = 0  # Счетчик для отслеживания расчета трендов

        print("=========== НАЧАЛО БЭКТЕСТИНГА ===========")
        print(f"Паттерн: {pattern_name}")
        print(f"Размер данных: {len(df)} свечей")
        print(f"Запуск бэктестинга с режимом тренда: {self.trend_filter_mode}")

        # Проходим по каждой свече
        for i in range(len(df) - 1):
            current_candle = df.iloc[i]
            next_candle = df.iloc[i + 1]

            # Определяем тренд для текущей свечи
            try:
                trend_direction, trend_strength, trend_info = self._calculate_trend(df, i)
                trend_count += 1

                # Логирование каждые 1000 свечей
                if i % 1000 == 0:
                    print(f"Обработка свечи {i}: тренд={trend_direction}, сила={trend_strength:.4f}")

                # Сохраняем информацию о тренде для визуализации
                trend_entry = {
                    'index': i,
                    'time': current_candle['time'],
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'data': trend_info
                }
                self.trend_data.append(trend_entry)
            except Exception as e:
                print(f"Ошибка при расчете тренда для свечи {i}: {e}")
                # Добавляем пустые данные, чтобы не сбить индексацию
                self.trend_data.append({
                    'index': i,
                    'time': current_candle['time'],
                    'direction': 'flat',
                    'strength': 0,
                    'data': None
                })

            # Обрабатываем ожидающие подтверждения сигналы с учетом тренда
            try:
                self._process_pending_signals(df, i, trend_direction)
            except Exception as e:
                print(f"Ошибка при обработке сигналов для свечи {i}: {e}")

            # Обрабатываем открытые позиции
            try:
                self._update_open_positions(df, i)
            except Exception as e:
                print(f"Ошибка при обновлении позиций для свечи {i}: {e}")

            # Ищем новые сигналы
            try:
                self._look_for_new_signals(df, i, pattern_name)
            except Exception as e:
                print(f"Ошибка при поиске новых сигналов для свечи {i}: {e}")

        # Закрываем все оставшиеся позиции по последней цене
        try:
            self._close_remaining_positions(df)
        except Exception as e:
            print(f"Ошибка при закрытии оставшихся позиций: {e}")

        # Логирование перед возвратом
        print(f"Всего свечей обработано: {len(df)}")
        print(f"Всего трендов рассчитано: {trend_count}")
        print(f"Размер списка trend_data: {len(self.trend_data)}")
        print(f"Открытых сделок: {len(self.trades)}")

        # Рассчитываем и возвращаем статистику
        statistics = self._calculate_statistics()

        # Явно добавляем данные о трендах и все сигналы в результат
        statistics['trend_data'] = self.trend_data
        statistics['all_signals'] = self.all_pattern_signals

        print(f"Тренды добавлены в статистику: {'trend_data' in statistics}")
        print(f"Размер trend_data в статистике: {len(statistics['trend_data'])}")
        print("=========== КОНЕЦ БЭКТЕСТИНГА ===========")

        return statistics

    def test_pattern_ver41(self, df, pattern_name):
        """
        Проводит бэктестинг конкретного свечного паттерна с учетом тренда

        Args:
            df (DataFrame): DataFrame с данными свечей и ATR
            pattern_name (str): Название паттерна для тестирования

        Returns:
            dict: Статистика по результатам бэктестинга, включая данные о трендах
        """
        # Убедимся, что у нас есть необходимые данные
        required_columns = ['open', 'high', 'low', 'close', 'time', 'atr', pattern_name]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"В DataFrame отсутствует столбец: {col}")

        # Сохраняем main_df, если он еще не установлен
        if not hasattr(self, 'main_df') or self.main_df is None:
            self.main_df = df

        # Очищаем предыдущие результаты
        self.trades = []
        self.open_positions = []
        self.pending_signals = []
        self.trend_data = []  # Очищаем данные о трендах
        self.all_pattern_signals = [] # Очищаем данные о сигналах от паттернов

        # Проверяем, есть ли 15m данные
        has_m15_data = hasattr(self, 'm15_df') and self.m15_df is not None
        if has_m15_data:
            utils.log_progress(f"Бэктестинг будет использовать 15m данные для исполнения сделок", True)

        trend_count = 0  # Счетчик для отслеживания расчета трендов

        print("=========== НАЧАЛО БЭКТЕСТИНГА ===========")
        print(f"Паттерн: {pattern_name}")
        print(f"Размер данных: {len(df)} свечей")
        print(f"Запуск бэктестинга с режимом тренда: {self.trend_filter_mode}")

        # Проходим по каждой свече
        for i in range(len(df) - 1):
            current_candle = df.iloc[i]
            next_candle = df.iloc[i + 1]

            # Определяем тренд для текущей свечи
            try:
                trend_direction, trend_strength, trend_info = self._calculate_trend(df, i)
                trend_count += 1

                # Логирование каждые 1000 свечей
                if i % 1000 == 0:
                    print(f"Обработка свечи {i}: тренд={trend_direction}, сила={trend_strength:.4f}")

                # Сохраняем информацию о тренде для визуализации
                trend_entry = {
                    'index': i,
                    'time': current_candle['time'],
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'data': trend_info
                }
                self.trend_data.append(trend_entry)
            except Exception as e:
                print(f"Ошибка при расчете тренда для свечи {i}: {e}")
                # Добавляем пустые данные, чтобы не сбить индексацию
                self.trend_data.append({
                    'index': i,
                    'time': current_candle['time'],
                    'direction': 'flat',
                    'strength': 0,
                    'data': None
                })

            # Обрабатываем ожидающие подтверждения сигналы с учетом тренда
            try:
                self._process_pending_signals(df, i, trend_direction)
            except Exception as e:
                print(f"Ошибка при обработке сигналов для свечи {i}: {e}")

            # Обрабатываем открытые позиции
            try:
                self._update_open_positions(df, i)
            except Exception as e:
                print(f"Ошибка при обновлении позиций для свечи {i}: {e}")

            # Ищем новые сигналы
            try:
                self._look_for_new_signals(df, i, pattern_name)
            except Exception as e:
                print(f"Ошибка при поиске новых сигналов для свечи {i}: {e}")

        # Закрываем все оставшиеся позиции по последней цене
        try:
            self._close_remaining_positions(df)
        except Exception as e:
            print(f"Ошибка при закрытии оставшихся позиций: {e}")

        # Логирование перед возвратом
        print(f"Всего свечей обработано: {len(df)}")
        print(f"Всего трендов рассчитано: {trend_count}")
        print(f"Размер списка trend_data: {len(self.trend_data)}")
        print(f"Открытых сделок: {len(self.trades)}")

        # Рассчитываем и возвращаем статистику
        statistics = self._calculate_statistics()

        # Явно добавляем данные о трендах и все сигналы в результат
        statistics['trend_data'] = self.trend_data
        statistics['all_signals'] = self.all_pattern_signals

        print(f"Тренды добавлены в статистику: {'trend_data' in statistics}")
        print(f"Размер trend_data в статистике: {len(statistics['trend_data'])}")
        print("=========== КОНЕЦ БЭКТЕСТИНГА ===========")

        return statistics

    def test_pattern(self, df, pattern_name):
        """
        Проводит бэктестинг конкретного свечного паттерна с учетом тренда

        Args:
            df (DataFrame): DataFrame с данными свечей и ATR
            pattern_name (str): Название паттерна для тестирования

        Returns:
            dict: Статистика по результатам бэктестинга, включая данные о трендах
        """
        # Убедимся, что у нас есть необходимые данные
        required_columns = ['open', 'high', 'low', 'close', 'time', 'atr', pattern_name]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"В DataFrame отсутствует столбец: {col}")

        # Сохраняем main_df, если он еще не установлен
        if not hasattr(self, 'main_df') or self.main_df is None:
            self.main_df = df

        # Очищаем предыдущие результаты
        self.trades = []
        self.open_positions = []
        self.pending_signals = []
        self.trend_data = []  # Очищаем данные о трендах
        self.all_pattern_signals = []  # Очищаем данные о сигналах от паттернов

        # Проверяем, есть ли 15m данные
        has_m15_data = hasattr(self, 'm15_df') and self.m15_df is not None
        if has_m15_data:
            print(f"Бэктестинг будет использовать 15m данные для исполнения сделок")
            print(f"Размер m15_df: {len(self.m15_df)}")
            print(f"Период m15_df: {self.m15_df['time'].min()} - {self.m15_df['time'].max()}")

        trend_count = 0  # Счетчик для отслеживания расчета трендов

        print("=========== НАЧАЛО БЭКТЕСТИНГА ===========")
        print(f"Паттерн: {pattern_name}")
        print(f"Размер данных: {len(df)} свечей")
        print(f"Запуск бэктестинга с режимом тренда: {self.trend_filter_mode}")

        # Проходим по каждой свече
        for i in range(len(df) - 1):
            current_candle = df.iloc[i]
            next_candle = df.iloc[i + 1]

            # Детальное логирование для первых нескольких свечей
            if i < 5:  # Логируем детально только первые несколько свечей
                print(f"\n==== Детальный лог для свечи {i} ====")
                print(f"Время свечи: {current_candle['time']}")

                # Размер m15_df
                if has_m15_data:
                    print(f"Размер m15_df: {len(self.m15_df)}")

                # Логируем перед каждым важным шагом
                print("Начинаем расчет тренда...")

            # Логирование каждые 100 свечей
            if i % 100 == 0:
                print(f"Обработка свечи {i} из {len(df)} ({i / len(df) * 100:.1f}%)")

            # Определяем тренд для текущей свечи
            try:
                trend_direction, trend_strength, trend_info = self._calculate_trend(df, i)
                trend_count += 1

                if i < 5:
                    print(f"Расчет тренда завершен: {trend_direction}, сила={trend_strength:.4f}")
                    print("Начинаем обработку сигналов...")

                # Сохраняем информацию о тренде для визуализации
                trend_entry = {
                    'index': i,
                    'time': current_candle['time'],
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'data': trend_info
                }
                self.trend_data.append(trend_entry)
            except Exception as e:
                print(f"Ошибка при расчете тренда для свечи {i}: {e}")
                # Добавляем пустые данные, чтобы не сбить индексацию
                self.trend_data.append({
                    'index': i,
                    'time': current_candle['time'],
                    'direction': 'flat',
                    'strength': 0,
                    'data': None
                })

            # Обрабатываем ожидающие подтверждения сигналы с учетом тренда
            try:
                if i < 5:
                    print(f"Количество ожидающих сигналов перед обработкой: {len(self.pending_signals)}")

                self._process_pending_signals(df, i, trend_direction)

                if i < 5:
                    print("Обработка сигналов завершена")
                    print(f"Количество ожидающих сигналов после обработки: {len(self.pending_signals)}")
                    print("Начинаем обновление позиций...")
            except Exception as e:
                print(f"Ошибка при обработке сигналов для свечи {i}: {e}")
                import traceback
                traceback.print_exc()

            # Обрабатываем открытые позиции
            try:
                if i < 5:
                    print(f"Количество открытых позиций перед обновлением: {len(self.open_positions)}")

                self._update_open_positions(df, i)

                if i < 5:
                    print("Обновление позиций завершено")
                    print(f"Количество открытых позиций после обновления: {len(self.open_positions)}")
                    print("Начинаем поиск новых сигналов...")
            except Exception as e:
                print(f"Ошибка при обновлении позиций для свечи {i}: {e}")
                import traceback
                traceback.print_exc()

            # Ищем новые сигналы
            try:
                self._look_for_new_signals(df, i, pattern_name)

                if i < 5:
                    print("Поиск новых сигналов завершен")
                    print(f"Количество всех сигналов: {len(self.all_pattern_signals)}")
                    print("==== Завершена обработка свечи ====\n")
            except Exception as e:
                print(f"Ошибка при поиске новых сигналов для свечи {i}: {e}")
                import traceback
                traceback.print_exc()

        # Закрываем все оставшиеся позиции по последней цене
        try:
            print("Закрытие оставшихся позиций...")
            self._close_remaining_positions(df)
            print(f"Закрыто позиций: {len(self.open_positions)}")
        except Exception as e:
            print(f"Ошибка при закрытии оставшихся позиций: {e}")
            import traceback
            traceback.print_exc()

        # Логирование перед возвратом
        print(f"Всего свечей обработано: {len(df)}")
        print(f"Всего трендов рассчитано: {trend_count}")
        print(f"Размер списка trend_data: {len(self.trend_data)}")
        print(f"Всего сделок: {len(self.trades)}")

        # Рассчитываем и возвращаем статистику
        try:
            print("Расчет статистики...")
            statistics = self._calculate_statistics()
            print("Расчет статистики завершен")
        except Exception as e:
            print(f"Ошибка при расчете статистики: {e}")
            import traceback
            traceback.print_exc()
            # Возвращаем пустую статистику в случае ошибки
            statistics = {
                'total_trades': len(self.trades),
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0
            }

        # Явно добавляем данные о трендах и все сигналы в результат
        statistics['trend_data'] = self.trend_data
        statistics['all_signals'] = self.all_pattern_signals

        print(f"Тренды добавлены в статистику: {'trend_data' in statistics}")
        print(f"Размер trend_data в статистике: {len(statistics['trend_data'])}")
        print(f"Размер all_signals в статистике: {len(statistics['all_signals'])}")
        print("=========== КОНЕЦ БЭКТЕСТИНГА ===========")

        return statistics


    def _process_pending_signals_ver4(self, df, current_index, trend_direction):
        """
        Обрабатывает сигналы, ожидающие подтверждения, с учетом тренда

        Args:
            df (DataFrame): DataFrame с данными
            current_index (int): Индекс текущей свечи
            trend_direction (str): Направление тренда ('up', 'down', 'flat')

            В зависимости от выбранного режима:

            1. with_trend - будут обрабатываться только сигналы, совпадающие с текущим трендом
            2. against_trend - будут обрабатываться только сигналы, противоположные текущему тренду
            3. ignore_trend - все сигналы будут обрабатываться независимо от тренда

            Также в функцию добавлено сохранение информации о тренде на момент входа в позицию,
            что может быть полезно для последующего анализа результатов.

        """
        # Создаем новый список для обновления pending_signals
        updated_pending_signals = []

        # Создаем множество для отслеживания уже обработанных индексов сигналов
        processed_signal_indices = set()

        # Создаем множество для отслеживания типов открытых позиций
        # Это предотвратит открытие многих однотипных позиций
        open_position_types = set()
        for pos in self.open_positions:
            open_position_types.add(pos['type'])

        current_candle = df.iloc[current_index]

        # Сортируем сигналы по времени (индексу) и приоритету
        # Для бычьего тренда приоритет у бычьих сигналов, для медвежьего - у медвежьих
        sorted_signals = sorted(self.pending_signals,
                                key=lambda x: (x['index'],
                                               0 if (x['type'] == 'bull' and trend_direction == 'up') or
                                                    (x['type'] == 'bear' and trend_direction == 'down') else 1))

        for signal in sorted_signals:
            signal_index = signal['index']
            signal_type = signal['type']
            confirmation_price = signal['confirmation_price']
            atr_value = signal['atr_value']
            elapsed_bars = current_index - signal_index
            pattern_name = signal['pattern']

            # Пропускаем, если индекс уже обработан
            if signal_index in processed_signal_indices:
                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = False
                        s['ignore_reason'] = 'duplicate_signal_index'
                        break
                continue

            # Проверяем период ожидания подтверждения
            if elapsed_bars > self.confirmation_period:
                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = False
                        s['ignore_reason'] = 'confirmation_period_expired'
                        break
                continue

            # Проверяем соответствие сигнала тренду в зависимости от режима фильтрации
            should_skip_signal = False
            trend_reason = None

            if self.trend_filter_mode == 'with_trend':
                # Оставляем только сигналы, совпадающие с трендом
                if (signal_type == 'bull' and trend_direction != 'up') or \
                        (signal_type == 'bear' and trend_direction != 'down'):
                    should_skip_signal = True
                    trend_reason = 'signal_against_trend'

            elif self.trend_filter_mode == 'against_trend':
                # Оставляем только сигналы, противоположные тренду
                if (signal_type == 'bull' and trend_direction != 'down') or \
                        (signal_type == 'bear' and trend_direction != 'up'):
                    should_skip_signal = True
                    trend_reason = 'signal_with_trend'

            # В режиме 'ignore_trend' мы пропускаем эту проверку полностью

            # Если нужно пропустить сигнал по причине тренда
            if should_skip_signal:
                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = False
                        s['ignore_reason'] = trend_reason
                        break

                updated_pending_signals.append(signal)  # Возвращаем необработанный сигнал в очередь
                continue

            # Избегаем открытия противоположных позиций
            if (signal_type == 'bull' and 'short' in open_position_types) or \
                    (signal_type == 'bear' and 'long' in open_position_types):
                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = False
                        s['ignore_reason'] = 'opposite_position_exists'
                        break
                continue

            # Помечаем индекс как обработанный
            processed_signal_indices.add(signal_index)

            # Проверяем, подтвердился ли сигнал
            if signal_type == 'bull' and current_candle['high'] >= confirmation_price:
                # Бычий сигнал подтвержден - открываем длинную позицию
                entry_price = confirmation_price
                stop_loss = entry_price - (self.atr_stop_multiple * atr_value)

                self.open_positions.append({
                    'type': 'long',
                    'entry_index': current_index,
                    'entry_time': current_candle['time'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'trailing_stop': stop_loss,
                    'atr_value': atr_value,
                    'max_price': entry_price,
                    'pattern': pattern_name,
                    'trend_at_entry': trend_direction
                })

                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = True
                        s['confirmation_time'] = current_candle['time']
                        s['confirmation_index'] = current_index
                        s['entry_price'] = entry_price
                        s['stop_loss'] = stop_loss
                        s['trend_at_entry'] = trend_direction
                        break

                # После открытия позиции удаляем все ожидающие бычьи сигналы
                # с тем же паттерном, чтобы избежать дублирования
                updated_pending_signals = [s for s in updated_pending_signals
                                           if not (s['type'] == 'bull' and s['pattern'] == pattern_name)]

            elif signal_type == 'bear' and current_candle['low'] <= confirmation_price:
                # Медвежий сигнал подтвержден - открываем короткую позицию
                entry_price = confirmation_price
                stop_loss = entry_price + (self.atr_stop_multiple * atr_value)

                self.open_positions.append({
                    'type': 'short',
                    'entry_index': current_index,
                    'entry_time': current_candle['time'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'trailing_stop': stop_loss,
                    'atr_value': atr_value,
                    'min_price': entry_price,
                    'pattern': pattern_name,
                    'trend_at_entry': trend_direction
                })

                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = True
                        s['confirmation_time'] = current_candle['time']
                        s['confirmation_index'] = current_index
                        s['entry_price'] = entry_price
                        s['stop_loss'] = stop_loss
                        s['trend_at_entry'] = trend_direction
                        break

                # После открытия позиции удаляем все ожидающие медвежьи сигналы
                # с тем же паттерном, чтобы избежать дублирования
                updated_pending_signals = [s for s in updated_pending_signals
                                           if not (s['type'] == 'bear' and s['pattern'] == pattern_name)]

            else:
                # Сигнал еще не подтвержден, добавляем его обратно в список ожидающих
                updated_pending_signals.append(signal)

                # Обновляем информацию в all_pattern_signals о том, что сигнал все еще в ожидании
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['still_pending'] = True
                        s['elapsed_bars'] = elapsed_bars
                        break

        # Обновляем список ожидающих сигналов
        self.pending_signals = updated_pending_signals

    def _process_pending_signals(self, df, current_index, trend_direction):
        """
        Обрабатывает сигналы, ожидающие подтверждения, с учетом тренда
        При наличии 15m данных использует их для поиска точек входа

        Args:
            df (DataFrame): DataFrame с данными 1h
            current_index (int): Индекс текущей 1h свечи
            trend_direction (str): Направление тренда ('up', 'down', 'flat')
        """
        # Создаем новый список для обновления pending_signals
        updated_pending_signals = []

        # Создаем множество для отслеживания уже обработанных индексов сигналов
        processed_signal_indices = set()

        # Создаем множество для отслеживания типов открытых позиций
        open_position_types = set()
        for pos in self.open_positions:
            open_position_types.add(pos['type'])

        current_candle = df.iloc[current_index]

        # Получаем 15m свечи для текущей 1h свечи, если они доступны
        m15_indices = []
        if hasattr(self, 'm15_df') and self.m15_df is not None:
            h1_time = df.iloc[current_index]['time']
            next_h1_time = df.iloc[current_index + 1]['time'] if current_index < len(df) - 1 else None

            # Выведем диагностическую информацию
            print(f"Поиск 15m свечей для h1 свечи {current_index}, h1_time={h1_time}, next_h1_time={next_h1_time}")

            # Вместо перебора всех строк используем прямую фильтрацию
            # Это может быть на порядки быстрее
            filtered_df = self.m15_df[
                (self.m15_df['time'] >= h1_time) &
                ((next_h1_time is None) | (self.m15_df['time'] < next_h1_time))
                ]
            m15_indices = filtered_df.index.tolist()

            # Ограничим количество для производительности
            if len(m15_indices) > 10:
                print(f"Ограничение 15m свечей с {len(m15_indices)} до 10 для оптимизации")
                m15_indices = m15_indices[:10]

        # Сортируем сигналы по времени (индексу) и приоритету
        # Для бычьего тренда приоритет у бычьих сигналов, для медвежьего - у медвежьих
        sorted_signals = sorted(self.pending_signals,
                                key=lambda x: (x['index'],
                                               0 if (x['type'] == 'bull' and trend_direction == 'up') or
                                                    (x['type'] == 'bear' and trend_direction == 'down') else 1))

        for signal in sorted_signals:
            signal_index = signal['index']
            signal_type = signal['type']
            confirmation_price = signal['confirmation_price']
            atr_value = signal['atr_value']
            elapsed_bars = current_index - signal_index
            pattern_name = signal['pattern']

            # Пропускаем, если индекс уже обработан
            if signal_index in processed_signal_indices:
                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = False
                        s['ignore_reason'] = 'duplicate_signal_index'
                        break
                continue

            # Проверяем период ожидания подтверждения
            if elapsed_bars > self.confirmation_period:
                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = False
                        s['ignore_reason'] = 'confirmation_period_expired'
                        break
                continue

            # Проверяем соответствие сигнала тренду в зависимости от режима фильтрации
            should_skip_signal = False
            trend_reason = None

            if self.trend_filter_mode == 'with_trend':
                # Оставляем только сигналы, совпадающие с трендом
                if (signal_type == 'bull' and trend_direction != 'up') or \
                        (signal_type == 'bear' and trend_direction != 'down'):
                    should_skip_signal = True
                    trend_reason = 'signal_against_trend'

            elif self.trend_filter_mode == 'against_trend':
                # Оставляем только сигналы, противоположные тренду
                if (signal_type == 'bull' and trend_direction != 'down') or \
                        (signal_type == 'bear' and trend_direction != 'up'):
                    should_skip_signal = True
                    trend_reason = 'signal_with_trend'

            # Если нужно пропустить сигнал по причине тренда
            if should_skip_signal:
                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = False
                        s['ignore_reason'] = trend_reason
                        break

                updated_pending_signals.append(signal)  # Возвращаем необработанный сигнал в очередь
                continue

            # Избегаем открытия противоположных позиций
            if (signal_type == 'bull' and 'short' in open_position_types) or \
                    (signal_type == 'bear' and 'long' in open_position_types):
                # Обновляем информацию в all_pattern_signals
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['processed'] = True
                        s['trade_opened'] = False
                        s['ignore_reason'] = 'opposite_position_exists'
                        break
                continue

            # Помечаем индекс как обработанный
            processed_signal_indices.add(signal_index)

            # Проверяем, подтвердился ли сигнал на 15m свечах
            is_confirmed = False
            confirmation_candle = None
            m15_confirmation_index = None

            # Если у нас есть 15m данные, проверяем их сначала
            if m15_indices:
                for m15_idx in m15_indices:
                    m15_candle = self.m15_df.iloc[m15_idx]

                    if signal_type == 'bull' and m15_candle['high'] >= confirmation_price:
                        is_confirmed = True
                        confirmation_candle = m15_candle
                        m15_confirmation_index = m15_idx
                        break

                    elif signal_type == 'bear' and m15_candle['low'] <= confirmation_price:
                        is_confirmed = True
                        confirmation_candle = m15_candle
                        m15_confirmation_index = m15_idx
                        break

            # Если не подтвердилось на 15m, проверяем на 1h
            if not is_confirmed:
                if signal_type == 'bull' and current_candle['high'] >= confirmation_price:
                    is_confirmed = True
                    confirmation_candle = current_candle

                elif signal_type == 'bear' and current_candle['low'] <= confirmation_price:
                    is_confirmed = True
                    confirmation_candle = current_candle

            if is_confirmed and confirmation_candle is not None:
                # Сигнал подтвержден - открываем позицию
                entry_price = confirmation_price

                if signal_type == 'bull':
                    # Бычий сигнал подтвержден - открываем длинную позицию
                    stop_loss = entry_price - (self.atr_stop_multiple * atr_value)

                    position = {
                        'type': 'long',
                        'entry_index': current_index,
                        'entry_time': confirmation_candle['time'],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'trailing_stop': stop_loss,
                        'atr_value': atr_value,
                        'max_price': entry_price,
                        'pattern': pattern_name,
                        'trend_at_entry': trend_direction
                    }

                    # Если подтверждение произошло на 15m, добавляем эту информацию
                    if m15_confirmation_index is not None:
                        position['m15_confirmation_index'] = m15_confirmation_index
                        position['confirmed_on_m15'] = True

                    self.open_positions.append(position)

                    # Обновляем информацию в all_pattern_signals
                    for s in self.all_pattern_signals:
                        if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                            s['processed'] = True
                            s['trade_opened'] = True
                            s['confirmation_time'] = confirmation_candle['time']
                            s['confirmation_index'] = current_index
                            s['entry_price'] = entry_price
                            s['stop_loss'] = stop_loss
                            s['trend_at_entry'] = trend_direction
                            if m15_confirmation_index is not None:
                                s['confirmed_on_m15'] = True
                                s['m15_confirmation_index'] = m15_confirmation_index
                            break

                    # После открытия позиции удаляем все ожидающие бычьи сигналы
                    # с тем же паттерном, чтобы избежать дублирования
                    updated_pending_signals = [s for s in updated_pending_signals
                                               if not (s['type'] == 'bull' and s['pattern'] == pattern_name)]

                elif signal_type == 'bear':
                    # Медвежий сигнал подтвержден - открываем короткую позицию
                    stop_loss = entry_price + (self.atr_stop_multiple * atr_value)

                    position = {
                        'type': 'short',
                        'entry_index': current_index,
                        'entry_time': confirmation_candle['time'],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'trailing_stop': stop_loss,
                        'atr_value': atr_value,
                        'min_price': entry_price,
                        'pattern': pattern_name,
                        'trend_at_entry': trend_direction
                    }

                    # Если подтверждение произошло на 15m, добавляем эту информацию
                    if m15_confirmation_index is not None:
                        position['m15_confirmation_index'] = m15_confirmation_index
                        position['confirmed_on_m15'] = True

                    self.open_positions.append(position)

                    # Обновляем информацию в all_pattern_signals
                    for s in self.all_pattern_signals:
                        if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                            s['processed'] = True
                            s['trade_opened'] = True
                            s['confirmation_time'] = confirmation_candle['time']
                            s['confirmation_index'] = current_index
                            s['entry_price'] = entry_price
                            s['stop_loss'] = stop_loss
                            s['trend_at_entry'] = trend_direction
                            if m15_confirmation_index is not None:
                                s['confirmed_on_m15'] = True
                                s['m15_confirmation_index'] = m15_confirmation_index
                            break

                    # После открытия позиции удаляем все ожидающие медвежьи сигналы
                    # с тем же паттерном, чтобы избежать дублирования
                    updated_pending_signals = [s for s in updated_pending_signals
                                               if not (s['type'] == 'bear' and s['pattern'] == pattern_name)]

            else:
                # Сигнал еще не подтвержден, добавляем его обратно в список ожидающих
                updated_pending_signals.append(signal)

                # Обновляем информацию в all_pattern_signals о том, что сигнал все еще в ожидании
                for s in self.all_pattern_signals:
                    if s['index'] == signal_index and s['pattern'] == pattern_name and s['type'] == signal_type:
                        s['still_pending'] = True
                        s['elapsed_bars'] = elapsed_bars
                        break

        # Обновляем список ожидающих сигналов
        self.pending_signals = updated_pending_signals

    def _calculate_trend(self, df, current_index):
        """
        Определяет тренд с помощью линейной регрессии

        Args:
            df (DataFrame): DataFrame с данными
            current_index (int): Индекс текущей свечи

        Returns:
            tuple: (направление тренда, сила тренда, данные о тренде)
        """
        if current_index < self.trend_lookback:
            return "flat", 0, None  # Недостаточно данных

        # Берем n последних свечей
        prices = df.iloc[current_index - self.trend_lookback + 1:current_index + 1]['close'].values
        x = np.arange(len(prices))

        # Рассчитываем линейную регрессию
        # Функция np.polyfit выполняет полиномиальную регрессию,
        # где третий параметр (1) указывает на степень полинома (в данном случае - линейная).
        # Она возвращает коэффициенты полинома, где slope (наклон) - это угловой коэффициент,
        # а intercept - смещение (точка пересечения с осью Y).
        slope, intercept = np.polyfit(x, prices, 1)

        # Оцениваем уверенность (R^2)
        # Этот блок рассчитывает коэффициент детерминации (R²),
        # который показывает, насколько хорошо наша модель объясняет вариацию данных:
        # y_pred - предсказанные значения на основе линейной регрессии
        # ss_tot - общая сумма квадратов отклонений от среднего (total sum of squares)
        # ss_res - сумма квадратов остатков (residual sum of squares)
        # r_squared = 1 - (ss_res / ss_tot) - формула для расчета R²
        # R² принимает значения от 0 до 1, где 1 означает, что модель идеально описывает данные.

        y_pred = x * slope + intercept
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        ss_res = np.sum((prices - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Данные для отрисовки тренда
        trend_info = (slope, intercept, x[0], x[-1], prices[0], prices[-1])

        # Определяем направление тренда с порогом значимости 
        # (порог задается в config TREND_R_SQUARED)
        if slope > 0 and r_squared > self.trend_r_squared:
            return "up", r_squared, trend_info
        elif slope < 0 and r_squared > self.trend_r_squared:
            return "down", r_squared, trend_info
        else:
            return "flat", r_squared, trend_info

    def _update_open_positions_ver4(self, df, current_index):
        """
        Обновляет открытые позиции, проверяя стоп-лоссы и обновляя трейлинг-стопы

        Args:
            df (DataFrame): DataFrame с данными
            current_index (int): Индекс текущей свечи
        """
        current_candle = df.iloc[current_index]





        # Создаем новый список для обновления open_positions
        updated_positions = []

        for position in self.open_positions:
            position_type = position['type']
            is_closed = False

            # Проверяем, сработал ли стоп-лосс
            if position_type == 'long' and current_candle['low'] <= position['trailing_stop']:
                # Длинная позиция закрыта по стопу
                exit_price = position['trailing_stop']  # Предполагаем исполнение по цене стопа
                is_closed = True
                exit_reason = 'stop_loss'
            elif position_type == 'short' and current_candle['high'] >= position['trailing_stop']:
                # Короткая позиция закрыта по стопу
                exit_price = position['trailing_stop']  # Предполагаем исполнение по цене стопа
                is_closed = True
                exit_reason = 'stop_loss'

            # Если позиция закрыта, добавляем ее в историю сделок
            if is_closed:
                # Рассчитываем прибыль/убыток
                if position_type == 'long':
                    profit_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    max_profit_pct = (position['max_price'] - position['entry_price']) / position['entry_price'] * 100
                else:
                    profit_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
                    max_profit_pct = (position['entry_price'] - position['min_price']) / position['entry_price'] * 100

                # Добавляем сделку в историю
                self.trades.append({
                    'type': position_type,
                    'entry_index': position['entry_index'],
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_index': current_index,
                    'exit_time': current_candle['time'],
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'max_profit_pct': max_profit_pct,
                    'bars_held': current_index - position['entry_index'],
                    'exit_reason': exit_reason,
                    'pattern': position['pattern'],
                    'atr_value': position['atr_value']
                })
            else:
                # Позиция остается открытой, обновляем трейлинг-стоп и добавляем в обновленный список
                if position_type == 'long':
                    # Обновляем максимальную достигнутую цену
                    position['max_price'] = max(position['max_price'], current_candle['high'])

                    # Обновляем трейлинг-стоп, если цена закрытия выросла
                    if current_candle['close'] > position['entry_price'] and current_candle['close'] > \
                            df.iloc[current_index - 1]['close']:
                        # new_stop = current_candle['close'] - (self.atr_stop_multiple * position['atr_value'])
                        new_stop = current_candle['low'] - (self.atr_stop_multiple * position['atr_value'])
                        position['trailing_stop'] = max(position['trailing_stop'], new_stop)
                else:
                    # Обновляем минимальную достигнутую цену
                    position['min_price'] = min(position['min_price'], current_candle['low'])

                    # Обновляем трейлинг-стоп, если цена закрытия упала
                    if current_candle['close'] < position['entry_price'] and current_candle['close'] < \
                            df.iloc[current_index - 1]['close']:
                        # new_stop = current_candle['close'] + (self.atr_stop_multiple * position['atr_value'])
                        new_stop = current_candle['high'] + (self.atr_stop_multiple * position['atr_value'])
                        position['trailing_stop'] = min(position['trailing_stop'], new_stop)

                updated_positions.append(position)

        # Обновляем список открытых позиций
        self.open_positions = updated_positions

    def _update_open_positions(self, df, current_index):
        """
        Обновляет открытые позиции, проверяя стоп-лоссы и обновляя трейлинг-стопы
        Если доступны 15m данные, использует их для более точного управления позицией

        Args:
            df (DataFrame): DataFrame с данными 1h
            current_index (int): Индекс текущей 1h свечи
        """
        current_candle = df.iloc[current_index]

        # Получаем 15m свечи для текущей 1h свечи, если они доступны
        m15_indices = []
        if hasattr(self, 'm15_df') and self.m15_df is not None:
            h1_time = df.iloc[current_index]['time']
            next_h1_time = df.iloc[current_index + 1]['time'] if current_index < len(df) - 1 else None

            # Выведем диагностическую информацию
            print(f"Поиск 15m свечей для h1 свечи {current_index}, h1_time={h1_time}, next_h1_time={next_h1_time}")

            # Вместо перебора всех строк используем прямую фильтрацию
            # Это может быть на порядки быстрее
            filtered_df = self.m15_df[
                (self.m15_df['time'] >= h1_time) &
                ((next_h1_time is None) | (self.m15_df['time'] < next_h1_time))
                ]
            m15_indices = filtered_df.index.tolist()

            # Ограничим количество для производительности
            if len(m15_indices) > 10:
                print(f"Ограничение 15m свечей с {len(m15_indices)} до 10 для оптимизации")
                m15_indices = m15_indices[:10]

        # Создаем новый список для обновления open_positions
        updated_positions = []

        for position in self.open_positions:
            position_type = position['type']
            is_closed = False
            exit_price = None
            exit_reason = None
            exit_time = None
            exit_index = None

            # Если у нас есть 15m данные, проверяем каждую 15m свечу
            if m15_indices:
                for m15_idx in m15_indices:
                    m15_candle = self.m15_df.iloc[m15_idx]

                    # Проверяем, сработал ли стоп-лосс на 15m свече
                    if position_type == 'long' and m15_candle['low'] <= position['trailing_stop']:
                        # Длинная позиция закрыта по стопу
                        exit_price = position['trailing_stop']
                        is_closed = True
                        exit_reason = 'stop_loss_15m'
                        exit_time = m15_candle['time']
                        exit_index = current_index  # Соответствующий 1h индекс
                        break

                    elif position_type == 'short' and m15_candle['high'] >= position['trailing_stop']:
                        # Короткая позиция закрыта по стопу
                        exit_price = position['trailing_stop']
                        is_closed = True
                        exit_reason = 'stop_loss_15m'
                        exit_time = m15_candle['time']
                        exit_index = current_index  # Соответствующий 1h индекс
                        break

                    # Обновляем максимальные/минимальные цены и трейлинг-стоп на основе 15m данных
                    if not is_closed:
                        if position_type == 'long':
                            # Обновляем максимальную достигнутую цену
                            position['max_price'] = max(position['max_price'], m15_candle['high'])

                            # Обновляем трейлинг-стоп, если цена закрытия выросла
                            if m15_candle['close'] > position['entry_price']:
                                new_stop = m15_candle['low'] - (self.atr_stop_multiple * position['atr_value'])
                                position['trailing_stop'] = max(position['trailing_stop'], new_stop)
                        else:  # short
                            # Обновляем минимальную достигнутую цену
                            position['min_price'] = min(position['min_price'], m15_candle['low'])

                            # Обновляем трейлинг-стоп, если цена закрытия упала
                            if m15_candle['close'] < position['entry_price']:
                                new_stop = m15_candle['high'] + (self.atr_stop_multiple * position['atr_value'])
                                position['trailing_stop'] = min(position['trailing_stop'], new_stop)
            else:
                # Если 15m данных нет, используем стандартную логику проверки на 1h
                if position_type == 'long' and current_candle['low'] <= position['trailing_stop']:
                    # Длинная позиция закрыта по стопу
                    exit_price = position['trailing_stop']
                    is_closed = True
                    exit_reason = 'stop_loss'
                    exit_time = current_candle['time']
                    exit_index = current_index

                elif position_type == 'short' and current_candle['high'] >= position['trailing_stop']:
                    # Короткая позиция закрыта по стопу
                    exit_price = position['trailing_stop']
                    is_closed = True
                    exit_reason = 'stop_loss'
                    exit_time = current_candle['time']
                    exit_index = current_index

                # Если позиция не закрыта, обновляем трейлинг-стоп и максимумы/минимумы
                if not is_closed:
                    if position_type == 'long':
                        # Обновляем максимальную достигнутую цену
                        position['max_price'] = max(position['max_price'], current_candle['high'])

                        # Обновляем трейлинг-стоп, если цена закрытия выросла
                        if current_candle['close'] > position['entry_price'] and current_candle['close'] > \
                                df.iloc[current_index - 1]['close']:
                            new_stop = current_candle['low'] - (self.atr_stop_multiple * position['atr_value'])
                            position['trailing_stop'] = max(position['trailing_stop'], new_stop)
                    else:  # short
                        # Обновляем минимальную достигнутую цену
                        position['min_price'] = min(position['min_price'], current_candle['low'])

                        # Обновляем трейлинг-стоп, если цена закрытия упала
                        if current_candle['close'] < position['entry_price'] and current_candle['close'] < \
                                df.iloc[current_index - 1]['close']:
                            new_stop = current_candle['high'] + (self.atr_stop_multiple * position['atr_value'])
                            position['trailing_stop'] = min(position['trailing_stop'], new_stop)

            # Если позиция закрыта, добавляем ее в историю сделок
            if is_closed:
                # Рассчитываем прибыль/убыток
                if position_type == 'long':
                    profit_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    max_profit_pct = (position['max_price'] - position['entry_price']) / position['entry_price'] * 100
                else:  # short
                    profit_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
                    max_profit_pct = (position['entry_price'] - position['min_price']) / position['entry_price'] * 100

                # Добавляем сделку в историю
                trade = {
                    'type': position_type,
                    'entry_index': position['entry_index'],
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_index': exit_index,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'max_profit_pct': max_profit_pct,
                    'bars_held': exit_index - position['entry_index'],
                    'exit_reason': exit_reason,
                    'pattern': position['pattern'],
                    'atr_value': position['atr_value']
                }

                # Добавляем информацию о использовании 15m таймфрейма, если она есть
                if 'confirmed_on_m15' in position:
                    trade['confirmed_on_m15'] = position['confirmed_on_m15']
                if 'm15_confirmation_index' in position:
                    trade['m15_confirmation_index'] = position['m15_confirmation_index']
                if m15_indices:
                    trade['closed_on_15m'] = True

                self.trades.append(trade)
            else:
                # Позиция остается открытой
                updated_positions.append(position)

        # Обновляем список открытых позиций
        self.open_positions = updated_positions

    def _look_for_new_signals(self, df, current_index, pattern_name):
        """
        Ищет новые сигналы паттернов и записывает все найденные сигналы

        Args:
            df (DataFrame): DataFrame с данными
            current_index (int): Индекс текущей свечи
            pattern_name (str): Название паттерна для поиска
            
            В зависимости от выбранного режима:

            1. with_trend - будут создаваться сигналы, только если они совпадают с текущим трендом
            2. against_trend - будут создаваться сигналы, только если они противоположны текущему тренду
            3. ignore_trend - все сигналы будут создаваться независимо от тренда

            Также в функцию добавлено сохранение информации о режиме фильтрации тренда в записях о сигналах, 
            что может быть полезно для последующего анализа.
            
        """
        # Получаем текущую свечу
        current_candle = df.iloc[current_index]

        # Проверяем наличие паттерна
        pattern_value = current_candle[pattern_name]

        # Если паттерн не обнаружен, просто выходим
        if pattern_value == 0:
            return

        # Получаем значение ATR и других индикаторов
        atr_value = current_candle['atr']

        # Получаем значение RSI (если доступно)
        rsi_value = current_candle['rsi'] if 'rsi' in current_candle else None

        # Определяем текущий тренд
        trend_direction, trend_strength, _ = self._calculate_trend(df, current_index)

        # Проверяем, нет ли уже открытой позиции с этим паттерном
        has_open_position = any(position['pattern'] == pattern_name for position in self.open_positions)

        # Проверяем, нет ли уже ожидающего сигнала с этим паттерном
        has_pending_signal = any(signal['pattern'] == pattern_name and signal['index'] == current_index
                                 for signal in self.pending_signals)

        # Проверяем соответствие сигнала тренду в зависимости от режима фильтрации
        signal_matches_trend_requirements = True
        trend_reason = None

        if self.trend_filter_mode == 'with_trend':
            # Для режима 'с трендом' проверяем совпадение с направлением тренда
            if ((pattern_value > 0 and trend_direction != 'up') or
                    (pattern_value < 0 and trend_direction != 'down')):
                signal_matches_trend_requirements = False
                trend_reason = 'signal_against_trend'

        elif self.trend_filter_mode == 'against_trend':
            # Для режима 'против тренда' проверяем противоположность направлению тренда
            if ((pattern_value > 0 and trend_direction != 'down') or
                    (pattern_value < 0 and trend_direction != 'up')):
                signal_matches_trend_requirements = False
                trend_reason = 'signal_with_trend'

        # В режиме 'ignore_trend' не делаем никаких проверок на соответствие тренду

        # Определяем причину, почему сигнал может быть проигнорирован
        ignore_reason = None
        if has_open_position:
            ignore_reason = 'position_already_open'
        elif has_pending_signal:
            ignore_reason = 'signal_already_pending'
        elif not signal_matches_trend_requirements:
            ignore_reason = trend_reason
        elif len(self.pending_signals) >= 5:
            ignore_reason = 'too_many_pending_signals'
        elif pattern_value > 0 and any(p['type'] == 'long' for p in self.open_positions):
            ignore_reason = 'long_position_already_exists'
        elif pattern_value < 0 and any(p['type'] == 'short' for p in self.open_positions):
            ignore_reason = 'short_position_already_exists'

        # Рассчитываем цену подтверждения
        confirmation_price = None
        if pattern_value > 0:  # Бычий сигнал
            confirmation_price = current_candle['close'] + (self.atr_confirmation_multiple * atr_value)
        elif pattern_value < 0:  # Медвежий сигнал
            confirmation_price = current_candle['close'] - (self.atr_confirmation_multiple * atr_value)

        # Записываем сигнал в общий список, независимо от того, будет ли по нему открыта сделка
        signal_data = {
            'pattern': pattern_name,
            'type': 'bull' if pattern_value > 0 else 'bear',
            'index': current_index,
            'time': current_candle['time'],
            'price': current_candle['close'],
            'confirmation_price': confirmation_price,
            'atr_value': atr_value,
            'rsi': rsi_value,
            'trend': trend_direction,
            'trend_strength': trend_strength,
            'processed': False,  # Сигнал еще не обработан
            'trade_opened': False,  # Сделка еще не открыта
            'ignore_reason': ignore_reason,
            'trend_filter_mode': self.trend_filter_mode  # Сохраняем режим фильтрации тренда
        }

        # Добавляем сигнал в список всех сигналов
        self.all_pattern_signals.append(signal_data)

        # Если есть причина игнорировать сигнал, выходим
        if ignore_reason:
            return

        # Ограничиваем количество одновременных сигналов
        if len(self.pending_signals) >= 5:
            # Сортируем существующие сигналы по времени (старые - первые)
            self.pending_signals.sort(key=lambda x: x['index'])
            # Оставляем только 4 самых новых сигнала
            self.pending_signals = self.pending_signals[-4:]

        # Обрабатываем бычий сигнал
        if pattern_value > 0:
            # Проверяем, нет ли уже открытой длинной позиции
            has_long_position = any(p['type'] == 'long' for p in self.open_positions)

            # Добавляем сигнал только если нет открытых длинных позиций
            if not has_long_position:
                self.pending_signals.append({
                    'type': 'bull',
                    'index': current_index,
                    'time': current_candle['time'],
                    'close_price': current_candle['close'],
                    'confirmation_price': confirmation_price,
                    'atr_value': atr_value,
                    'pattern': pattern_name,
                    'added_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rsi': rsi_value,
                    'trend': trend_direction,
                    'trend_filter_mode': self.trend_filter_mode
                })

        # Обрабатываем медвежий сигнал
        elif pattern_value < 0:
            # Проверяем, нет ли уже открытой короткой позиции
            has_short_position = any(p['type'] == 'short' for p in self.open_positions)

            # Добавляем сигнал только если нет открытых коротких позиций
            if not has_short_position:
                self.pending_signals.append({
                    'type': 'bear',
                    'index': current_index,
                    'time': current_candle['time'],
                    'close_price': current_candle['close'],
                    'confirmation_price': confirmation_price,
                    'atr_value': atr_value,
                    'pattern': pattern_name,
                    'added_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rsi': rsi_value,
                    'trend': trend_direction,
                    'trend_filter_mode': self.trend_filter_mode
                })

    def _close_remaining_positions(self, df):
        """
        Закрывает все оставшиеся открытые позиции по последней цене

        Args:
            df (DataFrame): DataFrame с данными
        """
        if not self.open_positions:
            return

        last_candle = df.iloc[-1]
        last_index = len(df) - 1

        for position in self.open_positions:
            position_type = position['type']
            exit_price = last_candle['close']

            # Рассчитываем прибыль/убыток
            if position_type == 'long':
                profit_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                max_profit_pct = (position['max_price'] - position['entry_price']) / position['entry_price'] * 100
            else:
                profit_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
                max_profit_pct = (position['entry_price'] - position['min_price']) / position['entry_price'] * 100

            # Добавляем сделку в историю
            self.trades.append({
                'type': position_type,
                'entry_index': position['entry_index'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_index': last_index,
                'exit_time': last_candle['time'],
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'max_profit_pct': max_profit_pct,
                'bars_held': last_index - position['entry_index'],
                'exit_reason': 'end_of_data',
                'pattern': position['pattern'],
                'atr_value': position['atr_value']
            })

        # Очищаем список открытых позиций
        self.open_positions = []

    def _calculate_statistics(self):
        """
        Рассчитывает статистику по результатам бэктестинга

        Returns:
            dict: Статистика по результатам бэктестинга
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_bars_held': 0,
                'max_consecutive_losses': 0,
                'bull_trades': 0,
                'bear_trades': 0,
                'bull_win_rate': 0,
                'bear_win_rate': 0,
                'trades': []
            }

        # Преобразуем список сделок в DataFrame для удобства анализа
        trades_df = pd.DataFrame(self.trades)

        # Основные метрики
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_pct'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        avg_profit = trades_df['profit_pct'].mean()
        total_profit = trades_df['profit_pct'].sum()
        avg_bars_held = trades_df['bars_held'].mean()

        # Статистика по типам сделок
        bull_trades = len(trades_df[trades_df['type'] == 'long'])
        bear_trades = len(trades_df[trades_df['type'] == 'short'])

        bull_win_rate = 0
        if bull_trades > 0:
            bull_wins = len(trades_df[(trades_df['type'] == 'long') & (trades_df['profit_pct'] > 0)])
            bull_win_rate = (bull_wins / bull_trades) * 100

        bear_win_rate = 0
        if bear_trades > 0:
            bear_wins = len(trades_df[(trades_df['type'] == 'short') & (trades_df['profit_pct'] > 0)])
            bear_win_rate = (bear_wins / bear_trades) * 100

        # Расчет максимальной просадки
        equity_curve = (1 + trades_df['profit_pct'] / 100).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1) * 100
        max_drawdown = drawdown.min()

        # Расчет коэффициента Шарпа (годовой, предполагаем 252 торговых дня)
        excess_returns = trades_df['profit_pct'] / 100
        if excess_returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        else:
            sharpe_ratio = 0

        # Расчет максимального количества последовательных убытков
        consecutive_losses = 0
        max_consecutive_losses = 0

        for profit in trades_df['profit_pct']:
            if profit <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_bars_held': avg_bars_held,
            'max_consecutive_losses': max_consecutive_losses,
            'bull_trades': bull_trades,
            'bear_trades': bear_trades,
            'bull_win_rate': bull_win_rate,
            'bear_win_rate': bear_win_rate,
            'trend_filter_mode': self.trend_filter_mode,
            'trades': self.trades
        }

    def save_all_signals_to_csv(self, filepath):
        """
        Сохраняет информацию о всех сигналах паттерна в CSV файл

        Args:
            filepath (str): Путь для сохранения файла
        """
        if not self.all_pattern_signals:
            return

        signals_df = pd.DataFrame(self.all_pattern_signals)

        # Добавляем информацию о сделках к связанным сигналам
        for idx, signal in enumerate(self.all_pattern_signals):
            if signal['trade_opened']:
                # Ищем соответствующую сделку
                for trade in self.trades:
                    if (trade['entry_index'] == signal['index'] and
                            trade['pattern'] == signal['pattern'] and
                            ((signal['type'] == 'bull' and trade['type'] == 'long') or
                             (signal['type'] == 'bear' and trade['type'] == 'short'))):
                        # Добавляем данные о сделке
                        signals_df.loc[idx, 'exit_index'] = trade['exit_index']
                        signals_df.loc[idx, 'exit_time'] = trade['exit_time']
                        signals_df.loc[idx, 'exit_price'] = trade['exit_price']
                        signals_df.loc[idx, 'profit_pct'] = trade['profit_pct']
                        signals_df.loc[idx, 'bars_held'] = trade['bars_held']
                        signals_df.loc[idx, 'exit_reason'] = trade['exit_reason']
                        break

        # Сохраняем в CSV
        signals_df.to_csv(filepath, index=False)

    def optimize_parameters(self, df, pattern_name, param_grid):
        """
        Оптимизирует параметры стратегии через перебор сетки параметров

        Args:
            df (DataFrame): DataFrame с данными
            pattern_name (str): Название паттерна для тестирования
            param_grid (dict): Словарь с параметрами для оптимизации
                Например: {
                    'atr_period': [10, 14, 20],
                    'atr_confirmation_multiple': [1.0, 1.5, 2.0],
                    'atr_stop_multiple': [1.0, 1.5, 2.0],
                    'confirmation_period': [3, 5, 7]
                }

        Returns:
            dict: Лучшие параметры и результаты оптимизации
        """
        from itertools import product

        # Получаем списки значений для каждого параметра
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Хранилище для результатов
        results = []

        # Перебираем все комбинации параметров
        for params in product(*param_values):
            # Создаем словарь параметров
            params_dict = dict(zip(param_keys, params))

            # Создаем экземпляр бэктестера с текущими параметрами
            backtester = ATRBacktester(
                atr_period=params_dict.get('atr_period', self.atr_period),
                atr_confirmation_multiple=params_dict.get('atr_confirmation_multiple', self.atr_confirmation_multiple),
                atr_stop_multiple=params_dict.get('atr_stop_multiple', self.atr_stop_multiple),
                confirmation_period=params_dict.get('confirmation_period', self.confirmation_period),
                position_size=params_dict.get('position_size', self.position_size)
            )

            # Запускаем бэктестинг
            stats = backtester.test_pattern(df, pattern_name)

            # Добавляем параметры к статистике
            stats.update(params_dict)

            # Добавляем результаты для текущей комбинации параметров
            results.append(stats)

        # Преобразуем результаты в DataFrame
        results_df = pd.DataFrame(results)

        # Сортируем по общей прибыли (можно изменить критерий)
        results_df = results_df.sort_values('total_profit', ascending=False)
        #results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        # Получаем лучшие параметры
        best_params = results_df.iloc[0][param_keys].to_dict()

        return {
            'best_params': best_params,
            'best_stats': results_df.iloc[0].to_dict(),
            'all_results': results_df
        }

    def test_all_patterns(self, df, pattern_names):
        """
        Проводит бэктестинг для нескольких паттернов и возвращает сравнительную статистику

        Args:
            df (DataFrame): DataFrame с данными свечей и ATR
            pattern_names (list): Список названий паттернов для тестирования

        Returns:
            DataFrame: Сравнительная статистика по всем паттернам
        """
        results = []

        for pattern in pattern_names:
            if pattern in df.columns:
                stats = self.test_pattern(df, pattern)
                stats['pattern'] = pattern
                results.append(stats)

        # Создаем DataFrame из результатов
        if results:
            results_df = pd.DataFrame(results)
            return results_df.sort_values('total_profit', ascending=False)
            #return results_df.sort_values('win_rate', ascending=False)
        else:
            return pd.DataFrame()

    def _get_m15_indices_for_h1(self, h1_index):
        """
        Находит индексы 15-минутных свечей, соответствующие данной часовой свече

        Args:
            h1_index (int): Индекс 1h свечи

        Returns:
            list: Список индексов соответствующих 15m свечей
        """
        if not hasattr(self, 'm15_df') or self.m15_df is None:
            return []

        h1_time = self.main_df.iloc[h1_index]['time']
        next_h1_time = self.main_df.iloc[h1_index + 1]['time'] if h1_index < len(self.main_df) - 1 else None

        # Используйте фильтрацию по условию вместо цикла
        mask = (self.m15_df['time'] >= h1_time) & (next_h1_time is None or self.m15_df['time'] < next_h1_time)

        return self.m15_df[mask].index.tolist()


    @staticmethod
    def generate_equity_curve(trades_list):
        """
        Генерирует кривую доходности на основе списка сделок

        Args:
            trades_list (list): Список сделок

        Returns:
            DataFrame: DataFrame с кривой доходности
        """
        if not trades_list:
            return pd.DataFrame()

        # Создаем DataFrame из списка сделок
        trades_df = pd.DataFrame(trades_list)

        # Сортируем по времени выхода
        trades_df = trades_df.sort_values('exit_time')

        # Рассчитываем кумулятивную кривую доходности
        trades_df['equity'] = (1 + trades_df['profit_pct'] / 100).cumprod()
        trades_df['drawdown'] = trades_df['equity'].div(trades_df['equity'].cummax()).sub(1).mul(100)

        return trades_df