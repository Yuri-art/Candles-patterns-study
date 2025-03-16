import pandas as pd
import numpy as np


class TradingStrategy:
    """
    Базовый класс для торговых стратегий
    """

    def __init__(self, name="BaseStrategy"):
        self.name = name
        self.positions = []
        self.trades = []

    def generate_signals(self, df):
        """
        Генерирует торговые сигналы

        Args:
            df (DataFrame): DataFrame с ценовыми данными и индикаторами

        Returns:
            DataFrame: DataFrame с добавленными торговыми сигналами
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def backtest(self, df):
        """
        Выполняет бэктестинг стратегии

        Args:
            df (DataFrame): DataFrame с ценовыми данными и сигналами

        Returns:
            dict: Результаты бэктестинга
        """
        raise NotImplementedError("Subclass must implement abstract method")


class CandlePatternStrategy(TradingStrategy):
    """
    Стратегия, основанная на свечных паттернах
    """

    def __init__(self, pattern_names, exit_strategy="color_change", stop_loss_pct=None, take_profit_pct=None):
        """
        Инициализирует стратегию свечных паттернов

        Args:
            pattern_names (list): Список названий паттернов для отслеживания
            exit_strategy (str): Стратегия выхода из позиции
            stop_loss_pct (float, optional): Процент для стоп-лосса
            take_profit_pct (float, optional): Процент для тейк-профита
        """
        super().__init__(name=f"CandlePattern_{'-'.join(pattern_names)}")
        self.pattern_names = pattern_names
        self.exit_strategy = exit_strategy
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate_signals(self, df):
        """
        Генерирует сигналы на основе выбранных свечных паттернов

        Args:
            df (DataFrame): DataFrame с ценовыми данными и паттернами

        Returns:
            DataFrame: DataFrame с добавленными сигналами
        """
        # Создаем копию DataFrame для добавления сигналов
        result_df = df.copy()

        # Добавляем столбцы для сигналов
        result_df['buy_signal'] = 0
        result_df['sell_signal'] = 0

        # Для каждого паттерна проверяем наличие сигналов
        for pattern in self.pattern_names:
            if pattern in result_df.columns:
                # Бычьи сигналы (положительные значения)
                result_df.loc[result_df[pattern] > 0, 'buy_signal'] = 1

                # Медвежьи сигналы (отрицательные значения)
                result_df.loc[result_df[pattern] < 0, 'sell_signal'] = 1

        return result_df

    def apply_exit_strategy(self, df):
        """
        Применяет стратегию выхода к данным

        Args:
            df (DataFrame): DataFrame с ценовыми данными и сигналами

        Returns:
            DataFrame: DataFrame с добавленными сигналами выхода
        """
        result_df = df.copy()
        result_df['exit_buy'] = 0  # Сигнал выхода из длинной позиции
        result_df['exit_sell'] = 0  # Сигнал выхода из короткой позиции

        # Применяем стратегию выхода в зависимости от выбранной опции
        if self.exit_strategy == "color_change":
            # Выход из длинной позиции при первой красной свече
            # Выход из короткой позиции при первой зеленой свече
            for i in range(1, len(result_df)):
                if not result_df.iloc[i]['is_green'] and result_df.iloc[i - 1]['is_green']:
                    result_df.iloc[i, result_df.columns.get_loc('exit_buy')] = 1

                if result_df.iloc[i]['is_green'] and not result_df.iloc[i - 1]['is_green']:
                    result_df.iloc[i, result_df.columns.get_loc('exit_sell')] = 1

        elif self.exit_strategy == "trailing_stop":
            # Реализация трейлинг-стопа
            # Для каждой открытой позиции отслеживаем максимум/минимум и выходим при откате
            pass

        elif self.exit_strategy == "fixed_bars":
            # Выход через фиксированное количество баров
            pass

        # Применяем стоп-лосс и тейк-профит, если указаны
        if self.stop_loss_pct is not None or self.take_profit_pct is not None:
            pass

        return result_df

    def backtest(self, df):
        """
        Выполняет бэктестинг стратегии

        Args:
            df (DataFrame): DataFrame с ценовыми данными и сигналами

        Returns:
            dict: Результаты бэктестинга
        """
        # Применяем стратегию выхода
        df_with_exits = self.apply_exit_strategy(df)

        # Инициализируем списки для хранения сделок
        trades = []
        open_position = None

        # Симулируем торговлю на исторических данных
        for i in range(1, len(df_with_exits)):
            row = df_with_exits.iloc[i]
            prev_row = df_with_exits.iloc[i - 1]

            # Если есть сигнал на покупку и нет открытой позиции
            if row['buy_signal'] == 1 and open_position is None:
                open_position = {
                    'type': 'buy',
                    'entry_price': row['close'],
                    'entry_time': row['time'],
                    'entry_index': i
                }

            # Если есть сигнал на продажу и нет открытой позиции
            elif row['sell_signal'] == 1 and open_position is None:
                open_position = {
                    'type': 'sell',
                    'entry_price': row['close'],
                    'entry_time': row['time'],
                    'entry_index': i
                }

            # Если есть открытая длинная позиция и сигнал на выход
            elif open_position is not None and open_position['type'] == 'buy' and row['exit_buy'] == 1:
                # Закрываем позицию
                trade = {
                    'type': open_position['type'],
                    'entry_price': open_position['entry_price'],
                    'entry_time': open_position['entry_time'],
                    'exit_price': row['close'],
                    'exit_time': row['time'],
                    'bars_held': i - open_position['entry_index'],
                    'profit_pct': (row['close'] - open_position['entry_price']) / open_position['entry_price'] * 100
                }
                trades.append(trade)
                open_position = None

            # Если есть открытая короткая позиция и сигнал на выход
            elif open_position is not None and open_position['type'] == 'sell' and row['exit_sell'] == 1:
                # Закрываем позицию
                trade = {
                    'type': open_position['type'],
                    'entry_price': open_position['entry_price'],
                    'entry_time': open_position['entry_time'],
                    'exit_price': row['close'],
                    'exit_time': row['time'],
                    'bars_held': i - open_position['entry_index'],
                    'profit_pct': (open_position['entry_price'] - row['close']) / open_position['entry_price'] * 100
                }
                trades.append(trade)
                open_position = None

        # Если в конце периода осталась открытая позиция, закрываем её
        if open_position is not None:
            last_row = df_with_exits.iloc[-1]
            trade = {
                'type': open_position['type'],
                'entry_price': open_position['entry_price'],
                'entry_time': open_position['entry_time'],
                'exit_price': last_row['close'],
                'exit_time': last_row['time'],
                'bars_held': len(df_with_exits) - 1 - open_position['entry_index'],
                'profit_pct': (last_row['close'] - open_position['entry_price']) / open_position['entry_price'] * 100 if
                open_position['type'] == 'buy' else (open_position['entry_price'] - last_row['close']) / open_position[
                    'entry_price'] * 100
            }
            trades.append(trade)

        # Вычисляем статистику по сделкам
        trades_df = pd.DataFrame(trades)

        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'trades': trades
            }

        # Основные метрики
        win_rate = len(trades_df[trades_df['profit_pct'] > 0]) / len(trades_df) * 100
        avg_profit = trades_df['profit_pct'].mean()
        total_profit = trades_df['profit_pct'].sum()

        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': total_profit,
            'max_drawdown': self._calculate_max_drawdown(trades_df),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades_df),
            'trades': trades
        }

    def _calculate_max_drawdown(self, trades_df):
        """Рассчитывает максимальную просадку на основе серии сделок"""
        if trades_df.empty:
            return 0

        # Создаем кумулятивную кривую доходности
        cumulative = (1 + trades_df['profit_pct'] / 100).cumprod()

        # Находим максимумы и спады
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1) * 100

        return drawdown.min() if not drawdown.empty else 0

    def _calculate_sharpe_ratio(self, trades_df, risk_free_rate=0, periods_per_year=252):
        """Рассчитывает коэффициент Шарпа"""
        if trades_df.empty:
            return 0

        returns = trades_df['profit_pct'] / 100

        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(
            periods_per_year) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0