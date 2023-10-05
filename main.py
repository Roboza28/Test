import pandas as pd
import numpy as np
import os
import statistics
from scipy.signal import find_peaks
from sklearn import preprocessing
from scipy.fft import rfft, rfftfreq
from numpy import pi, sin, cos
from math import radians
import csv

g = 9.81
VELOCITY_0 = 0.4
RHO = 998
M_C = 3050
H_C = 0.5
L = 2
V_LIC_ST = 1.044
H_LIC_ST = 0.18
V_LIC_MOVE = 0.261
CUTOFF_FREQ = 10

M_LIC_ST = V_LIC_ST * RHO
M_LIC_MOVE = V_LIC_MOVE * RHO
M_SYST = M_LIC_ST + M_C + M_LIC_MOVE
STATIC_LOAD = M_SYST * g / 2


def address() -> str:
    """ Функция возвращает имя csv файла, который находится в корне проекта.
        При этом при запуске скрипта в корне должен быть лишь один csv файл!
    Returns:
        str: имя найденного в корне csv файла
    """

    current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file)
    for file in os.listdir(current_directory):
        if file.endswith('.csv'):
            name_file = os.path.join(file)
            return name_file


def set_dataframe(name_file: str) -> pd.DataFrame:
    """ Функция возвращает 4 столбца с данными численного эксперимента из csv файла.
    Args:
        name_file: название исходного файла
    Returns:
        dataframe типа (['Время, сек', 'Rz_R, Н', 'Rz_L, Н', 'Rx, Н'])
    """

    return pd.read_csv(name_file, decimal=',', sep=';', encoding='UTF-8', usecols=[*range(4)])


def set_signal_timing_parameters(dataframe: pd.DataFrame) -> tuple[float, int]:
    """ Функция возвращает временные свойства численного эксперимента:
        длина эксперимента в секундах и частота дискретизации
    Args:
        dataframe: данные численного эксперимента
    Returns:
        кортеж типа (длина эксперимента в секундах, частота дискретизации)
    """

    computation_time = dataframe.iloc[:, 0][len(dataframe) - 1] + dataframe.iloc[:, 0][1]
    sample_rate = int(len(dataframe) / computation_time)
    return computation_time, sample_rate


class ClearSignal(object):
    """ Класс, в котором содержатся методы, предназначенные для обработки данных из численного эксперимента.
        Для краткости в силу поиска частоты колебаний и фильтрации, данные будут именовать сигналами.
    """

    def _find_coordinates_peaks(self, time_signal: pd.Series, signal: pd.Series) -> dict[float, float]:
        """ Функция возвращает координаты локальных min/max в виде словаря
        Args:
            time_signal: dataframe с данными времени
            signal: dataframe с данными реакций
        Returns:
            словарь типа {значение времени в точке локального min/max: значение реакций в точке локального min/max}
        """

        peaks_indexes, _ = find_peaks(signal, distance=int(self._average_period(time_signal, signal) * SAMPLE_RATE))
        coordinates_peaks = {}
        for peak_index in peaks_indexes:
            coordinates_peaks[time_signal[peak_index]] = signal[peak_index]
        return coordinates_peaks

    def _average_period(self, time_signal: pd.Series, signal: pd.Series) -> float:
        """ Функция возвращает средний период сигнала
        Args:
            time_signal: dataframe с данными времени
            signal: dataframe с данными реакций
        Returns:
            float равный среднему по медиане периоду.
            Медиана выбрана для минимизации влияния выбросов во время переходного процесса.
        Comments:
            Логический флаг unique_value реализован по причину того, что данные могут колебаться
            около среднего значения и переходить через него несколько раз в ближайших точках,
            что при вычислении периода может сказаться и сильно занизить значение.
            Поэтому переходы вблизи 0.1 уникального значения игнорируются.
            В качестве "нуля" берется среднее арифметическое между сигналов ниже среднего и выше.
        """

        time_of_transition_through_the_mean_value = []
        for index, _ in enumerate(signal[:-1]):
            if signal[index] > STATIC_LOAD > signal[index + 1]:
                unique_value = True
                for i in time_of_transition_through_the_mean_value:
                    if round(i, 1) == round((time_signal[index]+time_signal[index+1]) / 2, 1):
                        unique_value = False
                if unique_value:
                    time_of_transition_through_the_mean_value.append((time_signal[index]+time_signal[index+1]) / 2)

        periods = []
        for index, _ in enumerate(time_of_transition_through_the_mean_value):
            periods.append(abs(
                time_of_transition_through_the_mean_value[index-1] - time_of_transition_through_the_mean_value[index]))
        return statistics.median(periods)

    def _transient_response(self, time_signal: pd.Series, signal: pd.Series) -> float:
        """ Функция возвращает время переходного процесса
        Args:
            time_signal: dataframe с данными времени
            signal: dataframe с данными реакций
        Returns:
            float равный времени переходного процесса
        """

        coord_peaks_signal = self._find_coordinates_peaks(time_signal, signal).items()
        for coord_peak_signal in coord_peaks_signal:
            if abs(coord_peak_signal[1]-STATIC_LOAD) < STATIC_LOAD*0.05:
                return coord_peak_signal[0]

    def _filter_signal(self, time_after_fft: np.ndarray, signal_after_fft: np.ndarray) -> list:
        """ Функция возвращает отфильтрованный сигнал с учетом частоты среза.
        Args:
            time_after_fft: dataframe с данными времени после FFT
            signal_after_fft: dataframe с данными реакций после FFT
        Returns:
            кортеж типа (отфильтрованное время в пространстве Фурье, отфильтрованный сигнал)
        """

        signal_filter = []
        for freq, signal in zip(time_after_fft, signal_after_fft):
            if freq < CUTOFF_FREQ:
                signal_filter.append(signal)
            else:
                signal_filter.append(0)
        return signal_filter


class FrequencyCalculation(ClearSignal):
    """ Класс, в котором содержатся методы, предназначенные для вычисления частоты сигнала и его фильтрации.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe.fillna(0).astype(float)
        t_pp_rz_r = super()._transient_response(self._df['Время, сек'], self._df['Rz_R, Н'])
        t_pp_rz_l = super()._transient_response(self._df['Время, сек'], self._df['Rz_L, Н'])
        self._settling_time = max(t_pp_rz_r, t_pp_rz_l)
        self._duration_clear_signal = COMPUTATION_TIME - self._settling_time
        self._num_of_points_after_filter = int(SAMPLE_RATE * self._duration_clear_signal)
        self._clear_df = self._normal_signal(self._df.loc[self._settling_time * SAMPLE_RATE:])
        freq_1 = self._calc_frequency(self._clear_df['Rz_R, Н'])
        freq_2 = self._calc_frequency(self._clear_df['Rz_L, Н'])
        self._freq = statistics.mean([freq_1, freq_2])

    def _normal_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ Функция возвращает нормализованный dataframe.
        Args:
            dataframe: dataframe, переданный для нормализации
        Returns:
            dataframe типа (['Время, сек', 'Rz_R, Н', 'Rz_L, Н', 'Rx, Н'])
            с нормализованными значениями
        Comments:
            Нормировка происходит для значений от -1 до 1.
        """

        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(dataframe)
        normalize_dataframe = pd.DataFrame(scaler, columns=dataframe.columns.tolist())
        return normalize_dataframe

    def _calc_frequency(self, signal: pd.Series) -> float:
        """ Функция подсчитывает частоту сигнала
        Args:
            signal: dataframe с данными реакций
        Returns:
            float равный частоте колебаний
        Comments:
            time_fft - это фурье-образ столбца времени
            signal_fft это фурье-образ столбца сигнала
        """

        time_fft = rfftfreq(self._num_of_points_after_filter, 1/SAMPLE_RATE)
        signal_fft = rfft(np.array(signal))
        clear_signal = super()._filter_signal(time_fft, signal_fft)
        time_indexes_peaks, _ = find_peaks(np.abs(clear_signal))
        time_fft_peaks = [time_fft[i] for i in time_indexes_peaks]
        signal_fft_peaks = [np.abs(signal_fft[i]) for i in time_indexes_peaks]
        freq = time_fft_peaks[np.argmax(signal_fft_peaks)]
        return freq

    def get_freq(self) -> float:
        """ Функция возвращает частоту сигнала.
        Returns:
            float равное частоте сигнала.
        """

        return self._freq


class OscillatorAnalogy(object):

    def __init__(self, dataframe: pd.DataFrame, time: float=4.62, fi: int=10, fi0: int=0):
        self._index_time = int(time * SAMPLE_RATE)
        self._df = dataframe
        self._t, self._Rz_L, self._Rz_R, self._Rx = self._interesting_time(self._index_time)
        self._l = self._calc_l_pend()
        self._H = self._calc_h_pend()
        self._m = self._calc_m_pend(fi, fi0)
        self.list_results = [
            ' ',
            (' ', 'Параметры системы'),
            ('Значение', 'Размерность', 'Описание'),
            (f'{M_LIC_ST:.3f}', 'кг', 'Масса неподвижной жидкости'),
            (f'{M_LIC_MOVE:.3f}', 'кг', 'Масса подвижной жидкости'),
            (f'{M_SYST:.3f}', 'кг', 'Масса сосуда и жидкости'),
            ' ',
            (' ', 'Промежуточные результаты'),
            ('Значение', 'Размерность', 'Описание'),
            (f'{STATIC_LOAD:.3f}', 'Н', 'Статическая нагрузка на левую опору от сосуда с жидкостью'),
            (f'{STATIC_LOAD:.3f}', 'Н', 'Статическая нагрузка на правую опору от сосуда с жидкостью'),
            (f'{2 * pi * frequency:.3f}', 'рад/сек',
            'Циклическая частота колебаний свободной поверхности жидкости'),
            (f'{self._H[2]:.3f}', 'Н·м', 'Момент, образуемый из-за смещения жидкости'),
            ' ',
            (' ', f'В момент времени t = {time}'),
            ('Значение', 'Размерность', 'Описание'),
            (self._Rz_L, 'Н', 'Вертикальная реакция в левой опоре'),
            (self._Rz_R, 'Н', 'Вертикальная реакция в правой опоре'),
            (self._Rx, 'Н', 'Горизонтальная реакция'),
            ' ',
            (' ', 'МАЯТНИКОВАЯ АНАЛОГИЯ'),
            ('Значение', 'Размерность', 'Описание'),
            (f'{self._l[0]:.3f}', self._l[1], 'Длина подвеса математического маятника'),
            (f'{self._m[0]:.3f}', self._m[1], 'Масса маятника'),
            (f'{self._H[0]:.3f}', self._H[1], 'Высота подвеса математического маятника')
            ]
        self._my_writer(self.list_results)

    def _interesting_time(self, index_time: int) -> pd.Series:
        """ Функция возвращает реакции в выбранное время.
        Returns:
            pd.Series типа (['Время, сек', 'Rz_R, Н', 'Rz_L, Н', 'Rx, Н'])
        """
        return self._df.iloc[index_time]

    def _calc_l_pend(self) -> tuple[float, str]:
        return g / (2*pi*frequency)**2, 'м'

    def _calc_h_pend(self) -> tuple[float, str, float]:
        m_liq = abs(abs(self._Rz_L-self._Rz_R)*L - M_LIC_ST*H_LIC_ST*g - M_C*H_C*g)
        return m_liq/self._Rx, 'м', m_liq

    def _calc_m_pend(self, fi, fi0) -> tuple[float, str]:
        g_force = (self._Rx / sin(radians(fi))) * (
                    1/(VELOCITY_0**2/(g*self._l[0]) + 3*cos(radians(fi)) - 2*cos(radians(fi0))))
        return g_force/g, 'кг'

    def _my_writer(self, list_res: list, name_result_csv: str = 'result') -> None:
        """ Функция записывает результаты вычислений в новый csv файл.
            Стандартно он будет назван 'result'.
        Args:
            list_res: список, состоящий из кортежей, в каждом кортеже описание конкретной величины
            name_result_csv: выбор названия файла
        """
        with open(name_result_csv + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerows(list_res)
            pass


if __name__ == '__main__':
    df = set_dataframe(address())
    COMPUTATION_TIME, SAMPLE_RATE = set_signal_timing_parameters(df)
    find_freq = FrequencyCalculation(df)
    frequency = find_freq.get_freq()
    find_analogy = OscillatorAnalogy(df)
