import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

import scipy.signal
import pywt
import neurokit2 as nk


def init(sampling_rate = 500, init_nNotes = 12, init_detrend = True, init_norm = True, len_sig = 1000):
    global fs, nNotes, detrend, normalize, length
    fs = sampling_rate
    nNotes = init_nNotes
    detrend = init_detrend
    normalize = init_norm
    length = len_sig


def wavelet():
    """
    'cmor1.5-1.0', 'mexh', 'shan1.5-1.0', 'cgau3', 'fbsp1-1.5-1.0'
    """
    # print('mexh')
    # print('shan1.5-1.0')
    # print('cgau3')
    # print('fbsp1-1.5-1.0')
    # print('cmor1.5-1.0')
    return ['cmor1.5-1.0', 'mexh', 'shan1.5-1.0', 'cgau3', 'fbsp1-1.5-1.0']

def transform(x, method = wavelet()[0]):
    x = detrend_normalize(x)
    time, scales = cwt_scales(x)

    cwtmatr, freqs = pywt.cwt(x, scales, method)

    return time, cwtmatr, freqs


def cwt_transform(x, method = wavelet()[0]):
    global time, cwtmatr, freqs
    result_cwt = []
    freq_res = []
    if len(x.shape) == 2:
        N = x.shape[1]
        for sig in x:
            time, cwt, f = transform(sig, method)
            result_cwt.append(cwt)
            freq_res.append(f)
    elif len(x.shape) == 1:
        N = x.shape[0]
        time, result_cwt, freq_res = transform(x, method)
    else:
        raise Exception("Wrong Dimensions")
    
    cwtmatr = np.array(result_cwt)
    freqs = np.array(freq_res) * fs
    return time, cwtmatr, freqs


def detrend_normalize(x):
    if detrend:
        x = scipy.signal.detrend(x)
    if normalize:
        x = (x - np.mean(x)) / np.std(x)
    return x


def cwt_scales(x):
    #scales = pywt.scale2frequency('cmor1.5-1.0', np.arange(1, 100+ 1)) / (1/fs)
    N = len(x)
    dt = 1.0/fs
    time = np.arange(N) * dt
    nOctaves = int(np.log2(2*np.floor(N/2.0)))
    scales = 2**np.arange(1, nOctaves, 1.0/nNotes)
    return time, scales 


def cwt_transform_df(ECG_df, n_term_start, n_term_finish, method):
    # Создаем массив нулей для хранения результатов
    cwt_results = np.zeros((ECG_df.shape[0],), dtype=object)

    for index, row in ECG_df.iterrows():
        rpeaks = find_peaks(row["data"])

        if n_term_finish > len(rpeaks['ECG_R_Peaks']) - 1:
            print(f"Warning! There is no {n_term_finish} cycle in signal. \
                 Signal has only {len(rpeaks['ECG_R_Peaks'])} peaks")
            continue

        start_pos = rpeaks['ECG_R_Peaks'][n_term_start]
        end_pos = rpeaks['ECG_R_Peaks'][n_term_finish]

        data_cut = row["data"][:, start_pos:end_pos + 1]
        data_cut = data_cut[(1, 6), :]

        # Ресемплирование
        resampled = np.array([
            scipy.signal.resample(data_cut[0], length),
            scipy.signal.resample(data_cut[1], length)
        ])

        # Применение CWT
        _, cwt_m, _ = cwt_transform(resampled, method)
        cwt_results[index] = cwt_m

    ECG_df["CWT"] = cwt_results
    return ECG_df


def show_spectr(channel = -1):
    plt.subplot(2, 1, 2)

    if len(cwtmatr.shape) == 2:
        result = cwtmatr
        f_res = freqs
    else:
        result = cwtmatr[channel]
        f_res =  freqs[channel]
    plt.imshow(np.abs(result), extent=[0, 2 , f_res[-1], f_res[0]], aspect='auto', cmap='jet')
    plt.colorbar(label='Амплитуда')
    plt.title('Спектрограмма CWT')
    plt.xlabel('Время (с)')
    plt.ylabel('Частота (Гц)')
    plt.show()

def find_peaks(ecg):
    ## Поиск точек PQRST:
    signal = np.array(ecg[0])  

    # способ чистить сигнал перед поиском пиков:
    signal = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit") 

    # Поиск R зубцов:
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)

        # Проверка в случае отсутствия результатов и повторная попытка:
    if rpeaks['ECG_R_Peaks'].size < 3:
        signal = np.array(ecg[1])  
        signal = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit") 
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
        # При повторной проблеме выход из функции:
        if rpeaks['ECG_R_Peaks'].size < 2:
            print('в Сигнале ЭКГ слишком мало R зубцов')
            # Отобразим эти шумные сигналы:
            raise Exception("НЕ могу определить RR")
    return rpeaks