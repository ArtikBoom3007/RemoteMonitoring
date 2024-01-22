import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pywt
import neurokit2 as nk


def init(sampling_rate = 500, init_nNotes = 12, init_detrend = True, init_norm = True):
    global fs, nNotes, detrend, normalize
    fs = sampling_rate
    nNotes = init_nNotes
    detrend = init_detrend
    normalize = init_norm


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
        t, result_cwt, freq_res = transform(x, method)
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
    ECG_df["CWT"] = np.zeros(ECG_df.shape[0], dtype=object)
    for index, row in ECG_df.iterrows():
        rpeaks = find_peaks(ECG_df["data"][index])
     
        if n_term_finish > len(rpeaks['ECG_R_Peaks']) - 1:
            print(f"Warning! There is no {n_term_finish} cycle in signal. It will be scipped")
            ECG_df.loc[index, 'CWT'] = 0
            continue
        
        start_pos = rpeaks['ECG_R_Peaks'][n_term_start]
        end_pos = rpeaks['ECG_R_Peaks'][n_term_finish]

        ECG_df["data"][index] = ECG_df["data"][index][:, start_pos:end_pos+1]
    
        t, cwt_m, _ = cwt_transform(ECG_df["data"][index], method)
        ECG_df.loc[index, 'CWT'] = cwt_m
    return ECG_df


def cwt_transform_df(ECG_df, n_term_start, n_term_finish, method):
    ECG_df["CWT"] = np.zeros(ECG_df.shape[0], dtype=object)
    
    for index, row in ECG_df.iterrows():
        lead1_data = row["data"][0]  # I
        lead2_data = row["data"][6]  # V1

        rpeaks_lead1 = find_peaks(lead1_data)
        rpeaks_lead2 = find_peaks(lead2_data)
        
        # Проверка для lead1
        if n_term_finish > len(rpeaks_lead1) - 1:
            print(f"Warning! There is no {n_term_finish} cycle in lead1 signal. It will be skipped")
            ECG_df.loc[index, 'CWT'] = 0
            continue

        start_pos_lead1 = rpeaks_lead1[n_term_start]
        end_pos_lead1 = rpeaks_lead1[n_term_finish]
        lead1_data = lead1_data[:, start_pos_lead1:end_pos_lead1+1]

        # Проверка для lead2
        if n_term_finish > len(rpeaks_lead2) - 1:
            print(f"Warning! There is no {n_term_finish} cycle in lead2 signal. It will be skipped")
            ECG_df.loc[index, 'CWT'] = 0
            continue

        start_pos_lead2 = rpeaks_lead2[n_term_start]
        end_pos_lead2 = rpeaks_lead2[n_term_finish]
        lead2_data = lead2_data[:, start_pos_lead2:end_pos_lead2+1]

        # Применение CWT к lead1
        t_lead1, cwt_m_lead1, _ = cwt_transform(lead1_data, method)
        # Применение CWT к lead2
        t_lead2, cwt_m_lead2, _ = cwt_transform(lead2_data, method)

        # Сохранение результатов CWT
        ECG_df.loc[index, 'CWT'] = {"lead1": {"time": t_lead1, "cwt": cwt_m_lead1},
                                    "lead2": {"time": t_lead2, "cwt": cwt_m_lead2}}

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