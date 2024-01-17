import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pywt


def init(sampling_rate = 500, init_nNotes = 12, init_detrend = True, init_norm = True):
    global fs, nNotes, detrend, normalize
    fs = sampling_rate
    nNotes = init_nNotes
    detrend = init_detrend
    normalize = init_norm


def wavelet():
    # print('mexh')
    # print('shan1.5-1.0')
    # print('cgau3')
    # print('fbsp1-1.5-1.0')
    # print('cmor1.5-1.0')
    return ['cmor1.5-1.0', 'mexh', 'shan1.5-1.0', 'cgau3', 'fbsp1-1.5-1.0']

def transform(x, method = 'morlet'):
    x = detrend_normalize(x)
    time, scales = cwt_scales(x)

    cwtmatr, freqs = cwt_global(x, scales, method)

    return time, cwtmatr, freqs


def cwt_transform(x, method = 'morlet'):
    global time, cwtmatr, freqs
    result_cwt = []
    freq_res = []
    if len(x.shape) == 2:
        N = x.shape[1]
        for sig in x:
            t, cwt, f = transform(sig, method)
            result_cwt.append(cwt)
            freq_res.append(f)
    elif len(x.shape) == 1:
        N = x.shape[0]
        t, result_cwt, freq_res = transform(x, method)
    else:
        raise Exception("Wrong Dimensions")
    
    cwtmatr = np.array(result_cwt)
    freqs = np.array(freq_res) * fs
    return t, cwtmatr, freqs


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


def cwt_global(x, scales, method):
    cwtmatr, freqs = pywt.cwt(x, scales, method)
    return cwtmatr, freqs


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
