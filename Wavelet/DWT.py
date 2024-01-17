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


def dwt_transform(x, method):
    global time, dwtmatr, dfreqs
    result_dwt = []
    freq_res = []
    if len(x.shape) == 2:
        N = x.shape[1]
        for sig in x:
            t, dwt, f = transform(sig, method)
            result_dwt.append(dwt)
            freq_res.append(f)
    elif len(x.shape) == 1:
        N = x.shape[0]
        t, result_dwt, freq_res = transform(x, method)
    else:
        raise Exception("Wrong Dimensions")
    dwtmatr = np.array(result_dwt)
    dfreqs = np.array(freq_res)
    # freqs = np.array(freq_res) * fs
    return t, dwtmatr, dfreqs

def cwt_daubechies4(signal, scales):
    # Define Daubechies 4 wavelet filter coefficients
    db4 = np.array([0.48296, 0.8365, 0.22414, -0.12941])

    # Ensure the signal length is even for proper downsampling
    if len(signal) % 2 == 1:
        signal = signal[:-1]

    scales = np.array(scales, dtype=int)
    scales = np.array(list(set(scales)))
    scales = np.sort(scales)
    cwtmatr = np.zeros((len(scales), len(signal)))

    for i, scale in enumerate(scales):
        # Upsample the wavelet filter to match the scale
        upscaled_db4 = np.zeros(int(scale * len(db4)))
        upscaled_db4[::scale] = db4
        if len(upscaled_db4) > len(signal):
            scales = scales[:i]
            cwtmatr = cwtmatr[:i][:]
            break

        # Convolve the signal with the scaled wavelet filter
        conv_result = np.convolve(signal, upscaled_db4, mode='same')
        # Extract the central part (same length as the original signal)
        #conv_result = conv_result[len(upscaled_db4)//2:len(upscaled_db4)//2+len(signal)]
        
        # Store the CWT result
        cwtmatr[i, :] = np.abs(conv_result)
    fre = scales / len(signal) * fs
    return cwtmatr, fre

def cwt_db(x):
    nyquist_freq = 0.5 * fs

    #level_frequencies = [np.linspace(0, nyquist_freq, 2**(level+1)) for level in range(5)]  # Пример для 5 уровней

    if detrend:
        x = scipy.signal.detrend(x)
    if normalize:
        x = (x - np.mean(x)) / np.std(x)

    # coeffs = pywt.dwt(x, 'db4')
    # coeffs = np.array(coeffs)
    # print(coeffs.shape)
    # dwt_matr = np.vstack(coeffs[:-1])  # Игнорируем последний уровень (листовой коэффициент)
    # print(dwt_matr)
    N = len(x)
    dt = 1.0/fs
    time = np.arange(N) * dt
    nOctaves = int(np.log2(2*np.floor(N/2.0)))
    scales = 2**np.arange(1, nOctaves, 1.0/nNotes)
    dwt_matr, freqs = pywt.cwt(x, scales, 'db4') #cwt_daubechies4(x, scales)

    return time, dwt_matr, freqs

def show_spectr_dwt(channel = -1):
    plt.subplot(2, 1, 2)

    if len(dwtmatr.shape) == 2:
        result = dwtmatr
        f_res = dfreqs
    else:
        result = dwtmatr[channel]
        f_res =  dfreqs[channel]
    plt.imshow(np.abs(result), extent=[0, 2 , f_res[-1], f_res[0]], aspect='auto', cmap='jet')
    #plt.yticks(np.arange(len(dwtmatr)), [f'{freq:.2f} Гц' for freq in dfreqs[::-1]])
    plt.colorbar(label='Амплитуда')
    plt.title('Спектрограмма CWT')
    plt.xlabel('Время (с)')
    plt.ylabel('Частота (Гц)')
    plt.show()
   