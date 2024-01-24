import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.transform import resize
from joblib import Parallel, delayed
import neurokit2 as nk


def init(dt=True, norm=True, comp_fact = 2, sampling_rate = 500, resample_len = 2500):
    global detrend, normalize, compress_factor, fs, sample_len
    detrend = dt
    normalize = norm
    compress_factor = comp_fact
    fs = sampling_rate
    sample_len = resample_len


def detrend_normalize(x):
    detrend = True
    normalize = True
    if detrend:
        x = scipy.signal.detrend(x)
    if normalize:
        x = (x - np.mean(x)) / np.std(x)
    return x

def find_peaks(sig, fs = 500):
    ## Поиск точек PQRST:
    signal = np.array(sig[0])  
    time_new = len(signal) / fs

    # способ чистить сигнал перед поиском пиков:
    signal = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit") 

    # Поиск R зубцов:
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)

        # Проверка в случае отсутствия результатов и повторная попытка:
    if rpeaks['ECG_R_Peaks'].size < 3:
        #print("На I отведении не удалось детектировать R зубцы")
        #print("Проводим детектирование по II отведению:")
        signal = np.array(sig[1])  
        signal = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit") 
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
        # При повторной проблеме выход из функции:
        if rpeaks['ECG_R_Peaks'].size < 2:
            print(rpeaks['ECG_R_Peaks'].size)
            print('Сигналы ЭКГ слишком шумные для анализа')
            # Отобразим эти шумные сигналы:
            raise Exception("НЕ могу определить RR")
    return rpeaks

def make_image(ecg, compress_factor, n_term_start, n_term_finish, sample_len, plot = False):
    """
    Генерирует изображение смещенных графиков

    - ecg: массив размером (12, 2500) или меньше c данными
    - plot (bool, optional): Флаг для отображения графиков. Если True, графики будут отображены.
        По умолчанию False.
    
    """
    resampled = np.array([
            scipy.signal.resample(ecg[0], sample_len),
            scipy.signal.resample(ecg[1], sample_len),
            scipy.signal.resample(ecg[2], sample_len),
            scipy.signal.resample(ecg[3], sample_len),
            scipy.signal.resample(ecg[4], sample_len),
            scipy.signal.resample(ecg[5], sample_len),
            scipy.signal.resample(ecg[6], sample_len),
            scipy.signal.resample(ecg[7], sample_len),
            scipy.signal.resample(ecg[8], sample_len),
            scipy.signal.resample(ecg[9], sample_len),
            scipy.signal.resample(ecg[10], sample_len),
            scipy.signal.resample(ecg[11], sample_len),
    ])
    rpeaks = find_peaks(resampled)
    start_pos = rpeaks['ECG_R_Peaks'][n_term_start]
    end_pos = rpeaks['ECG_R_Peaks'][n_term_finish]
    resampled = resampled[:, start_pos:end_pos+1]
     # Вывод данных и построение графика
    fig, ax = plt.subplots(figsize=(40, 50), constrained_layout=True)
    plt.axis('off')
    
    # Находим максимальное значение амплитуды из всех 12 отведений
    max_amplitude = np.max(np.abs(resampled))

    for i in range(12):
        y_offset = i * max_amplitude * 1.5  # Смещение вдоль оси y (больше максимальной амплитуды)
        ax.plot(resampled[i] + y_offset, label=f'Lead {i + 1}')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image_data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image_width, image_height = canvas.get_width_height()

    reduction_factor = 1 / compress_factor
    image_array = image_data.reshape((image_height, image_width, 4))[:, :, :3]
    # Уменьшаем размер изображения
    image_array = resize(image_array[:, :, :3], (int(image_height * reduction_factor), int(image_width * reduction_factor)))
    # Применяем бинаризацию
    image_array = (image_array.mean(axis=2) < 1).astype(np.uint8) * 255

    # image_array = image_data.reshape((image_height, image_width, 4))[:, :, :3]
    # image_array = ndimage.interpolation.zoom(image_array,.5)
    #  # Применяем бинаризацию
    # image_array = (image_array[:, :, :3].mean(axis=2) < 255).astype(np.uint8) * 255
    
    # Закрываем текущую фигуру, чтобы не отображать ее
    plt.close()
    

    if plot == True:
        # Отображаем изображение
        plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()

    return image_array


def generate_image(data, comp, n_s, n_f, sample_l):
    return make_image(data, comp, n_s, n_f, sample_l)

def images_parallel(ECG_df, n_term_start, n_term_finish, n_jobs=-1):
    ECG_df["images"] = Parallel(n_jobs=n_jobs)(delayed(generate_image)(ECG_df["data"][index], compress_factor, n_term_start, n_term_finish, sample_len) \
        for index, row in ECG_df.iterrows())
    ECG_df = ECG_df.loc[:, ["images", "label"]]
    return ECG_df

def images(ECG_df, n_term_start, n_term_finish):
    ECG_df["images"] = np.zeros(ECG_df.shape[0], dtype=object)
    for index, row in ECG_df.iterrows():
        img = make_image(ECG_df["data"][index], compress_factor, n_term_start, n_term_finish, sample_len)
        # print(XY.shape)
        ECG_df.loc[index, 'images'] = img
    ECG_df = ECG_df.loc[:, ["images", "label"]]
    return ECG_df
