import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import plotly.express as px
import neurokit2 as nk
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time

#pd.options.mode.chained_assignment = None

def init(filtering=False, f = 0.7, canc_showing=True, plot3D=False, sampling_rate = 500):
    global cancel_showing, filt, f_sreza, plot_3D, fs
    """
    filtering - производить ли фильтрацию сигнала с ФВЧ
    f - частота среза
    cancel_showing - если в положении True, то графики не выводятся
    plot_3D - выводить ли 3д график
    """
    cancel_showing = canc_showing
    plot_3D = plot3D
    filt = filtering
    f_sreza = f
    fs = sampling_rate

def make_df(ecg):
    global df, channels
    df = pd.DataFrame()
    start = 0
    end = len(ecg[0])
    df['ECG I'] = ecg[0][start:end]
    df['ECG II'] = ecg[1][start:end]
    df['ECG V1'] = ecg[6][start:end]
    df['ECG V2'] = ecg[7][start:end]
    df['ECG V3'] = ecg[8][start:end]
    df['ECG V4'] = ecg[9][start:end]
    df['ECG V5'] = ecg[10][start:end]
    df['ECG V6'] = ecg[11][start:end]
    channels = df.columns


def find_peaks():
    ## Поиск точек PQRST:
    n_otvedenie = 'I'
    signal = np.array(df['ECG I'])  
    time_new = len(signal) / fs

    # способ чистить сигнал перед поиском пиков:
    signal = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit") 

    # Поиск R зубцов:
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)

        # Проверка в случае отсутствия результатов и повторная попытка:
    if rpeaks['ECG_R_Peaks'].size <= 5:
        print("На I отведении не удалось детектировать R зубцы")
        print("Проводим детектирование по II отведению:")
        n_otvedenie = 'II'
        signal = np.array(df['ECG II'])  
        signal = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit") 
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
        # При повторной проблеме выход из функции:
        if rpeaks['ECG_R_Peaks'].size <= 3:
            print('Сигналы ЭКГ слишком шумные для анализа')
            # Отобразим эти шумные сигналы:
            if not cancel_showing:
                num_channels = len(channels)
                fig, axs = plt.subplots(int(num_channels/2), 2, figsize=(11, 8), sharex=True)
                for i, graph in enumerate(channels):
                    row = i // 2
                    col = i % 2
                    sig = np.array(df[graph])
                    axs[row, col].plot(time_new, sig)
                    axs[row, col].set_title(graph)
                    axs[row, col].set_xlim([0, 6])
                    axs[row, col].set_title(graph)
                    axs[row, col].set_xlabel('Time (seconds)')
                plt.tight_layout()
                plt.show()
                plt.ioff()
                plt.show()
            raise Exception("НЕ могу определить RR")
            return
    return rpeaks

def filter():
    global df
    # ФВЧ фильтрация артефактов дыхания:
    if filt == True:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = scipy.signal.butter(1, f_sreza, 'hp', fs=fs, output='sos')
            avg = np.mean(sig)
            filtered = scipy.signal.sosfilt(sos, sig)
            filtered += avg
            df_new[graph] = pd.Series(filtered)
        df = df_new.copy()
            
        # ФНЧ фильтрация (по желанию можно включить):
    filt_low_pass = False
    if filt_low_pass:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = scipy.signal.butter(1, 100, 'lp', fs=fs, output='sos')
            avg = np.mean(sig)
            filtered = scipy.signal.sosfilt(sos, sig)
            filtered += avg
            df_new[graph] = pd.Series(filtered)
        df = df_new.copy()
    
def vecg(df_term):
    # Получает значения ВЭКГ из ЭКГ
    DI = df_term['ECG I']
    DII = df_term['ECG II']
    V1 = df_term['ECG V1']
    V2 = df_term['ECG V2']
    V3 = df_term['ECG V3']
    V4 = df_term['ECG V4']
    V5 = df_term['ECG V5']
    V6 = df_term['ECG V6']

    df_term['x'] = -(-0.172*V1-0.074*V2+0.122*V3+0.231*V4+0.239*V5+0.194*V6+0.156*DI-0.01*DII)
    df_term['y'] = (0.057*V1-0.019*V2-0.106*V3-0.022*V4+0.041*V5+0.048*V6-0.227*DI+0.887*DII)
    df_term['z'] = -(-0.229*V1-0.31*V2-0.246*V3-0.063*V4+0.055*V5+0.108*V6+0.022*DI+0.102*DII)

    df_term = df_term.loc[:, ['x', 'y', 'z']]
    return df_term

def show(df_term):
    if not cancel_showing:
        plt.figure(figsize=(29, 7), dpi=68)
        plt.subplot(1, 3, 1)
        plt.plot(df_term.y, df_term.z)
        plt.title('Фронтальная плоскость')
        plt.xlabel('Y')
        plt.ylabel('Z')

        plt.subplot(1, 3, 2)
        plt.gca().invert_xaxis()
        plt.plot(df_term.x, df_term.z)
        plt.title('Сагиттальная плоскость')
        plt.xlabel('X')
        plt.ylabel('Z')

        plt.subplot(1, 3, 3)
        plt.plot(df_term.y, df_term.x)
        plt.title('Аксиальная плоскость')  
        plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.show()

    # Интерактивное 3D отображение
    if plot_3D:
        fig = px.scatter_3d(df_term, x='x', y='y', z='z', size='size', size_max=10, opacity=1)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

def make_matrix(A, B):
    fig, ax = plt.subplots()
    ax.plot(A, B)
    ax.axis('off')

    # Получаем массив изображения из текущей фигуры
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image_data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image_width, image_height = canvas.get_width_height()

    # Преобразование массива изображения в массив NumPy uint8
    image_array = image_data.reshape((image_height, image_width, 4))[:, :, 0]
    # Закрываем текущую фигуру, чтобы не отображать ее
    plt.close()
    # Отображение матрицы
    if not cancel_showing:
        plt.imshow(image_array, cmap="gray")
        plt.show()
    return image_array


def make_vecg(ECG, n_term_start, n_term_finish):
    global df
    make_df(ECG)
    filter()
    rpeaks = find_peaks()
    # Расчет ВЭКГ
    start_pos = rpeaks['ECG_R_Peaks'][n_term_start]
    end_pos = rpeaks['ECG_R_Peaks'][n_term_finish]
    #plt.plot(df['ECG I'][start_pos:end_pos])

    df = df.iloc[start_pos:end_pos+1, :]
    df = vecg(df)
    df['size'] = end_pos - start_pos # задание размера для 3D визуализации
    show(df)
    X = df["x"]
    Y = df["y"]
    Z = df["z"]

    df = df.iloc[0:0]

    # Создание матрицы

    image_arrayXY = make_matrix(X, Y)
    image_arrayZX = make_matrix(Z, X)
    image_arrayYZ = make_matrix(Y, Z)
    
    return image_arrayXY, image_arrayYZ, image_arrayZX

def make_vecg_df(ECG_df, n_term_start, n_term_finish):
    ECG_df["XY"] = np.zeros(ECG_df.shape[0], dtype=object)
    ECG_df["YZ"] = ECG_df["XY"]
    ECG_df["ZX"] = ECG_df["XY"]
    for index, row in ECG_df.iterrows():

        XY, YZ, ZX = make_vecg(np.array(ECG_df["data"][index]), n_term_start, n_term_finish)
        # print(XY.shape)
        ECG_df.loc[index, 'XY'] = XY
        ECG_df.loc[index, "YZ"] = YZ
        ECG_df.loc[index, "ZX"] = ZX
    return ECG_df