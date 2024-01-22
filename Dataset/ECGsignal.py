import pandas as pd
import numpy as np
import scipy.signal
from scipy import ndimage
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.transform import resize

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

current_directory = os.getcwd()

current_directory = os.path.join(current_directory, '../Dataset')

X,Y = 0, 0

def init():
    global X,Y
    """
    Загружает данные с двух файлов.

    Parameters:
        path1 (str): Путь к npy файлу с записями ЭКГ.
        path2 (str): Путь к csv файлу с данными для датасета.

    Returns:
        tuple: Тупль, содержащий X и Y.
    """
    X = np.load(os.path.join(current_directory, 'ecgeq-500hzsrfava.npy'))
    Y = pd.read_csv(os.path.join(current_directory, 'coorteeqsrafva.csv'), sep=';')


def plot_signal (selected_value):
    """
    Вывести графики для выбранного значения в столбце "Unnamed: 0"

    Parameters:
    selected_value (int): Значение в столбце "Unnamed: 0"

    Returns:
    None
    """
    # Замените 'YourColumnName' на фактическое имя столбца
    column_name = 'Unnamed: 0'

    # Замените 'YourValue' на значение, которое вы хотите использовать
    # selected_value = 0  # Замените на фактическое значение

    # Находим индекс, соответствующий выбранному значению в столбце "Unnamed: 0"
    NumberedCase = Y[Y[column_name] == selected_value].index.item()

    # Вывод данных и построение графика
    fig, ax = plt.subplots(12, 1, figsize=(20, 25), sharex=True, sharey=False, constrained_layout=True)
    for i in range(12):
        ax[i].plot(X[NumberedCase, :2500, i])
    fig.suptitle(Y.loc[NumberedCase, 'ritmi'], fontsize=25)

    # Покажем график
    plt.show()
    pass


def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact, sy//fact)
    return res


def detrend_normalize(x):
    detrend = True
    normalize = True
    if detrend:
        x = scipy.signal.detrend(x)
    if normalize:
        x = (x - np.mean(x)) / np.std(x)
    return x

def make_image(ecg, plot = False):
    """
    Генерирует изображение смещенных графиков

    - ecg: массив размером (12, 2500) или меньше c данными
    - plot (bool, optional): Флаг для отображения графиков. Если True, графики будут отображены.
        По умолчанию False.
    
    """
     # Вывод данных и построение графика
    fig, ax = plt.subplots(figsize=(40, 50), constrained_layout=True)
    plt.axis('off')
    
    # Находим максимальное значение амплитуды из всех 12 отведений
    max_amplitude = np.max(np.abs(ecg))

    for i in range(12):
        y_offset = i * max_amplitude * 1.5  # Смещение вдоль оси y (больше максимальной амплитуды)
        ax.plot(ecg[i] + y_offset, label=f'Lead {i + 1}')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image_data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image_width, image_height = canvas.get_width_height()

    reduction_factor = 0.5
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
  
def extract_images(patient_num = -1, number_labels = {"SR" : 1, "VA" : 0, "AF" : 0}, plot = False):
    """
    Генерирует DataFrame с сигналами ЭКГ, создает изображения и выводит графики смещенных сигналов.

    Parameters:
    - patient_num (int, optional): Номер пациента. Если не указан, функция создает DataFrame для нескольких пациентов.
    - number_labels (dict, optional): Словарь, определяющий количество случайных записей для каждого класса.
        Ключи словаря - метки классов, значения - количество записей.
        По умолчанию {"SR": 1, "VA": 0, "AF": 0}.
    - plot (bool, optional): Флаг для отображения графиков. Если True, графики будут отображены.
        По умолчанию False.

    Returns:
    - df (pd.DataFrame): DataFrame, содержащий сигналы ЭКГ, соответствующие метки классов и изображения смещенных сигналов.
    """

    df = pd.DataFrame(columns=['data', 'label'])
    if patient_num != -1:
        # Итерируемся по уникальным лейблам в Y
        for label in Y['ritmi'].unique():
            # Выбираем случайные записи с заданным лейблом
            if number_labels[label] == 0:
                continue
            samples = Y[Y['ritmi'] == label].sample(number_labels[label])
            # Для каждой выбранной записи создаем массив (12, 2500)
            for index, row in samples.iterrows():
                arr = X[index, :2500]  # Ваш массив NumPy размером (12, 2500)
                arr = np.transpose(arr)
                arr = detrend_normalize(arr)
                arr = make_image(arr, plot)

                # Добавляем запись в результаты
                df = df.append({'data': arr, 'label': label}, ignore_index=True)
    else:
        arr = X[patient_num, :2500]
        arr = arr.T
        arr = make_image(arr, plot)
        label = Y["ritmi"][patient_num]
        df = df.append({'data' : arr, 'label' : label})
    # Выводим результаты
    return df


def signal(patient_num=-1, number_labels={"SR": 1, "VA": 0, "AF": 0}): 
    """ 
    Генерирует DataFrame с сигналами электрокардиограммы (ЭКГ) и соответствующими метками классов. 
 
    Параметры: 
    - patient_num (int, optional): Номер пациента. Если не указан, функция создает DataFrame для нескольких пациентов. 
    - number_labels (dict, optional): Словарь, определяющий количество случайных записей для каждого класса. 
        Ключи словаря - метки классов, значения - количество записей. 
        По умолчанию {"SR": 1, "VA": 0, "AF": 0}. 
 
    Возвращает: 
    - df (pd.DataFrame): DataFrame, содержащий сигналы ЭКГ и метки классов. 
    """ 
    dfs_to_concat = []  # Список для хранения DataFrame для каждой строки в df 
 
    if patient_num == -1: 
        for label in Y['ritmi'].unique(): 
            if number_labels[label] == 0: 
                continue 
            samples = Y[Y['ritmi'] == label].sample(number_labels[label]) 
            for index, row in samples.iterrows(): 
                arr = X[index, :2500] 
                arr = arr.T 
                df_entry = pd.DataFrame({'data': [arr], 'label': [label]}) 
                dfs_to_concat.append(df_entry) 
    else: 
        arr = X[patient_num, :2500] 
        arr = arr.T 
        label = Y["ritmi"][patient_num] 
        df_entry = pd.DataFrame({'data': [arr], 'label': [label]}) 
        dfs_to_concat.append(df_entry) 
 
    # Используем concat для объединения DataFrame 
    df = pd.concat(dfs_to_concat, ignore_index=True) 
 
    return df