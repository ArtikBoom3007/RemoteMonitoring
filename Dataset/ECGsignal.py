import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg

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
  
  
def return_signal(selected_value, plot = False):
    """
    Вывести графики для выбранного значения в столбце "Unnamed: 0" на одном графике с смещением.

    Parameters:
    selected_value (int): Значение в столбце "Unnamed: 0"

    Returns:
    image_array (numpy.ndarray): NumPy array representing the image as uint8
    """


    # Замените 'YourColumnName' на фактическое имя столбца
    column_name = 'Unnamed: 0'

    # Замените 'YourValue' на значение, которое вы хотите использовать
    # selected_value = 0  # Замените на фактическое значение

    # Находим индекс, соответствующий выбранному значению в столбце "Unnamed: 0"
    NumberedCase = Y[Y[column_name] == selected_value].index.item()

    # Вывод данных и построение графика
    fig, ax = plt.subplots(figsize=(40, 50), constrained_layout=True)
    plt.axis('off')
    
    # Находим максимальное значение амплитуды из всех 12 отведений
    max_amplitude = np.max(np.abs(X[NumberedCase, :2500, :]))

    for i in range(12):
        y_offset = i * max_amplitude * 1.5  # Смещение вдоль оси y (больше максимальной амплитуды)
        ax.plot(X[NumberedCase, :2500, i] + y_offset, label=f'Lead {i + 1}')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image_data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image_width, image_height = canvas.get_width_height()

    image_array = image_data.reshape((image_height, image_width, 4))[:, :, :3]
   
    # Закрываем текущую фигуру, чтобы не отображать ее
    plt.close()

    if plot == True:
        # Отображаем изображение
        plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()

    return image_array


def signal(selected_value = 1, number_of_signals = 1, rhytm = False, type_of_rhytm = 'SR'):
    """
    Выводит ECG сигнал для выбранного значения в столбце "Unnamed: 0".

    Parameters:
    selected_value (str): Параметр для выбора значения в столбце "Unnamed: 0".

    Returns:
    list: ЭКГ сигнал для выбранного значения, содержащий 12 отведений.
    """
    # Замените 'YourColumnName' на фактическое имя столбца
    column_name = 'Unnamed: 0'

    if rhytm == True:
        column_name = 'ritmi'
        selected_value = type_of_rhytm
        if number_of_signals == 1:
            normalCase = random.choice(list(Y[Y['ritmi']==type_of_rhytm].index))
            ECG_signal_array = np.empty((12, 2500))
            for i in range(12):
                ECG_signal_array[i] = X[normalCase, :2500, i]  
            return ECG_signal_array
        else:
            ECG_signal_array = np.empty((number_of_signals, 12, 2500))
            for j in range(number_of_signals):
                normalCase = random.choice(list(Y[Y['ritmi']==type_of_rhytm].index))
                for i in range(12):
                    ECG_signal_array[j][i] = X[normalCase, :2500, i]
            return ECG_signal_array
   
    if number_of_signals == 1:
         # Находим индекс, соответствующий выбранному значению в столбце "Unnamed: 0"
        NumberedCase = Y[Y[column_name] == selected_value].index.item()
        # Инициализация массива перед циклом
        ECG_signal_array = np.empty((12, 2500))

        for i in range(12):
            # Добавление строки в виде подмассива в массив
            ECG_signal_array[i] = X[NumberedCase, :2500, i]    
    else:   
        ECG_signal_array = np.empty((number_of_signals, 12, 2500))
        # Вывод данных
        for j in range(number_of_signals):
            NumberedCase = Y[Y[column_name] == selected_value[j]].index.item()
            for i in range(12):
                # Добавление строки в виде подмассива в подмассив
                ECG_signal_array[j, i] = X[NumberedCase, :2500, i]

    # Возврат всего массива после цикла
    return ECG_signal_array