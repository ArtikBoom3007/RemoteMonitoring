import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


current_directory = os.getcwd()

file_path = os.path.join(current_directory, r'C:\Users\lelca\GIT_project\RemoteMonitoring\ecg_dataset.csv')
# X = np.load(r'C:\Users\lelca\GIT_project\RemoteMonitoring\ecgeq-500hzsrfava.npy')
# Y = pd.read_csv(r'C:\Users\lelca\GIT_project\RemoteMonitoring\coorteeqsrafva.csv',sep=';')

def load_func(path1, path2):
    """
    Загружает данные с двух файлов.

    Parameters:
        path1 (str): Путь к npy файлу с записями ЭКГ.
        path2 (str): Путь к csv файлу с данными для датасета.

    Returns:
        tuple: Тупль, содержащий X и Y.
    """
    X = np.load(os.path.join(current_directory, path1))
    Y = pd.read_csv(os.path.join(current_directory, path2),sep=';')
    return X, Y


# X, Y = load_func(r'C:\Users\lelca\GIT_project\RemoteMonitoring\ecgeq-500hzsrfava.npy', r'C:\Users\lelca\GIT_project\RemoteMonitoring\coorteeqsrafva.csv')

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
        print(X[NumberedCase, :2500, i])
        ax[i].plot(X[NumberedCase, :2500, i])
    fig.suptitle(Y.loc[NumberedCase, 'ritmi'], fontsize=25)

    # Покажем график
    plt.show()
    pass


# plot_signal(63)

def ECG_signal(selected_value):
    """
    Выводит ECG сигнал для выбранного значения в столбце "Unnamed: 0".

    Parameters:
    selected_value (str): Параметр для выбора значения в столбце "Unnamed: 0".

    Returns:
    list: ЭКГ сигнал для выбранного значения, содержащий 12 отведений.
    """
    # Замените 'YourColumnName' на фактическое имя столбца
    column_name = 'Unnamed: 0'

    # Находим индекс, соответствующий выбранному значению в столбце "Unnamed: 0"
    NumberedCase = Y[Y[column_name] == selected_value].index.item()

    # Инициализация массива перед циклом
    ECG_signal_array = []

    # Вывод данных
    for i in range(12):
        # Добавление строки в виде подмассива в массив
        ECG_signal_array.append(list(X[NumberedCase, :2500, i]))

    # Возврат всего массива после цикла
    return ECG_signal_array
    
# print(ECG_signal(0)[0])