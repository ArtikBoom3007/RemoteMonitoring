import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import random
import os

current_directory = os.getcwd()

file_path = os.path.join(current_directory, r'C:\Users\lelca\GIT_project\RemoteMonitoring\ecg_dataset.csv')

# Чтение файла
df = pd.read_csv(file_path)

a = literal_eval(df['Отведение 1'][0])

plt.plot(a)

fig, ax = plt.subplots(12, 1, figsize=(20, 25), sharex=True, sharey=False, constrained_layout=True)
for i in range(12):
    ax[i].plot(literal_eval(df[f'Отведение {i+1}'][1]))
fig.suptitle('VA', fontsize=25)

FibCase = random.choice(list(df[df['Ритм'] == 'VA'].index))
fig, ax = plt.subplots(12, 1, figsize=(20, 25), sharex=True, sharey=False, constrained_layout=True)
for i in range(12):
    ax[i].plot(literal_eval(df[f'Отведение {i+1}'][FibCase]))
fig.suptitle('VA', fontsize=25)

plt.show()
