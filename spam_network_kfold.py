import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
# importowanie przygotowanych danych
from analize import dataset
import projekt.networks.network_templates.network_2_hidden_layers as nn2
import projekt.networks.network_templates.network_1_hidden_layers as nn1
from networks.network_manager import network_train
from projekt.analize.plot_manager import plot_metrics, calculate_average

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

seed = 42
input_size = X.shape[1]
num_epochs = 100
learning_rate = 0.01
"""
siec z dwiema warstwamy ukrytymi
Funkcje aktywacji:
* ReLU
* ReLU
* Sigmoid
"""
# set_seed(seed)
# network_model_1 = nn2.NeuralNetwork(input_size, 4, 4, 1,
#                                     nn.ReLU(), nn.ReLU(), nn.Sigmoid())
# set_seed(seed)
# network_model_2 = nn2.NeuralNetwork(input_size, 8, 8, 1,
#                                     nn.ReLU(), nn.ReLU(), nn.Sigmoid())
# set_seed(seed)
# network_model_3 = nn2.NeuralNetwork(input_size, 32, 64, 1,
#                                     nn.ReLU(), nn.ReLU(), nn.Sigmoid())

"""
siec z jedna warstwa ukrta
Funkcje aktywacji:
* ReLU
* Sigmoid
"""
set_seed(seed)
network_model_1 = nn1.NeuralNetwork(input_size, 4,  1,
                                    nn.ReLU(), nn.Sigmoid())
set_seed(seed)
network_model_2 = nn1.NeuralNetwork(input_size, 8,  1,
                                    nn.ReLU(), nn.Sigmoid())
set_seed(seed)
network_model_3 = nn1.NeuralNetwork(input_size, 32,  1,
                                    nn.ReLU(), nn.Sigmoid())


# funkcja straty - jak bardzo model myli sie w przewidywaniach
criterion = nn.BCELoss()
#criterion = nn.MSELoss()
# algorytm optymalizacji - aktualizuje wagi modelu
# poprawa modelu
optimizer_1 = optim.Adam(network_model_1.parameters(), lr=learning_rate)
optimizer_2 = optim.Adam(network_model_2.parameters(), lr=learning_rate)
optimizer_3 = optim.Adam(network_model_3.parameters(), lr=learning_rate)
# optimizer_1 = optim.SGD(network_model_1.parameters(), lr=learning_rate)
# optimizer_2 = optim.SGD(network_model_2.parameters(), lr=learning_rate)
# optimizer_3 = optim.SGD(network_model_3.parameters(), lr=learning_rate)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# wszystkie wyniki z kazdego folda
results_1 = []
results_2 = []
results_3 = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # skalowanie danych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # for name, param in network_model_1.named_parameters():
    #     if param.requires_grad:
    #         print(f'Layer: {name} | Weights: {param.data} | Gradients: {param.grad}')

    network_1 = network_train('siec 1', network_model_1, X_train_scaled, y_train, X_test_scaled, y_test,
                              num_epochs, optimizer_1, criterion)
    results_1.append(network_1)

    for name, param in network_model_2.named_parameters():
        if param.requires_grad:
            print(f'Layer: {name} | Weights: {param.data} | Gradients: {param.grad}')
    network_2 = network_train('siec 2', network_model_2, X_train_scaled, y_train, X_test_scaled, y_test,
                              num_epochs, optimizer_2, criterion)
    results_2.append(network_2)

    break

    network_3 = network_train('siec 3', network_model_3, X_train_scaled, y_train, X_test_scaled, y_test,
                              num_epochs, optimizer_3, criterion)
    results_3.append(network_3)


# ze wszystkich wynikow KFold robie srednia
avg_results_1 = calculate_average(results_1, 'siec 1')
time_1 = avg_results_1['time']
print(f"czas trwania: {time_1} sekund")

avg_results_2 = calculate_average(results_2, 'siec 2')
time_2 = avg_results_2['time']
print(f"czas trwania: {time_2} sekund")

avg_results_3 = calculate_average(results_3, 'siec 3')
time_3 = avg_results_3['time']
print(f"czas trwania: {time_3} sekund")

# wykres
plot_metrics([avg_results_1, avg_results_2, avg_results_3], width=8, height=6, postion='lower left')