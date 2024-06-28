import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
# importowanie przygotowanych danych
from analize import dataset
from analize.plot_manager import plot_metrics
import projekt.networks.network_templates.network_1_hidden_layers as nn1
import projekt.networks.network_templates.network_2_hidden_layers as nn2
from projekt.networks.network_manager import network_train

"""
ustawianie tego samego seeda
w celu wylosowania tych samych wag
początkowych
"""
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

seed = 42
input_size = X.shape[1]
num_epochs = 100
learning_rate = 0.01

# skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
siec z dwiema warstwamy ukrytymi
Funkcje aktywacji:
* Tanh
* Tanh
* Sigmoid
"""
# resetowanie seed'a (domyślne wagi początkowe)
set_seed(seed)
network_model_1 = nn2.NeuralNetwork(input_size, 4, 4, 1,
                                    nn.Tanh(), nn.Tanh(), nn.Sigmoid())
# resetowanie seed'a (domyślne wagi początkowe)
set_seed(seed)
network_model_2 = nn2.NeuralNetwork(input_size, 8, 8, 1,
                                    nn.Tanh(), nn.Tanh(), nn.Sigmoid())
# resetowanie seed'a (domyślne wagi początkowe)
set_seed(seed)
network_model_3 = nn2.NeuralNetwork(input_size, 32, 64, 1,
                                    nn.Tanh(), nn.Tanh(), nn.Sigmoid())


"""
siec z jedna warstwa ukrta
Funkcje aktywacji:
* Tanh
* Sigmoid
"""
# network_model_1 = nn1.NeuralNetwork(input_size, 4, 1,
#                                     nn.Tanh(), nn.Sigmoid())
# network_model_2 = nn1.NeuralNetwork(input_size, 8, 1,
#                                     nn.Tanh(), nn.Sigmoid())
# network_model_3 = nn1.NeuralNetwork(input_size, 32,1,
#                                     nn.Tanh(), nn.Sigmoid())


"""
siec z dwiema warstwamy ukrytymi
Funkcje aktywacji:
* ReLU
* ReLU
* Sigmoid
"""
# network_model_1 = nn2.NeuralNetwork(input_size, 4, 4, 1,
#                                     nn.ReLU(), nn.ReLU(), nn.Sigmoid())
# network_model_2 = nn2.NeuralNetwork(input_size, 8, 8, 1,
#                                     nn.ReLU(), nn.ReLU(), nn.Sigmoid())
# network_model_3 = nn2.NeuralNetwork(input_size, 32, 64, 1,
#                                     nn.ReLU(), nn.ReLU(), nn.Sigmoid())


"""
siec z jedna warstwa ukrta
Funkcje aktywacji:
* ReLU
* Sigmoid
"""
# network_model_1 = nn1.NeuralNetwork(input_size, 4, 1,
#                                     nn.ReLU(), nn.Sigmoid())
# network_model_2 = nn1.NeuralNetwork(input_size, 8, 1,
#                                     nn.ReLU(), nn.Sigmoid())
# network_model_3 = nn1.NeuralNetwork(input_size, 32,1,
#                                     nn.ReLU(), nn.Sigmoid())

"""
funkcja straty - jak bardzo model myli sie w przewidywaniach
"""
criterion = nn.BCELoss()

#[ 1 ]#######################################################################
"""
algorytm optymalizacji - aktualizuje wagi modelu
poprawa modelu
"""

optimizer_1 = optim.Adam(network_model_1.parameters(), lr=learning_rate)
network_1 = network_train('siec 1', network_model_1, X_train_scaled, y_train, X_test_scaled, y_test,
                          num_epochs, optimizer_1, criterion)
time_1 = network_1['time']
print(f"czas trwania: {time_1} sekund")

#[ 2 ]#######################################################################
optimizer_2 = optim.Adam(network_model_2.parameters(), lr=learning_rate)
network_2 = network_train('siec 2', network_model_2, X_train_scaled, y_train, X_test_scaled, y_test,
                          num_epochs, optimizer_2, criterion)

time_2 = network_2['time']
print(f"czas trwania: {time_2} sekund")

#[ 3 ]#######################################################################
optimizer_3 = optim.Adam(network_model_3.parameters(), lr=learning_rate)
network_3 = network_train('siec 3', network_model_3, X_train_scaled, y_train, X_test_scaled, y_test,
                          num_epochs, optimizer_3, criterion)

time_3 = network_3['time']
print(f"czas trwania: {time_3} sekund")

plot_metrics([network_1, network_2, network_3], width=8, height=6, postion='lower left')