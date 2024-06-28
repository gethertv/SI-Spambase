import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# importowanie przygotowanych danych
from analize import dataset, std_columns
from analize.plot_manager import plot_metrics
from model_utils import train_model, cross_validate_model
import seaborn as sns

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# czas rozpoczecia
start_time = time.time()

# dzielenie na czesc testowa i czesc do uczenia
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
# zbior treningowy
X_train_scaled = scaler.fit_transform(X_train)
# zbior testowy
X_test_scaled = scaler.transform(X_test)

# # srednia dla 54 kolumny (ta ktora bedzie skalowana)
# scaled_54_mean_all = dataset.iloc[:, 54].mean()
# scaled_54_mean_train = np.mean(X_train[:, 54])
# scaled_54_mean_test = np.mean(X_test[:, 54])
#
# print(f"średnia dla całego zbioru: {scaled_54_mean_all}")
# print(f"Średnia dla zbioru treningowego: {scaled_54_mean_train}")
# print(f"Średnia dla zbioru testowego: {scaled_54_mean_test}")
#
# # prawidlowe skalowanie
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # srednia po skalowaniu
# scaled_54_mean_train_scaled = np.mean(X_train_scaled[:, 54])
# scaled_54_mean_test_scaled = np.mean(X_test_scaled[:, 54])
#
# print(f"średnia po skalowaniu dla zbioru treningowego: {scaled_54_mean_train_scaled}")
# print(f"średnia po skalowaniu dla zbioru testowego: {scaled_54_mean_test_scaled}")
#
# # odchylenie standardowe
# std_54_all = dataset.iloc[:, 54].std()
# std_54_train = np.std(X_train[:, 54])
# std_54_test = np.std(X_test[:, 54])
# std_54_train_scaled = np.std(X_train_scaled[:, 54])
# std_54_test_scaled = np.std(X_test_scaled[:, 54])
#
# print(f"Odchylenie standardowe dla: ")
# print(f"* oryginalny zbiór: {std_54_all}")
# print(f"* zbiór treningowy: {std_54_train}")
# print(f"* zbiór testowy: {std_54_test}")
# print(f"* zbiór treningowy po skalowaniu: {std_54_train_scaled}")
# print(f"* zbiór testowy po skalowaniu: {std_54_test_scaled}")


# # # standaryzacja danych
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

"""
XGBoost - predykcja w formie drzew decyzyjnych
"""
xgboost_frame = train_model(XGBClassifier(), 'xgboost', X_train_scaled, y_train, X_test_scaled, y_test)

"""
* regresja logistyczna 
* glownie do klasyfikacji binarnej
* zaklada liniowa zaleznosc
"""
log_reg_frame = train_model(LogisticRegression(), 'log_regresja', X_train_scaled, y_train, X_test_scaled, y_test)

"""
* drzewa decyzyjne
* cykliczny podzial wektorow
"""
dtree_frame = train_model(DecisionTreeClassifier(), 'drzewo decyzyjne', X_train_scaled, y_train, X_test_scaled, y_test)

"""
* las losowy
* algorytm nieliniowy
* zbydowany z wielu drzew dezycyjnych
* klasa ktora ma najwiecej drzew wygrywa (glosowanie)
"""
rforest_frame = train_model(RandomForestClassifier(), 'las losowy', X_train_scaled, y_train, X_test_scaled, y_test)

"""
* svc - support vector classifier
* dobre dla danych nieliniowych
"""
svc_frame = train_model(SVC(), 'svc', X_train_scaled, y_train, X_test_scaled, y_test)

"""
* knn - k-nearest neighbors
* klasyfikacja na podstawie wiekszosci sasiadow
"""
knn_frame = train_model(KNeighborsClassifier(), 'knn', X_train_scaled, y_train, X_test_scaled, y_test)

"""
* gradient boosting
* agreguje wyniki wielu prostych modeli.
"""
gboost_frame = train_model(GradientBoostingClassifier(), 'gboost', X_train_scaled, y_train, X_test_scaled, y_test)

# czas zakonczenia
end_time = time.time()
diff_time = end_time - start_time
print(f"czas trwania: {diff_time:.2f} sekund")

# laczenie wszystkich modeli do jednego frama
models = [xgboost_frame, log_reg_frame, dtree_frame, rforest_frame, svc_frame, knn_frame, gboost_frame]
final_frame = pd.concat(models)
#print(final_frame)


# macierz pomylek
fig, axes = plt.subplots(1, len(models), figsize=(18, 6))

for i, model in enumerate(models):
    # pobieranie y_pred z frame
    y_pred = model.loc[model.index[0], 'y_pred']
    #y_pred = np.array(y_pred)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[i])
    # ustawienie tytulu
    axes[i].set_title(model.index[0])

plt.tight_layout()
plt.show()

# wyswietlanie DataFrame
# bez kolumny y_pred
temp_model = final_frame.drop(columns='y_pred')
print(temp_model)


# wykres slupkowy z porownaniem
plot_metrics(models)


