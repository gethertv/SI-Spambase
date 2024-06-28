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
from analize import dataset
from analize.plot_manager import plot_metrics
from model_utils import train_model, cross_validate_model

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# czas rozpoczecia
start_time = time.time()

# help(accuracy_score)
# help(f1_score)
# help(roc_auc_score)
# help(r2_score)

"""
XGBoost - predykcja w formie drzew decyzyjnych
"""

xgboost_frame = cross_validate_model(XGBClassifier(), 'xgboost', X, y)

"""
* regresja logistyczna 
* glownie do klasyfikacji binarnej
* zaklada liniowa zaleznosc
"""
log_reg_frame = cross_validate_model(LogisticRegression(), 'log_regresja', X, y)

"""
* drzewa decyzyjne
* cykliczny podzial wektorow
"""
dtree_frame = cross_validate_model(DecisionTreeClassifier(), 'drzewo decyzyjne', X, y)

"""
* las losowy
* algorytm nieliniowy
* zbydowany z wielu drzew dezycyjnych
* klasa ktora ma najwiecej drzew wygrywa (glosowanie)
"""
rforest_frame = cross_validate_model(RandomForestClassifier(), 'las losowy', X, y)

"""
* svc - support vector classifier
* dobre dla danych nieliniowych
"""
svc_frame = cross_validate_model(SVC(), 'svc', X, y)

"""
* knn - k-nearest neighbors
* klasyfikacja na podstawie wiekszosci sasiadow
"""
knn_frame = cross_validate_model(KNeighborsClassifier(), 'knn', X, y)

"""
* gradient boosting
* agreguje wyniki wielu prostych modeli.
"""
gboost_frame = cross_validate_model(GradientBoostingClassifier(), 'gboost', X, y)

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

# for i, model in enumerate(models):
#     # pobieranie y_pred z frame
#     y_pred = model.loc[model.index[0], 'y_pred']
#     #y_pred = np.array(y_pred)
#
#     ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[i])
#     # ustawienie tytulu
#     axes[i].set_title(model.index[0])
#
# plt.tight_layout()
# plt.show()

# wyswietlanie DataFrame
# bez kolumny y_pred
print(final_frame.head())
temp_model = final_frame.drop(columns='y_pred')
print(temp_model)


# wykres slupkowy z porownaniem
plot_metrics(models)



