import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

"""
----------[ DOKLADNOSC ]----------
TP - true positive
TN - true negative
accuracy = (TP + TN) / REKORDY
----------------------------------

----------[ KRZYWA ROC ]----------
FN - false negative
FP - false positive
TPR - true positive rate
FPR - false positive rate

TPR - TP / (TP + FN) -- os Y   # prezycja
# procent rzeczywistych pozytywow
# ktory został poprawnie zweryfikowany


FPR = FP / (FP + TN) -- os X 
# procent rzeczywistych negatywow
# ktory zostal blednie zweryfikowany jako pozytywny
----------------------------------

--------------[ F1 ]--------------
TP / (TP + FP) -- os Y   # prezycja
TP / (TP + FN) -- os Y   # czulosc

F1 = 2 * ((prezycja * czulosc) / (prezyzja + czulosc))
# srednia harmoniczna precyzji i czulosci

--> precyzja to stosunek prawdziwie pozytywnych
przypadkow do wszystkich przypadkow 
sklasyfikowanych jako pozytywne
--> czulosc to stosunek prawdziwie 
pozytywnych przypadkow do wszystkich
rzeczywistych pozytywnych przypadkow
----------------------------------

--------------[ R2 ]--------------

R2 = 1 - (   (suma(y - y_pred))^2  /   (suma(y - avg(y_pred))^2)   )
wskazuje czy model dobrze pasuje do obserwowanych danych 
R2 nie zawsze jest wyznacznikiem skutecznego modelu
R2 nie jest odpowiedni do uzycia w walidacji krzyzowej K-Fold dla zadan klasyfikacji,
poniewaz jest miara uzywana glownie w regresji
----------------------------------
"""

def train_model(model, name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = pd.DataFrame({
        name: [y_pred, accuracy, f1, roc_score, r2]
    }, index=['y_pred', 'accuracy', 'f1', 'roc_score', 'r2'])
    return results.transpose()


# def cross_validate_model(model, name, X, y, n_splits=3):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     # y_pred
#     y_pred = cross_val_predict(model, X, y, cv=kf)
#     # tablica z wszystkimi wartosciami
#     scores_accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
#     scores_f1 = cross_val_score(model, X, y, cv=kf, scoring='f1')
#     scores_roc_auc = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
#
#     results = pd.DataFrame({
#         name: [y_pred, np.mean(scores_accuracy), np.mean(scores_f1), np.mean(scores_roc_auc), 0]
#     }, index=['y_pred', 'accuracy', 'f1', 'roc_score', 'r2'])
#
#     return results.transpose()
def cross_validate_model(model, name, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # przeksztalcenie danych (skalowanie) z modelem
    pipeline = make_pipeline(StandardScaler(), model)

    y_pred = cross_val_predict(pipeline, X, y, cv=kf)
    scores_accuracy = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
    scores_f1 = cross_val_score(pipeline, X, y, cv=kf, scoring='f1')
    scores_roc_auc = cross_val_score(pipeline, X, y, cv=kf, scoring='roc_auc')
    scores_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')

    results = pd.DataFrame({
        name: [y_pred, np.mean(scores_accuracy), np.mean(scores_f1), np.mean(scores_roc_auc), np.mean(scores_r2)]
    }, index=['y_pred', 'accuracy', 'f1', 'roc_score', 'r2'])

    return results.transpose()



# def network_train(name, model, X_train, y_train, X_test, y_test, num_epochs, optimizer, criterion):
#     X_train_tensor = torch.FloatTensor(X_train)
#     y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
#     X_test_tensor = torch.FloatTensor(X_test)
#     y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
#
#     # trenowanie
#     for epoch in range(num_epochs):
#         model.train()
#         outputs = model(X_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (epoch + 1) % 10 == 0:
#             print(f'epoch [{epoch + 1}/{num_epochs}], loss: {loss.item():.4f}')
#
#     # testowanie
#     model.eval()
#     with torch.no_grad():
#         predictions = model(X_test_tensor)
#         predicted_classes = (predictions > 0.5).float()  # prog
#
#         # metryki
#         accuracy = accuracy_score(y_test, predicted_classes.numpy())
#         f1 = f1_score(y_test, predicted_classes.numpy())
#         roc = roc_auc_score(y_test, predictions.numpy())
#         r2 = r2_score(y_test, predictions.numpy())
#
#     return pd.DataFrame({
#         'accuracy': [accuracy],
#         'f1': [f1],
#         'roc_score': [roc],
#         'r2': [r2]
#     }, index=[name])
