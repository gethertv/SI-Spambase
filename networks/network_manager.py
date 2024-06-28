import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from torch import nn


def network_train(name, model, X_train, y_train, X_test, y_test, num_epochs, optimizer, criterion):
    # czas rozpoczecia
    start_time = time.time()


    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

    # trenowanie
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad() # zerowanie gradientu
        loss.backward() # obliczenie wagi
        optimizer.step() # aktualizacja wag

        if (epoch + 1) % 10 == 0:
            print(f'epoch [{epoch + 1}/{num_epochs}], loss: {loss.item():.4f}')

    # testowanie
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_classes = (predictions > 0.5).float()  # prog

        # metryki
        accuracy = accuracy_score(y_test_tensor, predicted_classes.numpy())
        f1 = f1_score(y_test_tensor, predicted_classes.numpy())
        roc = roc_auc_score(y_test_tensor, predictions.numpy())
        r2 = r2_score(y_test_tensor, predictions.numpy())

    # czas zakonczenia
    end_time = time.time()
    diff_time = end_time - start_time

    return pd.DataFrame({
        'accuracy': [accuracy],
        'f1': [f1],
        'roc_score': [roc],
        'r2': [r2],
        'time': f"{diff_time:.2f}",
    }, index=[name])