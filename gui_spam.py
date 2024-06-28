import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import Counter
import re
from email import policy
from email.parser import BytesParser

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import docs

from sklearn.model_selection import KFold
from torch import nn, optim

import projekt.networks.network_templates.network_1_hidden_layers as nn1
from projekt import analize
from projekt.analize.plot_manager import calculate_average, plot_metrics
from projekt.networks.network_manager import network_train

X = analize.dataset.iloc[:, :-1].values
y = analize.dataset.iloc[:, -1].values
input_size = X.shape[1]
num_epochs = 1000
learning_rate = 0.01

network_model = nn1.NeuralNetwork(input_size, 32,  1,
                                    nn.ReLU(), nn.Sigmoid())

criterion = nn.BCELoss()
optimizer_3 = optim.Adam(network_model.parameters(), lr=learning_rate)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    network_3 = network_train('siec 3', network_model, X_train, y_train, X_test, y_test,
                              num_epochs, optimizer_3, criterion)
    results.append(network_3)

# ze wszystkich wynikow KFold robie srednia
print(results)
avg_results_1 = calculate_average(results, 'siec')
time_1 = avg_results_1['time']
print(f"czas trwania: {time_1} sekund")

# wykres
plot_metrics([avg_results_1], width=8, height=6, postion='lower left')

word_frequencies = ['make', 'address', 'all', '3d', 'our', 'over', 'remove',
                    'internet', 'order', 'mail', 'receive', 'will', 'people',
                    'report', 'addresses', 'free', 'business', 'email', 'you',
                    'credit', 'your', 'font', '000', 'money', 'hp', 'hpl',
                    'george', '650', 'lab', 'labs', 'telnet', '857', 'data',
                    '415', '85', 'technology', '1999', 'parts', 'pm', 'direct',
                    'cs', 'meeting', 'original', 'project', 're', 'edu', 'table']

char_frequencies = [';', '(', '[', '!', '$', '#']
def analyze_email(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # naglowek
    headers = dict(msg.items())
    header_text = "\n".join(f"{key}: {value}" for key, value in headers.items())

    # mail - wiadomosc
    body_parts = []
    for part in msg.walk():
        if part.get_content_type() in ["text/plain", "text/html"]:
            body_parts.append(part.get_payload(decode=True).decode(errors='ignore'))

    body = "\n".join(body_parts)
    full_text = header_text + "\n" + body
    print(full_text)

    # slowa
    words = re.findall(r'\b\w+\b', full_text.lower())
    total_words = len(words)
    word_count = Counter(words)
    word_freq_results = {f"word_freq_{word}": (word_count[word] / total_words) * 100 for word in word_frequencies}

    # zliczanie znakow specjalnych
    total_chars = len(full_text)
    char_count = Counter(full_text)
    char_freq_results = {f"char_freq_{char}": (char_count[char] / total_chars) * 100 for char in char_frequencies}

    # wielkie litery - zliczanie
    capital_sequences = re.findall(r'[A-Z]+', full_text)
    capital_length_avg = sum(len(seq) for seq in capital_sequences) / len(capital_sequences) if capital_sequences else 0
    capital_length_longest = max((len(seq) for seq in capital_sequences), default=0)
    capital_length_total = sum(len(seq) for seq in capital_sequences)

    # data
    data_dict = {**word_freq_results, **char_freq_results,
                 'capital_run_length_average': capital_length_avg,
                 'capital_run_length_longest': capital_length_longest,
                 'capital_run_length_total': capital_length_total}

    return data_dict

network_model.eval()
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[(".eml", "*.eml")])
    if file_path:
        try:
            data = analyze_email(file_path)
            dataset = pd.DataFrame([data])

            #scaler = StandardScaler()
            #X_own = scaler.fit_transform(dataset.values)
            X_own_tensor = torch.FloatTensor(dataset.values)

            # y_pred
            with torch.no_grad():
                prediction = network_model(X_own_tensor)
                predicted_class = (prediction > 0.5).float().item()  # Zakładając próg 0.5 dla klasyfikacji binarnej

            # wynik
            is_spam = "SPAM" if predicted_class == 1.0 else "NOT SPAM"

            # czyszczenie tabeli
            for row in tree.get_children():
                tree.delete(row)

            # dodanie do tabeli
            for i, (col_name, value) in enumerate(dataset.iloc[0].items()):
                tree.insert("", "end", values=(col_name, value))

            # etykiety
            spam_label.config(text=f"Wynik klasyfikacji: {is_spam}")
        except Exception as e:
            messagebox.showerror("Error", f"Nie udało się załadować pliku: {e}")

# gui
root = tk.Tk()
root.title("Analiza spamu email")

frame = tk.Frame(root)
frame.pack(pady=10)

btn_select = tk.Button(frame, text="Wybierz plik .EML", command=select_file)
btn_select.pack()


tree_frame = tk.Frame(root)
tree_frame.pack(pady=10)

# kolumny
columns = ["Cechy", "Wartości"]
tree = ttk.Treeview(tree_frame, columns=columns, show="headings")

# naglowek
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=300, anchor="center")

tree.pack()

# etykieta finalna
spam_label = tk.Label(root, text="", font=("Arial", 16), fg="red")
spam_label.pack(pady=20)

root.mainloop()
