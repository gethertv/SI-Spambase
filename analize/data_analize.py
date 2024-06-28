import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from .feature_analize import analyze_word_frequency

columns = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
    "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
    "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
    "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
    "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
    "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
    "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
    "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average",
    "capital_run_length_longest", "capital_run_length_total", "is_spam"
]

# dane
dataset = pd.read_csv('data/spambase.data', names=columns)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# podglad danych
print(dataset.head())

# sprawdzenie czy nie ma brakujacych danych
missing_values = dataset.isnull().sum()
print("Brakujące dane:\n", missing_values.sum())

# opis dataset
print(dataset.describe(include='all'))

# wyswietlanie typow
print(dataset.dtypes)
# zliczenie cech o typie 'object'
number_object = (dataset.dtypes == 'object').sum()
print(f"Liczba kolumn z typem object: {number_object}")

if number_object > 0:
    # wyświetlenie kolumn które sa obiektami
    object_columns = dataset.dtypes[dataset.dtypes == 'object']
    print(object_columns.sum())

# ilosc SPAM/ NIE SPAM
print(f"0 - nie spam \n1 - spam\n {dataset.groupby(dataset.columns[-1])[dataset.columns[-1]].count()}")
# wykres z pogrupowaniem na wiadomosci SPAM/NIE SPAM
plt.figure(figsize=(8, 6))
sns.countplot(x='is_spam', data=dataset)
plt.title('SPAM / NIE SPAM')
plt.xlabel('0 - nie spam | 1 - spam')
plt.ylabel('liczba wiadomosci')
plt.show()

# histogram
# dla czterech cech
# x - czestotliwosc wystepowania 2.5%
# y - ilosc_wiadomosci
selected_columns = ['word_freq_free', 'word_freq_hp', 'char_freq_$', 'char_freq_(']
dataset[selected_columns].hist(bins=10, figsize=(12, 8), layout=(2, 2), color='blue', alpha=0.7)
plt.suptitle('histogramy wybranych cech')
plt.show()

word_frequency = analyze_word_frequency(dataset)
# konwertowanie na dataframe
word_frequency_df = pd.DataFrame(word_frequency).T

# sortowanie po liczbie wiadomosci
sorted_word_frequency_df = word_frequency_df.sort_values(by='ilosc_wiadomosci', ascending=False)

print("najczesciej wystepowane slowa (5)")
print(sorted_word_frequency_df.head(5))

# macierz korelacji
plt.figure(figsize=(14, 12))
sns.heatmap(dataset.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('macierz korelacji')
plt.show()

# sprawdzenie korelacji miedzy kolumna 'word_freq_hp' a 'word_freq_hpl'
correlation_hp_hpl = dataset['word_freq_415'].corr(dataset['word_freq_857'])
print(f"wynik korelacji: {correlation_hp_hpl:.2f}")

# macierz korelacji
corr_matrix = dataset.corr().abs()

# wybrania z macierzy gornego trojkata
# aby zapowiedz niepotrzebnemu liczeniu
# tych samych wartosci
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# skanowanie kolumn ktore maja wysoka korelacje
# prog korelacji kiedy dana kolumna moze zostac wyrzucona
threshold = 0.9
columns_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print(f"kolumna ktora moze zostac wyrzucona:\n {columns_to_drop}")
# usuniecie kolumny
#print(dataset.shape[1])
dataset.drop(columns_to_drop, axis=1, inplace=True)
#print(dataset.shape[1])

# odchylenie standardowe
std_dev = dataset.std()
threshold = 3
std_columns = std_dev[std_dev > threshold].index
print("kolumny z odchyleniem standardowym:\n", std_columns)

#scaler = StandardScaler()
#scaled_data = scaler.fit_transform(dataset)
#dataset[std_columns] = scaler.fit_transform(dataset[std_columns])
print(dataset.head())
