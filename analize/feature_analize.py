def analyze_word_frequency(dataset):
    word_columns = [col for col in dataset.columns if col.startswith('word_freq_')]

    word_frequency = {}

    for col in word_columns:
        # slowo
        word = col.split('_')[2]

        # liczba wystapien slowa
        total_occurrences = dataset[col].sum()

        # liczba wiadomosci w ilu to slowo wystepuje
        num_messages = (dataset[col] > 0).sum()

        # dodanie do slownika
        word_frequency[word] = {
            'precent_wystepowania': total_occurrences,
            'ilosc_wiadomosci': num_messages
        }

    return word_frequency