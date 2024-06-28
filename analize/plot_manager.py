import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

metrics = ['accuracy', 'f1', 'roc_score', 'r2']
colors = ['orange', 'blue', 'green', 'red']
"""
'best', 'upper right', 'upper left', 'lower left', 'lower right',
'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
"""
def plot_metrics(models, width=16, height=6, bar_width=0.15, postion='lower right'):
    n_groups = len(models)
    n_metrics = len(metrics)

    data = {metric: [] for metric in metrics}
    for metric in metrics:
        for model in models:
            data[metric].append(model.loc[model.index[0], metric])

    index = np.arange(n_groups)
    positions = [index + i * bar_width for i in range(n_metrics)]

    fig, ax = plt.subplots(figsize=(width, height))

    # Create bars for each metric
    for pos, metric, color in zip(positions, metrics, colors):
        bars = ax.bar(pos, data[metric], bar_width, label=metric, color=color)

    # etykiety do wykresu
    ax.set_title('dokladnosc modeli')
    ax.set_xticks(index + bar_width * 1.5)
    ax.set_xticklabels([model.index[0] for model in models])
    ax.legend(loc=postion, bbox_to_anchor=(1, 0))
    # zakres wysokosci
    ax.set_ylim(0.5, 1.0)

    for bars in ax.containers:
        ax.bar_label(bars, padding=3, fmt='%.2f')

    plt.tight_layout()
    plt.show()


def calculate_average(results, name):
    combined_results = pd.concat(results)
    combined_results['time'] = combined_results['time'].astype(float)
    numeric_columns = combined_results.drop(columns=['time']).select_dtypes(include='number')
    avg_result = numeric_columns.mean().to_frame().T

    # sumowanie czasu
    total_time = combined_results['time'].sum()

    avg_result['time'] = total_time

    # nazwa indeksu
    avg_result.index = [name]

    return avg_result