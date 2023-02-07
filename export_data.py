import pandas as pd
import numpy as np

def export_data(path: str, name: str, dataset: pd.DataFrame, columns: list, predictions: list, label_probability: np.array, cleanReview: list) -> None:
    """ Export data """
    new_dataset = dataset[columns].copy()
    new_dataset['sentiment'] = predictions
    probability_labels = pd.DataFrame(label_probability, columns=['neg_prob', 'pos_prob'])
    new_dataset = pd.concat([new_dataset.reset_index(), probability_labels], axis=1)
    new_dataset['cleanReview'] = cleanReview
    file_name = f'{path}/{name}_results.csv'
    new_dataset.to_csv(file_name, index=False)
    print(f'{name}.csv successfully created at path {path}')