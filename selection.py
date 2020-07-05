from pybalu.feature_selection import clean
from my_utils import get_data, sfs_features
import json
import numpy as np


def selection(feature_dataset, file_name):
    train_features = feature_dataset[:, :3, :]
    val_features = feature_dataset[:, 3:4, :]
    test_features = feature_dataset[:, 4:6, :]

    # limpiamos las features muy correlacionadas
    clean_idx = clean(train_features.reshape(
        train_features.shape[0] * train_features.shape[1],
        train_features.shape[-1]
    ))

    train_features = train_features[:, :, clean_idx]
    test_features = test_features[:, :, clean_idx]
    val_features = val_features[:, :, clean_idx]

    X_train, y_train = get_data(train_features, feature_dataset.shape[0])
    X_val, y_val = get_data(val_features, feature_dataset.shape[0])
    X_test, y_test = get_data(test_features, feature_dataset.shape[0])
    
    # X_train, X_val, sfs_idx = sfs_features(
    #    X_train, X_val, y_train, n_features=50)
    # X_test = X_test[:, sfs_idx]

    with open(file_name, 'w') as f:
        d = {
            'test': {
                'X': X_test.tolist(),
                'y': y_test.tolist()
            },
            'train': {
                'X': X_train.tolist(),
                'y': y_train.tolist()
            },
            'val': {
                'X': X_val.tolist(),
                'y': y_val.tolist()
            }
        }
        json.dump(d, f)


if __name__ == "__main__":
    # recuperamos las features guardadas
    with open('all_features.json') as f:
        feats_d = json.load(f)
        all_features = np.array(feats_d['0'])

    A = all_features[:16, :, :]
    B = all_features[:40, :, :]
    C = all_features[:100, :, :]
    D = all_features[:166, :, :]

    selection(D, 'selected_features.json')