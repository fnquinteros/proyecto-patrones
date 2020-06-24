from pybalu.feature_selection import clean
from my_utils import get_data, sfs_features
import json
import numpy as np

if __name__ == "__main__":
    # recuperamos las features guardadas
    with open('all_features.json') as f:
        feats_d = json.load(f)
        all_features = np.array(feats_d['0'])

    A = all_features[:16, :, :]
    B = all_features[:40, :, :]

    A_train_feats = A[:, :3, :]
    A_val_feats = A[:, 3, :]
    A_test_feats = A[:, 4:6, :]

    B_train_feats = B[:, :3, :]
    B_val_feats = B[:, 3, :]
    B_test_feats = B[:, 4:6, :]

    # limpiamos las features muy correlacionadas
    A_clean_idx = clean(A_train_feats.reshape(
        A_train_feats.shape[0] * A_train_feats.shape[1],
        A_train_feats.shape[-1]
    ))

    print(A_clean_idx.shape)

    A_train_feats = A[:, :3, A_clean_idx]
    A_test_feats = A[:, 4:6, A_clean_idx]
    A_val_feats = A[:, 3, A_clean_idx].reshape(
        16, 1, A_clean_idx.shape[0]
    )

    # seleccionamos las mejores 50 features
    X_train, y_train = get_data(A_train_feats, 16)
    X_val, y_val = get_data(A_val_feats, 16)
    X_test, y_test = get_data(A_test_feats, 16)
    print(y_train, y_val)

    # X_train, X_val, sfs_idx = sfs_features(
    #     X_train, X_val, y_train, n_features=50)
    # x_test = X_test[:, sfs_idx]

    with open('selected_features.json', 'w') as f:
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
