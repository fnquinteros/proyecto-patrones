from my_utils import extract_features_img
import numpy as np
import os
import json

OPTIONS = {
    'lbp': {
        'red': True,
        'hdiv': 8,
        'vdiv': 8,
    },
    'hog': {
        'v_windows': 4,
        'h_windows': 4,
        'n_bins': 16
    },
    'gabor': True,
    'masked': True
}

# Extraemos características
train = []
for i in range(1, 167):
    n = str(i)
    if len(n) == 1:
        n = '00' + n
    elif len(n) == 2:
        n = '0' + n
    # print(f'... reading FaceMask166/FM000{n} ...')
    for j in range(6):
        features = extract_features_img(
            f'FaceMask166/FM000{n}_0{j+1}.jpg', OPTIONS)
        if j == 0:
            m = features.shape[0]
            data = np.zeros((6, m))
        data[j] = features
    train.append(data)
train = np.array(train)
print(train.shape)
t = train.tolist()

all_feats_dict = {0: t}

with open('all_features.json', 'w') as f:
    json.dump(all_feats_dict, f)
