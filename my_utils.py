import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pybalu.feature_extraction import (
    lbp_features, hog_features, gupta_features, hugeo_features, flusser_features)
from pybalu.feature_selection import clean, exsearch, sfs
from pybalu.img_processing import segbalu
from sklearn.metrics import confusion_matrix, accuracy_score


def imshow(image):
    pil_image = Image.fromarray(image)
    pil_image.show()


def get_image(path, show=False):
    img = cv2.imread(path)
    if show:
        imshow(img)
    return img


def extract_features_img(st, options):
    img = get_image(st)
    if options.get('masked'):
        img = img[:img.shape[0]//2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b_img, _, _ = segbalu(img)

    def get_features(img, gray, b_img, options):
        feats = []
        lbp = options.get('lbp')
        if lbp:
            hdiv = lbp.get('hdiv')
            vdiv = lbp.get('vdiv')
            if lbp.get('bw'):
                XBW = lbp_features(b_img, hdiv=hdiv, vdiv=vdiv,
                                   mapping='nri_uniform')
                feats.append(XBW)
            if lbp.get('gray'):
                X_0 = lbp_features(gray, hdiv=hdiv, vdiv=vdiv,
                                   mapping='nri_uniform')
                feats.append(X_0)
            if lbp.get('red'):
                XRed = lbp_features(img[:, :, 0], hdiv=hdiv,
                                    vdiv=vdiv, mapping='nri_uniform')
                feats.append(XRed)
            if lbp.get('green'):
                XGreen = lbp_features(
                    img[:, :, 1], hdiv=hdiv, vdiv=vdiv, mapping='nri_uniform')
                feats.append(XGreen)
            if lbp.get('blue'):
                XBlue = lbp_features(img[:, :, 2], hdiv=hdiv,
                                     vdiv=vdiv, mapping='nri_uniform')
                feats.append(XBlue)
        if options.get('hog'):
            Hog = hog_features(gray, v_windows=4, h_windows=4, n_bins=16)
            feats.append(Hog)
        if options.get('gupta'):
            Gupta = gupta_features(b_img)
            feats.append(Gupta)
        if options.get('hugeo'):
            Hugeo = hugeo_features(b_img)
            feats.append(Hugeo)
        if options.get('flusser'):
            Flusser = flusser_features(b_img)
            feats.append(Flusser)
        return tuple(feats)
    feats = np.concatenate(get_features(img, gray, b_img, options))
    features = np.asarray(feats)
    return features


def extract_features(dirpath, fmt, options):
    st = '*.'+fmt
    img_names = dirfiles(dirpath+'/', st)
    n = len(img_names)
    print(n)
    for i in range(n):
        img_path = img_names[i]
        print('... reading '+img_path)
        features = extract_features_img(dirpath+'/'+img_path, options)
        if i == 0:
            m = features.shape[0]
            data = np.zeros((n, m))
            print('size of extracted features:')
            print(features.shape)
        data[i] = features
    return data


def fisher_score(feats):
    N = feats.shape[1]
    num_classes = feats.shape[0]
    num_features = feats.shape[2]
    images_per_class = N//num_classes
    N_k = N/num_classes
    p_k = 1/num_classes
    C_k_param = 1/(N_k - 1)
    z_ = np.zeros(num_features)
    for i in range(num_classes):
        for j in range(N//num_classes):
            z_ += feats[i][j]
    z_ = z_/N

    z_k_ = [np.zeros(num_features) for i in range(num_classes)]
    for i in range(num_classes):
        for j in range(images_per_class):
            z_k_[i] += feats[i][j]
        z_k_[i] /= images_per_class

    C_b = 0
    for i in range(len(z_k_)):
        vect = (z_k_[i] - z_)
        vect = vect.reshape(1, vect.shape[0])
        t_vect = vect.reshape(vect.shape[1], 1)
        C_b += p_k*(t_vect).dot(vect)
    C_w = 0
    for k in range(num_classes):
        C_k = 0
        for j in range(images_per_class):
            vect = (feats[k][j] - z_k_[k])
            vect = vect.reshape(1, vect.shape[0])
            t_vect = vect.reshape(vect.shape[1], 1)
            C_k += (t_vect).dot(vect)
        C_k *= C_k_param
        C_w += p_k * C_k
    try:
        return (np.linalg.inv(C_w)*C_b).trace()
    except np.linalg.LinAlgError:
        return -np.inf


def filter_feats(features, selected):
    col_idx = list(selected)
    return features[:, :, col_idx]


def sfs_features(X_train, X_test, y_train, n_features):
    sfs_idx = sfs(X_train, y_train, n_features=n_features)
    X_train_sfs = X_train[:, sfs_idx]
    X_test_sfs = X_test[:, sfs_idx]
    return X_train_sfs, X_test_sfs, sfs_idx


def dirfiles(img_path, img_ext):
    img_names = fnmatch.filter(sorted(os.listdir(img_path)), img_ext)
    return img_names

# Acomoda los datos de las features para pasarlos al clasificador


def get_data(train, num_classes):
    all_data_l = []
    labels = []
    for i in range(num_classes):
        for img_vector in train[i]:
            all_data_l.append(img_vector)
            labels.append(i)
    all_data = np.array(all_data_l)
    labels = np.array(labels)
    return all_data, labels
