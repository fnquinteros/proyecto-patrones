import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pybalu.feature_extraction import (
    lbp_features, gupta_features, hugeo_features, flusser_features, haralick_features, gabor_features)
from pybalu.feature_selection import clean, exsearch, sfs
from pybalu.img_processing import segbalu
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pybalu.feature_transformation import pca


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
        img = img[:img.shape[0] // 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
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
            if lbp.get('ycrcb'):
                XYcrcb = lbp_features(ycrcb[:, :, 0], hdiv=hdiv, vdiv=vdiv, mapping='nri_uniform')
                feats.append(XYcrcb)
        if options.get('hog'):
            hog_opt = options.get('hog')
            v_windows = hog_opt.get('v_windows')
            h_windows = hog_opt.get('h_windows')
            n_bins = hog_opt.get('n_bins')
            w = 125 // h_windows
            h = 62 // v_windows
            Hog = hog(img, orientations=n_bins, pixels_per_cell=(h, w),
                cells_per_block=(1, 1), block_norm='L2-Hys', multichannel=True)
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
        if options.get('haralick'):
            Haralick = haralick_features(gray)
            feats.append(Haralick)
        if options.get('gabor'):
            Gabor = gabor_features(gray, rotations=4, dilations=4)
            feats.append(Gabor)
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


def filter_feats(features, selected):
    col_idx = list(selected)
    return features[:, :, col_idx]


def sfs_features(X_train, X_val, X_test, y_train, n_features):
    sfs_idx = sfs(X_train, y_train, n_features=n_features, show=True)
    X_train_sfs = X_train[:, sfs_idx]
    X_val_sfs = X_val[:, sfs_idx]
    X_test_sfs = X_test[:, sfs_idx]
    return X_train_sfs, X_val_sfs, X_test_sfs


def dirfiles(img_path, img_ext):
    img_names = fnmatch.filter(sorted(os.listdir(img_path)), img_ext)
    return img_names

def norm_features(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_val_norm, X_test_norm

def pca_features(X_train, X_val, X_test, n):
    X_train_pca, _, A, Xm, _ = pca(X_train, n_components=n)
    X_val_pca = np.matmul(X_val - Xm, A)
    X_test_pca = np.matmul(X_test - Xm, A)
    
    print(X_train_pca.shape)
    print(X_val_pca.shape)
    print(X_test_pca.shape)
    
    return X_train_pca, X_val_pca, X_test_pca

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
