from my_utils import filter_feats, get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json
import numpy as np

# Recuperamos los datos de selección.

with open('selected_features.json') as f:
    d = json.load(f)
    train_data = d['train']
    test_data = d['test']
    val_data = d['val']

X, y = np.array(train_data['X']), np.array(train_data['y'])
Xt, yt = np.array(test_data['X']), np.array(test_data['y'])
Xv, yv = np.array(val_data['X']), np.array(val_data['y'])
# svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# svc.fit(X, y)


if __name__ == "__main__":

    def test(Xv, yv):
        # Probamos con 1 hasta 10 vecinos para ver qué resulta mejor.
        max_knn = 0
        best_k = -1
        for k in range(1, 30):
            knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
            knn.fit(X, y)
            score = knn.score(Xv, yv)
            if score >= max_knn:
                max_knn = score
                best_k = k
        print(f'best knn score: {max_knn} with {best_k} neighbors')

        best_rf = 0
        best_n = 0
        for n in range(1, 20):
            rforest = RandomForestClassifier(max_depth=n, n_estimators=100)
            rforest.fit(X, y)
            score = rforest.score(Xv, yv)
            if score >= best_rf:
                best_rf = score
                best_n = n
        print(
            f'best random forest score with 100 estimators: {best_rf} with {best_n} max_depth')

        best_est = 0
        for est in range(10, 100, 5):
            rforest = RandomForestClassifier(max_depth=best_n, n_estimators=est)
            rforest.fit(X, y)
            score = rforest.score(Xv, yv)
            if score >= best_rf:
                best_rf = score
                best_est = est
        print(f'best random forest score: {best_rf} with {est} estimators')

        svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        svc.fit(X, y)
        score = svc.score(Xv, yv)
        print(f'SVM score: {score}')

        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                multi_class='auto', max_iter=1000).fit(X, y)
        score = clf.score(Xv, yv)
        print(f'Logistic regression score: {score}')
    
    test(Xv, yv)
