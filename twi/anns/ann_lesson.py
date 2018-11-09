# _*_ coding:utf-8 _*_

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from twi.com.utility import plot_decision_regions
from sklearn.metrics import accuracy_score
from twi.anns.ann_bp_multi import ANN_Multi


def show_digit():
    data = load_digits()
    print(data['data'].shape)
    print(data['target'].shape)
    fig, ax = plt.subplots(2, 5)
    ax = ax.flatten()
    for i in range(10):
        ax[i].imshow(data['images'][i], cmap='Greys', interpolation='nearest')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()

def test_ann():
    data = load_digits()
    ohe = OneHotEncoder()
    ohe.fit(data['target_names'].reshape(-1,1))
    print(ohe.transform([[9]]).toarray())
    indice_all = range(data['target'].shape[0])
    indice_train, indice_test = train_test_split(indice_all, shuffle=True, test_size=0.3)
    X ,y = data['data'][indice_train], data['target'][indice_train]
    y = ohe.transform(y.reshape(-1, 1)).toarray()
    ann = ANN_Multi(topo_stru=[64, 64, 10], n_iter=1000, epsilon=0.1)
    print(X.shape, y.shape)
    ann.fit(X, y, ANN_Multi.active_simoid)
    image_test = data['images'][indice_test]
    X_test,  y_test = data['data'][indice_test], data['target'][indice_test]
    y_pred = ann.predict(X_test)
    s = accuracy_score(y_pred, y_test)
    print(s)
    l_err = []
    l_cmp = y_pred == y_test.reshape(-1, 1)
    for j in range(len(l_cmp)):
        if not l_cmp[j]:
            l_err.append(j)
    # plot
    fig, ax = plt.subplots(2, 5)
    ax = ax.flatten()
    for i in range(10):
        ax[i].imshow(image_test[l_err[i]], cmap='Greys', interpolation='nearest')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title('%s|%s' % (y_test[l_err[i]], y_pred[l_err[i]]))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_ann()