import h5py
import numpy as np
from scipy.spatial.distance import pdist as scipy_pdist
from sklearn.preprocessing import normalize

def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]


if __name__ == '__main__':
    with h5py.File("./data/nytimes-256-angular.hdf5", 'r') as f:
        distance = f.attrs['distance']
        X_train = np.array(f['train'])
        X_test = np.array(f['test'])

        X0 = X_test[0]

        print(X0)
        print(X_train[105658])
        print(X_train[137002])

        # v, X_train[idx]
        dis = pdist(X0, X_train[105658], "cosine")
        print(dis)
        dis = pdist(X0, X_train[137002], "cosine")
        print(dis)
        dis = pdist(normalize([X0])[0], normalize([X_train[137002]])[0], "cosine")
        print(dis)
