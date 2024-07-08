from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


def get_dataset(train_samples:int)->list:        
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    return train_test_split(
        X, y, train_size=train_samples, test_size=10000
    )