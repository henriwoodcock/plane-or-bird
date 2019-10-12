from sklearn.utils import shuffle
def shuffle_data(x_train, y_train):
    x_train, y_train = shuffle(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    return x_train, y_train
