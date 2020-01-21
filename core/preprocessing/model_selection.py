from sklearn import model_selection


def train_test_split(x, y, test_size=0.2):
    return model_selection.train_test_split(x, y, test_size=test_size, random_state=0)

