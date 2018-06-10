
import copy
import numpy as np

class Data(object):
    ''' Standardized data format that stores basic metadata and allows selecting on rows '''

    def __init__(self, X, y = None, group = None, which_categorical = None):
        self.X = X.copy()
        self.y = None if y is None else np.copy(y)
        self.which_categorical = copy.deepcopy(which_categorical)
        self.group = None if group is None else np.copy(group)

    def select(self, idx):
        y = None if self.y is None else self.y[idx]
        group = None if self.group is None else self.group[idx]
        return Data(
            X = self.X.iloc[idx],
            y = y,
            group = group,
            which_categorical = self.which_categorical
        )