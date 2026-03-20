import numpy as np


class AbsoluteScore:
    """ |y - y_pred| """

    def __call__(self, y, y_pred):
        return np.abs(y - y_pred)
