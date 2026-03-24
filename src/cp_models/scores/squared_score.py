import numpy as np


class SquaredScore:
    """ (y - y_pred)^2 """

    def __call__(self, y, y_pred):
        return (y - y_pred) ** 2
