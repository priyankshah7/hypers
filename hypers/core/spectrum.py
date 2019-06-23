import numpy as np


def map_units(current: list, future: list, tomap: list):
    p = np.polyfit(current, future, 1)
    mapped = np.polyval(p, tomap)
    return mapped


