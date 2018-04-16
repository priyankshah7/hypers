import os
import h5py


def multibead():
    path = os.path.dirname(os.path.realpath(__file__)) + '\\multibead_cars_data.h5'

    with h5py.File(path, 'r') as file:
        data_output = file['data']['multibead_data'].value

    return data_output
