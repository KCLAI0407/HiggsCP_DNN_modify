import numpy as np
from numpy import *

def Pack_Matrices_with_NaN(List_of_matrices, Matrix_size):
    def Pack_Matrices_with_NaN(List_of_matrices, Matrix_size):
        Matrix_with_nan = np.arange(Matrix_size)
        for array in List_of_matrices:
            start_position = len(array[0])
            for x in range(start_position,Matrix_size):
                array = np.insert(array, (x), 1, axis=1)
            Matrix_with_nan = np.vstack([Matrix_with_nan, array])
        Matrix_with_nan = Matrix_with_nan[1:]
        return Matrix_with_nan