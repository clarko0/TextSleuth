from cupy import ndarray
import cupy as np

class HelperMethods:

    @staticmethod # type: ignore
    def ReLu(x: ndarray) -> ndarray:
        return np.maximum(x, 0)
    
    @staticmethod # type: ignore
    def softmax(x: ndarray) -> ndarray:
        result = np.exp(x) / sum(np.exp(x))
        return result

    @staticmethod # type: ignore
    def ReLU_deriv(x: ndarray) -> ndarray:
        return x > 0
    
    @staticmethod # type: ignore
    def one_hot(x: ndarray) -> ndarray:
        one_hot_X = np.zeros((x.size, int(x.max()) + 1))
        one_hot_X[np.arange(x.size), x] = 1
        return one_hot_X.T