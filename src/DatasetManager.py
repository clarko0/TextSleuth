from numpy import ndarray
from keras.datasets import mnist

class Dataset:
    X: ndarray
    Y: ndarray

    def __init__(self, x: ndarray, y: ndarray) -> None:
        self.X = x
        self.Y = y
        pass

class DatasetManager:
    training_data: Dataset
    testing_data: Dataset

    def _load_datasets(self) -> None:
        (training_x, training_y), (testing_x, testing_y) = mnist.load_data()

        self.training_data = Dataset(self._process_x(training_x), training_y)
        self.testing_data = Dataset(self._process_x(testing_x), testing_y)

    def _process_x(self, x: ndarray) -> ndarray:
        # x = x >= 128
        # x = x.reshape(x.shape[0], 784)
        # x = x.T

        return x

    def __init__(self) -> None:
        self._load_datasets()
        pass