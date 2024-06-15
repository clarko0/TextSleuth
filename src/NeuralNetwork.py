import time
from cupy import ndarray
import cupy as np
from src.HelperMethods import HelperMethods
from src.DatasetManager import Dataset
from numba import jit
import os
import math

class PropagateForwardsResults:
    Z1: ndarray
    A1: ndarray
    Z2: ndarray
    A2: ndarray
    Z3: ndarray
    A3: ndarray

    def __init__(self, Z1: ndarray, A1: ndarray, Z2: ndarray, A2: ndarray, Z3: ndarray, A3: ndarray) -> None:
        self.Z1 = Z1
        self.A1 = A1
        self.Z2 = Z2
        self.A2 = A2
        self.Z3 = Z3
        self.A3 = A3
        pass

class NeuralNetwork:
    X: ndarray
    Y: ndarray
    dataset_size: int
    inv_dataset_size: float

    W1: ndarray
    B1: ndarray
    W2: ndarray
    B2: ndarray
    W3: ndarray
    B3: ndarray

    Z1: ndarray
    A1: ndarray
    Z2: ndarray
    A2: ndarray
    Z3: ndarray
    A3: ndarray

    d_Z3: ndarray
    d_W3: ndarray
    d_B3: ndarray
    d_Z2: ndarray
    d_W2: ndarray
    d_B2: ndarray
    d_Z1: ndarray
    d_W1: ndarray
    d_B1: ndarray

    neuron_count: int
    learning_rate: float

    def __init__(self, training_dataset: Dataset, neuron_count: int) -> None:
        self.neuron_count = neuron_count
        
        self._init_params()
        self.X = training_dataset.X
        self.Y = training_dataset.Y
        self.dataset_size = len(self.Y)
        self.inv_dataset_size = 1 / self.dataset_size

    def _init_params(self) -> None:
        self.W1 = np.random.rand(self.neuron_count, 784) - 0.5
        self.B1 = np.random.rand(self.neuron_count, 1) - 0.5
        self.W2 = np.random.rand(self.neuron_count, self.neuron_count) - 0.5
        self.B2 = np.random.rand(self.neuron_count, 1) - 0.5
        self.W3 = np.random.rand(10, self.neuron_count) - 0.5
        self.B3 = np.random.rand(10, 1) - 0.5

    def propagate_forwards(self, X: ndarray) -> PropagateForwardsResults:
        Z1 = self.W1.dot(X) + self.B1
        A1 = HelperMethods.ReLu(Z1)

        Z2 = self.W2.dot(A1) + self.B2
        A2 = HelperMethods.ReLu(Z2)

        Z3 = self.W3.dot(A2) + self.B3
        A3 = HelperMethods.softmax(Z3)

        return PropagateForwardsResults(Z1, A1, Z2, A2, Z3, A3)

    def _propagate_forwards(self) -> None:
        results = self.propagate_forwards(self.X)

        self.Z1 = results.Z1
        self.A1 = results.A1

        self.Z2 = results.Z2
        self.A2 = results.A2

        self.Z3 = results.Z3
        self.A3 = results.A3

    def _propagate_backwards(self) -> None:
        one_hot_Y = HelperMethods.one_hot(self.Y)

        self.d_Z3 = self.A3 - one_hot_Y
        self.d_W3 = self.inv_dataset_size * self.d_Z3.dot(self.A2.T)
        self.d_B3 = self.inv_dataset_size * np.sum(self.d_Z3)

        self.d_Z2 = self.W3.T.dot(self.d_Z3) * HelperMethods.ReLU_deriv(self.Z2)
        self.d_W2 = self.inv_dataset_size * self.d_Z2.dot(self.A1.T)
        self.d_B2 = self.inv_dataset_size * np.sum(self.d_Z2)


        self.d_Z1 = self.W2.T.dot(self.d_Z2) * HelperMethods.ReLU_deriv(self.Z1)
        self.d_W1 = self.inv_dataset_size * self.d_Z1.dot(self.X.T)
        self.d_B1 = self.inv_dataset_size * np.sum(self.d_Z1)
    
    def _update_params(self) -> None:
        self.W1 = self.W1 - self.learning_rate * self.d_W1
        self.B1 = self.B1 - self.learning_rate * self.d_B1

        self.W2 = self.W2 - self.learning_rate * self.d_W2
        self.B2 = self.B2 - self.learning_rate * self.d_B2

        self.W3 = self.W3 - self.learning_rate * self.d_W3
        self.B3 = self.B3 - self.learning_rate * self.d_B3

    def _list_current_predictions(self) -> ndarray:
        return np.argmax(self.A3, 0)

    def _calculate_current_accuracy(self) -> float:
        predictions = self._list_current_predictions()
        return np.sum(predictions == self.Y) / self.Y.size
    
    def gradient_descent(self, learning_rate: float, iterations: int, log_progress: bool = False) -> None:
        self.learning_rate = learning_rate
        progress_bar = ""
        bar_size = 30
        for current_iteration in range(iterations):
            start_time = time.time()
            self._propagate_forwards()
            self._propagate_backwards()
            self._update_params()

            if log_progress and current_iteration % 50 == 0:
                progress = (current_iteration + 1) / iterations
                bar_points = math.floor(progress * bar_size)
                accuracy = self._calculate_current_accuracy()
                accuracy = round(float(accuracy), 2)  # Convert accuracy to float before rounding
                progress_bar = (bar_points * "=") + ((bar_size - bar_points) * "_")
                d_time = time.time() - start_time
                d_time = d_time * 1000
                d_time = round(d_time)
                os.system('cls')
                print(f"Accuracy : {accuracy} | [{progress_bar}] | Iteration : {current_iteration + 1} / {iterations} | Latest Iteration Took : {d_time}ms")