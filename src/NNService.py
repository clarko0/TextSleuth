from src.NeuralNetwork import NeuralNetwork
from src.DatasetManager import Dataset
import cupy as np
from cupy import ndarray

LEARNING_RATE = 0.1
ITERATIONS = 6_000

class NNService:

    neural_networks: list[NeuralNetwork]
    testing_data: Dataset
    
    def make_predictions(self, X: ndarray) -> ndarray:
        results = np.zeros((10, X.shape[1]))

        for neural_network in self.neural_networks:
            prop_result = neural_network.propagate_forwards(X).A3
            results += prop_result

        return np.argmax(results, 0)

    def calculate_accuracy(self) -> float:
        predictions = self.make_predictions(self.testing_data.X)
        return np.sum(predictions == self.testing_data.Y) / self.testing_data.Y.size

    def train_networks(self) -> None:
        for neural_network in self.neural_networks:
            neural_network.gradient_descent(LEARNING_RATE, ITERATIONS, True)        

    def __init__(self, neural_networks: list[NeuralNetwork], testing_data: Dataset) -> None:
        self.neural_networks = neural_networks
        self.testing_data = testing_data
        pass