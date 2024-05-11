from src.NeuralNetwork import NeuralNetwork
from src.DatasetManager import Dataset
import numpy as np
from numpy import ndarray

class NNService:

    neural_network: NeuralNetwork
    testing_data: Dataset
    
    def make_predictions(self, X: ndarray) -> ndarray:
        results = self.neural_network.propagate_forwards(X)
        return np.argmax(results.A2, 0)

    def calculate_accuracy(self) -> float:
        predictions = self.make_predictions(self.testing_data.X)
        return np.sum(predictions == self.testing_data.Y) / self.testing_data.Y.size

    def __init__(self, neural_network: NeuralNetwork, testing_data: Dataset) -> None:
        self.neural_network = neural_network
        self.testing_data = testing_data
        pass