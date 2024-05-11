from src.DatasetManager import DatasetManager
from src.NeuralNetwork import NeuralNetwork
from src.NNService import NNService
from src.EngineController import EngineController

def main() -> None:
    dataset_manager = DatasetManager()

    neural_network = NeuralNetwork(dataset_manager.training_data)
    neural_network.gradient_descent(0.1, 250)

    nn_service = NNService(neural_network, dataset_manager.testing_data)
    print(f"Actual NN Accuracy: {nn_service.calculate_accuracy()}")

    app = EngineController(nn_service)



if __name__ == "__main__":
    main()