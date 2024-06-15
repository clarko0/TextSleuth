from src.DatasetManager import DatasetManager
from src.NeuralNetwork import NeuralNetwork
from src.NNService import NNService
from OldEngineController import EngineController
from flask import Flask, render_template

NUMBER_OF_NEURAL_NETWORKS = 1
NEURON_COUNT = 64
LEARNING_RATE = 0.1
ITERATIONS = 6_000

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/get-prediction')
def get_predictions():
    return {}

def main() -> None:
    dataset_manager = DatasetManager()
    neural_networks = [
        NeuralNetwork(dataset_manager.training_data, NEURON_COUNT) 
        for i in range(NUMBER_OF_NEURAL_NETWORKS)
    ]

    for neural_network in neural_networks:
        neural_network.gradient_descent(LEARNING_RATE, ITERATIONS, True)

    # neural_network = NeuralNetwork(dataset_manager.training_data, NEURON_COUNT)
    # neural_network.gradient_descent(LEARNING_RATE, ITERATIONS, log_progress = True)

    nn_service = NNService(neural_networks, dataset_manager.testing_data)
    print(f"Actual NN Accuracy: {nn_service.calculate_accuracy()}")

    app = EngineController(nn_service)



if __name__ == "__main__":
    app.run(debug=True)
    # main()