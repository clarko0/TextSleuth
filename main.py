from src.DatasetManager import DatasetManager
from src.NeuralNetwork import NeuralNetwork
from src.ImageManager import ImageManager
from src.NNService import NNService
import cupy as np
import flask

app = flask.Flask(__name__)

@app.route("/", methods=['GET'])
def index() -> str:
    return flask.render_template("index.html")

@app.route("/calculate-predictions", methods=['POST'])
def calculate_predictions() -> flask.Response:
    result = flask.request.json
    response = []
    if result:
        images = image_manager.isolate_digits(result['image'])
        for image in images:
            prediction = nn_service.make_predictions(np.asarray([image]).T)[0]
            response.append(int(prediction))
    return flask.jsonify(response)

NUMBER_OF_NEURAL_NETWORKS = 1
NEURON_COUNT = 64

def main() -> None:
    global image_manager, nn_service
    dataset_manager = DatasetManager()

    neural_networks = [
        NeuralNetwork(dataset_manager.training_data, NEURON_COUNT) 
        for i in range(NUMBER_OF_NEURAL_NETWORKS)
    ]

    nn_service = NNService(neural_networks, dataset_manager.testing_data)

    image_manager = ImageManager()
    app.run()

    # nn_service.train_networks()

    # image_manager = ImageManager()
    # app.run()

if __name__ == "__main__":
    main()