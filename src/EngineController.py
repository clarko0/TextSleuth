import tkinter as tk
from src.Constants import Constants
from src.NNService import NNService
import numpy as np

class EngineController:
    CANVAS_SIZE: int = Constants.IMAGE_SIZE * 10

    master: tk.Tk
    canvas: tk.Canvas
    output: tk.Text
    drawn_image: list[float]
    nn_service: NNService

    def handle_draw(self, event: tk.Event) -> None:
        x, y = event.x, event.y

        if x >= self.CANVAS_SIZE or y >= self.CANVAS_SIZE:
            return
        
        x1, y1 = (x // 10) * 10, (y // 10) * 10
        x2, y2 = x1 + 10, y1 + 10

        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="")

        x_index = (x // 10) % Constants.IMAGE_SIZE
        y_index = (y // 10) % Constants.IMAGE_SIZE

        index = (Constants.IMAGE_SIZE * y_index) + x_index

        self.drawn_image[index] = 1

    def calculate_prediction(self) -> None:
        x = np.array([self.drawn_image]).T
        predictions = self.nn_service.make_predictions(x)
        self.output.insert(tk.END, str(predictions[0]))

    def init_app(self) -> None:
        self.master.title("OCR Engine")

        self.canvas = tk.Canvas(self.master, width=Constants.IMAGE_SIZE*10-1, height=Constants.IMAGE_SIZE*10-1, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.handle_draw)

        self.output = tk.Text(self.master, width=24, height=5)
        self.output.pack()

        self.button = tk.Button(self.master, text="Predict Character", command=self.calculate_prediction)
        self.button.pack()

        self.button2 = tk.Button(self.master, text="Clear Canvas", command=self.clear_image)
        self.button2.pack()

    def clear_image(self) -> None:
        self.drawn_image = [0 for i in range(Constants.IMAGE_SIZE * Constants.IMAGE_SIZE)]
        self.canvas.delete("all")
        self.output.delete(1.0, tk.END)

    def __init__(self, nn_service: NNService) -> None:
        self.master = tk.Tk()
        self.init_app()
        self.clear_image()
        self.nn_service = nn_service

        self.master.mainloop()