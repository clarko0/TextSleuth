import cv2
import os
from PIL import Image
import numpy as np
import base64
from cupy import ndarray

class ImageManager:

    def __init__(self) -> None:
        pass

    def isolate_digits(self, b64_image: str) -> ndarray:
        characters = self._find_digits(b64_image)
        result = self._scale_digits(characters)

        return result
        
    def _find_digits(self, b64_image: str) -> list:
        encoded_data = b64_image.split(',')[1]

        nparray = np.frombuffer(base64.b64decode(encoded_data), np.uint8) # type: ignore
        image = cv2.imdecode(nparray, cv2.IMREAD_GRAYSCALE)

        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        characters = []

        for index, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            isolated_character = binary_image[y:y+h, x:x+w]
            isolated_character = cv2.bitwise_not(isolated_character)

            characters.append(isolated_character)
        
        return characters
        
    def _scale_digits(self, characters: list) -> ndarray:

        result = []

        for character in characters:
            img = Image.fromarray(character)

            height_percent = 16 / float(img.size[1])
            new_width = int((float(img.size[1]) * float(height_percent)))
            
            # Resize the image
            img = img.resize((new_width, 16), Image.LANCZOS)
            
            # Create a new 28x28 image with a white background
            new_img = Image.new("RGB", (28, 28), (255, 255, 255))
            
            # Calculate the position to center the resized image
            top_left_x = (28 - img.size[0]) // 2
            top_left_y = (28 - img.size[1]) // 2
            
            # Paste the resized image onto the new white background
            new_img.paste(img, (top_left_x, top_left_y))
            
            np_array = np.asarray(new_img)
            np_array = np.reshape(np_array, (784,3))
            np_array = np.mean(np_array, axis=1)
            np_array = np_array / 255
            np_array = np_array >= 0.5

            result.append(np_array)
        
        return np.asarray(result)