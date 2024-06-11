import numpy as np
from PIL import Image
from collections import defaultdict


class RGBMatrix:
    def __init__(self, data: defaultdict) -> None:
        self.data = data

    def create_matrix(self):
        rows = set()
        columns = set()
        values = {}
        # Parse and collect all unique row and column names, and extract quantiles
        for key, value_list in self.data.items():
            row, column = key.split("|*")
            rows.add(row)
            columns.add(column)
            x1, x2, x3 = np.quantile(value_list, [0.25, 0.5, 0.75])
            values[(row, column)] = (x1, x2, x3)
        # Convert sets to list to have a fixed order and create index mappings
        rows = sorted(list(rows))
        columns = sorted(list(columns))
        # Initialize a 3D matrix for RGB values
        # ? Why do the colors lessen when removing the dtype? I would think without the number trying to be restricted by a uint8 cast, we would get more colors, but we get less
        rgb_matrix = np.zeros((len(rows), len(columns), 3), dtype=np.uint8)

        # Populate the 3D matrix
        for (row, column), (r, g, b) in values.items():
            row_index = rows.index(row)
            column_index = columns.index(column)
            print((r, g, b))
            # print((row_index, column_index))
            rgb_matrix[row_index, column_index, 0] = r
            rgb_matrix[row_index, column_index, 1] = g
            rgb_matrix[row_index, column_index, 2] = b

        return rgb_matrix

    def visualize_matrix(self, matrix, filename):
        image = Image.fromarray(matrix, "RGB")
        image.save(filename)
        return image


def validate_image_pixels(matrix, filename):
    """
    Validates that the pixels in the saved image have the correct RGB values as in the provided matrix.

    Parameters:
    - matrix (np.ndarray): The original RGB matrix used to create the image.
    - filename (str): The path to the saved image file.

    Returns:
    - bool: True if all pixels match, False otherwise.
    """
    # Load the image from disk
    with Image.open(filename) as img:
        # Convert the image to an array
        image_array = np.array(img)

    # Check if the shapes are the same
    if image_array.shape != matrix.shape:
        print("Shape mismatch:", image_array.shape, "expected", matrix.shape)
        return False

    # Check each pixel
    mismatch_count = np.sum(image_array != matrix)
    if mismatch_count == 0:
        print("All pixels match exactly.")
        return True
    else:
        print(f"Mismatched pixels count: {mismatch_count}")
        raise ValueError("PIXELS DO NOT MATCH")
